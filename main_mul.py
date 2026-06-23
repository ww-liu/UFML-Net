# -*- coding: utf-8 -*-

import warnings
import os
import argparse
import json


def _sanitize_thread_env_var(name, default='1'):
    value = os.environ.get(name)
    if value is None:
        return
    try:
        if int(value) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        os.environ[name] = default


_sanitize_thread_env_var('OMP_NUM_THREADS')
_sanitize_thread_env_var('MKL_NUM_THREADS')

import torch.nn.functional as F
import torch
import time
from torch import nn, optim
try:
    from segmentation_models_pytorch.losses import FocalLoss
except ModuleNotFoundError:
    class FocalLoss(nn.Module):
        def __init__(self, mode='multilabel', alpha=0.75, gamma=2.0, reduction='mean'):
            super().__init__()
            if mode != 'multilabel':
                raise ValueError('Fallback FocalLoss only supports mode="multilabel".')
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits, targets):
            targets = targets.float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            prob = torch.sigmoid(logits)
            p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
            if self.reduction == 'mean':
                return loss.mean()
            if self.reduction == 'sum':
                return loss.sum()
            return loss
from torch.optim.lr_scheduler import MultiStepLR
import utils
from dataset import load_datasets, detect_dataset_type
from config import config
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import numpy as np
import random
import pandas as pd
import atexit
from copy import deepcopy
from datetime import datetime
from models.common import (
    count_parameters,
    extract_logits,
    is_baseline_model,
    is_external_model,
    is_mrm_baseline_model,
    is_ufml_model,
)
from models.evidence_reliability import EvidenceReliabilityHead
from models.registry import build_model

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillKL(nn.Module):
    """KL divergence for distillation."""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)


class OneHotCrossEntropyLoss(nn.Module):
    """Cross entropy for one-hot targets used by the CPSC single-label loader."""

    def __init__(self, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        if class_weights is None:
            self.register_buffer('class_weights', None)
        else:
            self.register_buffer('class_weights', torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, targets):
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=weights,
            label_smoothing=self.label_smoothing,
        )


def is_single_label_task(cfg=None):
    cfg = config if cfg is None else cfg
    return str(getattr(cfg, 'target_mode', 'multilabel')).lower() == 'single_label'


def logits_to_prob(logits, cfg=None):
    cfg = config if cfg is None else cfg
    if str(getattr(cfg, 'target_mode', 'multilabel')).lower() == 'single_label':
        return torch.softmax(logits, dim=1)
    return torch.sigmoid(logits)


def build_criterion(cfg):
    loss_name = str(getattr(cfg, 'loss_name', 'focal')).lower()
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    if loss_name == 'focal':
        return FocalLoss(mode='multilabel', alpha=0.75, gamma=2.0, reduction='mean')
    if loss_name == 'ce':
        return OneHotCrossEntropyLoss(
            label_smoothing=getattr(cfg, 'label_smoothing', 0.0),
            class_weights=getattr(cfg, 'ce_class_weights', None),
        )
    raise ValueError(f'Unsupported loss_name: {loss_name}')


def _apply_gradient_clip(model, cfg):
    clip_norm = float(getattr(cfg, 'gradient_clip_norm', 0.0) or 0.0)
    if clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)


def _should_skip_nonfinite_loss(loss, context):
    if torch.isfinite(loss):
        return False
    message = f'Warning: non-finite loss detected in {context}; skipping this batch.'
    if getattr(config, 'skip_nonfinite_batches', True):
        print(message)
        return True
    raise FloatingPointError(message)


def _sanitize_prob_tensor(prob, context):
    if torch.isfinite(prob).all():
        return prob
    print(f'Warning: non-finite probabilities detected in {context}; replacing NaN/Inf for metric computation.')
    return torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


class ModelEma(nn.Module):
    # Exponential moving average for late-stage evaluation stability.
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.module = self._clone_model(model)
        self.decay = decay

    @staticmethod
    def _clone_model(model):
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model

    def update(self, model):
        with torch.no_grad():
            model_state = model.state_dict()
            for name, ema_v in self.module.state_dict().items():
                model_v = model_state[name].detach()
                if not torch.is_floating_point(ema_v):
                    ema_v.copy_(model_v)
                else:
                    ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)


class ModelAverage(nn.Module):
    # Simple checkpoint averaging used as lightweight SWA.
    def __init__(self, model):
        super().__init__()
        self.module = ModelEma._clone_model(model)
        self.num_models = 0

    def update(self, model):
        with torch.no_grad():
            model_state = model.state_dict()
            self.num_models += 1
            for name, avg_v in self.module.state_dict().items():
                model_v = model_state[name].detach()
                if not torch.is_floating_point(avg_v):
                    avg_v.copy_(model_v)
                else:
                    avg_v.mul_((self.num_models - 1.0) / self.num_models).add_(model_v / self.num_models)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, model, optimizer):
    if os.path.isfile(filepath):
        print("Loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model
    print("No checkpoint found at '{}'".format(filepath))
    return None


def load_lcdr_pretrained_weights(model, cfg, dataset_tag):
    if not getattr(cfg, 'use_lcdr_pretrained', False):
        return

    pretrain_path = getattr(cfg, 'lcdr_pretrained_path', '')
    if not pretrain_path:
        pretrain_path = os.path.join(
            'checkpoints',
            f'{cfg.model_name}_{dataset_tag}_{get_tagged_experiment_name(cfg.experiment)}_lcdr_pretrain.pth'
        )

    if not os.path.isfile(pretrain_path):
        print(f'LCDR pretrained checkpoint not found, skip loading: {pretrain_path}')
        return

    checkpoint = torch.load(pretrain_path, map_location=device)
    pretrained_state = checkpoint.get('model_state_dict', checkpoint)
    target_model = model.module if hasattr(model, 'module') else model
    current_state = target_model.state_dict()
    matched_state = {}
    skipped = []
    filtered = []
    load_mode = getattr(cfg, 'lcdr_pretrained_load_mode', 'shallow')
    allowed_prefixes = tuple(getattr(
        cfg,
        'lcdr_pretrained_prefixes',
        ['shape_enhance', 'lead_calibration', 'lead_relation_graph', 'BaseBlock']
    ))

    for key, value in pretrained_state.items():
        if load_mode != 'all' and not key.startswith(allowed_prefixes):
            filtered.append(key)
            continue
        if key in current_state and current_state[key].shape == value.shape:
            matched_state[key] = value
        else:
            skipped.append(key)

    current_state.update(matched_state)
    target_model.load_state_dict(current_state)
    print(
        f'Loaded LCDR pretrained weights from {pretrain_path}: '
        f'{len(matched_state)} tensors loaded, {len(skipped)} skipped, '
        f'{len(filtered)} filtered by mode={load_mode}.'
    )


def load_eval_models_from_checkpoint(base_model, checkpoint):
    if 'ensemble_state_dicts' not in checkpoint:
        if hasattr(base_model, 'module'):
            base_model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            base_model.load_state_dict(checkpoint['model_state_dict'])
        return [base_model]

    eval_models = load_eval_models_from_state_dicts(base_model, checkpoint['ensemble_state_dicts'])
    print(f'Loaded best validation ensemble for final test: {len(eval_models)} models')
    return eval_models


def get_result_dataset_tag(datafolder):
    dataset_type = detect_dataset_type(datafolder)
    if dataset_type == 'ptbxl':
        return 'ptbxl'
    if dataset_type == 'cpsc':
        return 'cpsc'
    if dataset_type == 'georgia':
        return 'georgia'
    return os.path.basename(os.path.abspath(datafolder)) or 'hf'


def get_artifact_dataset_tag(datafolder):
    dataset_type = detect_dataset_type(datafolder)
    if dataset_type == 'ptbxl':
        return 'ptbxl'
    if dataset_type == 'cpsc':
        return 'cpsc'
    if dataset_type == 'georgia':
        return 'georgia'
    return os.path.basename(os.path.abspath(datafolder)) or 'hf'


def apply_dataset_overrides(cfg, dataset_tag):
    if dataset_tag == 'georgia':
        applied = {}
        for key, value in getattr(cfg, 'georgia_overrides', {}).items():
            setattr(cfg, key, value)
            applied[key] = value
        applied['georgia_min_label_count'] = getattr(cfg, 'georgia_min_label_count', 100)
        return applied

    if dataset_tag != 'cpsc':
        return {}
    applied = {}
    for key, value in getattr(cfg, 'cpsc_overrides', {}).items():
        setattr(cfg, key, value)
        applied[key] = value
    recipe_name = getattr(cfg, 'cpsc_recipe', 'stable')
    recipe_overrides = getattr(cfg, 'cpsc_recipe_overrides', {}).get(recipe_name, {})
    for key, value in recipe_overrides.items():
        setattr(cfg, key, value)
        applied[key] = value
    applied['cpsc_recipe'] = recipe_name
    return applied


def apply_experiment_overrides(cfg, dataset_tag=None):
    applied = {}
    overrides = getattr(cfg, 'experiment_overrides', {}).get(cfg.experiment, {})
    allowed_datasets = overrides.get('_datasets')
    if allowed_datasets is not None and dataset_tag not in allowed_datasets:
        return applied
    for key, value in overrides.items():
        if key.startswith('_'):
            continue
        setattr(cfg, key, value)
        applied[key] = value
    return applied


def apply_model_overrides(cfg):
    model_name = str(getattr(cfg, 'model_name', 'ufml_net')).lower()
    if is_ufml_model(model_name) or model_name == 'mynet6view':
        return {}
    if is_mrm_baseline_model(model_name):
        applied = {}
        overrides = getattr(cfg, 'mrm_baseline_overrides', {})
        if not overrides:
            overrides = {
                'mutual_feat_weight': 0.0,
                'mutual_logit_weight': 0.0,
                'use_frequency_aux': False,
                'use_multiband_frequency_aux': False,
                'use_classwise_frequency_aux': False,
                'use_tf_branch': False,
                'use_periodic_aux_branch': False,
                'use_category_fusion': False,
                'use_lcd_regularization': False,
                'use_fdconv_scale': False,
                'use_ema': False,
                'use_swa': False,
                'final_eval_mode': 'checkpoint',
                'run_note': 'mrm_baseline_clean',
            }
        for key, value in overrides.items():
            setattr(cfg, key, value)
            applied[key] = value
        return applied
    if is_external_model(model_name):
        return {}
    applied = {}
    for key, value in getattr(cfg, 'comparison_model_overrides', {}).items():
        setattr(cfg, key, value)
        applied[key] = value
    model_tag = ''.join(
        ch if ch.isalnum() or ch in ('-', '_') else '_'
        for ch in str(getattr(cfg, 'model_name', 'baseline')).lower()
    )
    cfg.run_tag = f'baseline_{model_tag}'
    applied['run_tag'] = cfg.run_tag
    return applied


def reapply_cli_overrides(cfg):
    applied = {}
    for key, value in getattr(cfg, '_cli_overrides', {}).items():
        setattr(cfg, key, value)
        if key == 'dataset':
            cfg.dataset_name = value
        applied[key] = value
    return applied


def get_active_modules_summary(cfg):
    module_flags = [
        ('UIML', getattr(cfg, 'mutual_feat_weight', 0.0) > 0 or getattr(cfg, 'mutual_logit_weight', 0.0) > 0),
        ('FreqAux', getattr(cfg, 'use_frequency_aux', False)),
        ('MultiBandFreqAux', getattr(cfg, 'use_multiband_frequency_aux', False)),
        ('FDConvScale', getattr(cfg, 'use_fdconv_scale', False)),
        ('EMA', getattr(cfg, 'use_ema', False)),
        ('SWA', getattr(cfg, 'use_swa', False)),
        ('Shape', getattr(cfg, 'use_shape_enhance', False)),
        ('LeadCalib', getattr(cfg, 'use_lead_calibration', False)),
        ('LeadGraph', getattr(cfg, 'use_lead_relation_graph', False)),
        ('LCDR', getattr(cfg, 'use_lcd_regularization', False)),
        ('LCDRPretrain', getattr(cfg, 'use_lcdr_pretrained', False)),
        ('FreqGate', getattr(cfg, 'use_frequency_gate', False)),
        ('ATWA', getattr(cfg, 'use_adaptive_temporal_window', False)),
        ('MultiRange', getattr(cfg, 'use_multirange_reweight', False)),
        ('TFBranch', getattr(cfg, 'use_tf_branch', False)),
        ('PeriodicAux', getattr(cfg, 'use_periodic_aux_branch', False)),
        ('CategoryFusion', getattr(cfg, 'use_category_fusion', False)),
        ('FDAug', getattr(cfg, 'use_fda_augmentation', False)),
        ('SelectiveML', getattr(cfg, 'use_selective_mutual_learning', False)),
        ('AUCRankLoss', getattr(cfg, 'use_auc_ranking_loss', False)),
        ('BranchComp', getattr(cfg, 'use_branch_complement_loss', False)),
        ('EvidenceReliability', str(getattr(cfg, 'reliability_mode', 'entropy')).lower() in ('evidence', 'hybrid')),
        ('EvidenceFusion', str(getattr(cfg, 'fusion_mode', 'standard')).lower() == 'evidence_opinion'),
        ('EvidenceCons', getattr(cfg, 'use_evidence_consistency', False)),
        ('FusionF1Boost', getattr(cfg, 'use_fusion_f1_boost', False)),
        ('ValThreshold', getattr(cfg, 'use_val_threshold_optimization', False)),
        ('ThresholdReport', getattr(cfg, 'report_val_threshold_metrics', False)),
    ]
    active = [name for name, enabled in module_flags if enabled]
    return '+'.join(active) if active else 'baseline'


def append_dataframe_safely(df, csv_path):
    if os.path.exists(csv_path):
        try:
            existing_header = pd.read_csv(csv_path, nrows=0)
            if list(existing_header.columns) != list(df.columns):
                root, ext = os.path.splitext(csv_path)
                csv_path = f'{root}_v2{ext}'
        except Exception:
            root, ext = os.path.splitext(csv_path)
            csv_path = f'{root}_v2{ext}'
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)
    return csv_path, write_header


def get_results_dir():
    return getattr(config, 'result_dir', 'results')


def get_checkpoints_dir():
    return getattr(config, 'save_dir', 'checkpoints')


def get_run_tag():
    tag = str(getattr(config, 'run_tag', '') or '').strip()
    if not tag:
        return ''
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in tag)


def get_canonical_result_dataset_tag(dataset_tag):
    """Keep best-result CSVs grouped by dataset, not by run tag or folder name."""
    name = str(dataset_tag or 'dataset').lower()
    if 'ptb' in name:
        return 'ptbxl'
    if 'cpsc' in name or 'tmp_data' in name:
        return 'cpsc'
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name) or 'dataset'


def get_safe_filename_token(value):
    token = str(value or 'unknown').strip()
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in token) or 'unknown'


def get_tagged_experiment_name(experiment):
    tag = get_run_tag()
    return f'{experiment}_{tag}' if tag else experiment


def get_tagged_dataset_name(dataset_tag):
    tag = get_run_tag()
    return f'{dataset_tag}_{tag}' if tag else dataset_tag


def get_run_dir(dataset_tag):
    dataset_tag = get_canonical_result_dataset_tag(dataset_tag)
    run_name = (
        str(getattr(config, 'run_name', '') or '').strip() or
        get_run_tag() or
        f"{get_safe_filename_token(getattr(config, 'model_name', 'model'))}_"
        f"{get_safe_filename_token(getattr(config, 'experiment', 'exp'))}_"
        f"seed{getattr(config, 'seed', 0)}"
    )
    return os.path.join(get_results_dir(), dataset_tag, get_safe_filename_token(run_name))


def ensure_run_dir(dataset_tag):
    run_dir = get_run_dir(dataset_tag)
    os.makedirs(run_dir, exist_ok=True)
    config.run_dir = run_dir
    return run_dir


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_run_config_json(run_dir, dataset_tag):
    payload = {
        key: _json_safe(value)
        for key, value in config.__dict__.items()
        if not key.startswith('_') and key != 'latest_predictions'
    }
    payload['dataset'] = get_canonical_result_dataset_tag(dataset_tag)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as config_file:
        json.dump(payload, config_file, indent=2, ensure_ascii=False)


def save_run_final_metrics(run_dir, run_id, experiment, model_name, seed, best_epoch,
                           bauc1, btpr1, bacc1, bf11,
                           bauc2, btpr2, bacc2, bf12,
                           baucf, btprf, baccf, bf1f,
                           dataset_tag='dataset'):
    result_data = {
        'run_id': [run_id],
        'dataset': [get_canonical_result_dataset_tag(dataset_tag)],
        'experiment': [experiment],
        'run_name': [getattr(config, 'run_name', '') or get_run_tag()],
        'model_name': [model_name],
        'seed': [seed],
        'best_epoch': [best_epoch],
        'selection_score_mode': [getattr(config, 'selection_score_mode', 'single_auc')],
        'training_selection_mode': [getattr(config, 'training_selection_mode', 'legacy_test')],
        'run_tag': [get_run_tag()],
        'active_modules': [get_active_modules_summary(config)],
        'run_note': [getattr(config, 'run_note', '')],
        'branch1_auc': [bauc1],
        'branch1_tpr': [btpr1],
        'branch1_acc': [bacc1],
        'branch1_f1': [bf11],
        'branch2_auc': [bauc2],
        'branch2_tpr': [btpr2],
        'branch2_acc': [bacc2],
        'branch2_f1': [bf12],
        'fusion_auc': [baucf],
        'fusion_tpr': [btprf],
        'fusion_acc': [baccf],
        'fusion_f1': [bf1f],
        'num_params': [getattr(config, 'num_params', '')],
        'checkpoint_path': [getattr(config, 'last_checkpoint_path', '')],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    pd.DataFrame(result_data).to_csv(os.path.join(run_dir, 'final_metrics.csv'), index=False)


def get_run_lock_path(dataset_tag, experiment):
    return os.path.join(get_results_dir(), f'.running_{dataset_tag}_{get_tagged_experiment_name(experiment)}.lock')


def acquire_run_lock(dataset_tag, experiment):
    if getattr(config, 'allow_parallel_same_experiment', False):
        return None
    results_dir = get_results_dir()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    lock_path = get_run_lock_path(dataset_tag, experiment)
    if os.path.exists(lock_path):
        stale_hours = getattr(config, 'run_lock_stale_hours', 12)
        age_hours = (time.time() - os.path.getmtime(lock_path)) / 3600.0
        if age_hours <= stale_hours:
            with open(lock_path, 'r', encoding='utf-8', errors='ignore') as lock_file:
                lock_info = lock_file.read().strip()
            raise RuntimeError(
                'Another run seems to be writing the same experiment log. '
                f'Lock file: {lock_path}. Info: {lock_info}. '
                'If this is a stale lock, delete it or set allow_parallel_same_experiment=True.'
            )
        os.remove(lock_path)
    with open(lock_path, 'w', encoding='utf-8') as lock_file:
        lock_file.write(
            f'pid={os.getpid()}, dataset={dataset_tag}, experiment={experiment}, '
            f'time={datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        )
    atexit.register(release_run_lock, lock_path)
    return lock_path


def release_run_lock(lock_path):
    if lock_path and os.path.exists(lock_path):
        os.remove(lock_path)


def _state_dict_from_model(model):
    if hasattr(model, 'module'):
        return model.module.state_dict()
    return model.state_dict()


def _cpu_state_dict_from_model(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in _state_dict_from_model(model).items()
    }


def load_eval_models_from_state_dicts(base_model, state_dicts):
    eval_models = []
    for state_dict in state_dicts:
        eval_model = deepcopy(base_model)
        if hasattr(eval_model, 'module'):
            eval_model.module.load_state_dict(state_dict)
        else:
            eval_model.load_state_dict(state_dict)
        eval_model = eval_model.to(device)
        eval_model.eval()
        eval_models.append(eval_model)
    return eval_models


def configure_class_weighted_ce(cfg, train_dataloader):
    if not (
        is_single_label_task(cfg) and
        str(getattr(cfg, 'loss_name', '')).lower() in ('ce', 'cross_entropy', 'crossentropy') and
        getattr(cfg, 'use_class_weighted_ce', False)
    ):
        cfg.ce_class_weights = None
        return None

    cls_num_list = getattr(getattr(train_dataloader, 'dataset', None), 'cls_num_list', None)
    if cls_num_list is None:
        cfg.ce_class_weights = None
        return None

    counts = np.asarray(cls_num_list, dtype=np.float32)
    counts[counts <= 0] = 1.0
    weights = 1.0 / np.sqrt(counts)
    weights = weights / np.mean(weights)
    cfg.ce_class_weights = weights.astype(np.float32).tolist()
    return cfg.ce_class_weights


def save_checkpoint(best_auc, model, optimizer, epoch, dataset_tag='dataset', ensemble_models=None):
    print('Model Saving...')
    model_state_dict = _state_dict_from_model(model)
    checkpoint_data = {
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    }
    save_ensemble = getattr(config, 'save_ensemble_checkpoints', False)
    if save_ensemble and ensemble_models is not None and len(ensemble_models) > 1:
        checkpoint_data['ensemble_state_dicts'] = [
            _state_dict_from_model(ensemble_model)
            for ensemble_model in ensemble_models
        ]
        checkpoint_data['ensemble_size'] = len(ensemble_models)

    checkpoint_name = getattr(config, 'checkpoints', None)
    if not checkpoint_name:
        checkpoint_name = (
            f'{get_safe_filename_token(config.model_name)}_'
            f'{get_safe_filename_token(dataset_tag)}_'
            f'{get_safe_filename_token(get_tagged_experiment_name(config.experiment))}_'
            f'seed{config.seed}_best.pth'
        )
        config.checkpoints = checkpoint_name
    checkpoint_path = os.path.join(get_checkpoints_dir(), checkpoint_name)
    os.makedirs(get_checkpoints_dir(), exist_ok=True)
    tmp_path = checkpoint_path + '.tmp'
    try:
        torch.save(checkpoint_data, tmp_path)
        os.replace(tmp_path, checkpoint_path)
        config.last_checkpoint_path = checkpoint_path
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        # If disk is tight, retry with the smallest useful checkpoint. The
        # in-memory best_eval_state_dicts still supports ensemble final testing.
        light_checkpoint = {
            'model_state_dict': model_state_dict,
            'global_epoch': epoch,
            'best_auc': best_auc,
        }
        torch.save(light_checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)
        config.last_checkpoint_path = checkpoint_path
        print('Checkpoint saved without ensemble states after a write retry.')


def get_uncertainty_weight(logits, min_weight=0.1, reduce=True):
    if is_single_label_task():
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)
        entropy = entropy / np.log(max(2, logits.size(1)))
        certainty = (1.0 - entropy).clamp(0.0, 1.0)
        certainty = certainty * (1.0 - min_weight) + min_weight
        if reduce:
            return certainty
        return certainty.unsqueeze(1).expand_as(logits)

    probs = torch.sigmoid(logits)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6)) +
                (1.0 - probs) * torch.log((1.0 - probs).clamp_min(1e-6)))
    entropy = entropy / np.log(2.0)
    if reduce:
        entropy = entropy.mean(dim=1)
    certainty = (1.0 - entropy).clamp(0.0, 1.0)
    return certainty * (1.0 - min_weight) + min_weight


def get_evidence_opinion(logits):
    mode = 'dirichlet' if is_single_label_task() else 'beta'
    head = EvidenceReliabilityHead(logits.size(1), mode=mode).to(logits.device)
    return head(logits)


def get_evidence_reliability_weight(logits, reduce=True):
    opinion = get_evidence_opinion(logits)
    reliability = opinion['reliability']
    if reduce:
        return reliability.mean(dim=1)
    if is_single_label_task():
        return reliability.expand_as(logits)
    return reliability


def get_teacher_reliability_weight(logits, min_weight=0.1, reduce=True):
    mode = str(getattr(config, 'reliability_mode', 'entropy')).lower()
    entropy_weight = None
    if mode in ('entropy', 'hybrid'):
        entropy_weight = get_uncertainty_weight(logits, min_weight=min_weight, reduce=reduce)
    if mode == 'entropy':
        return entropy_weight
    evidence_weight = get_evidence_reliability_weight(logits, reduce=reduce)
    if mode == 'evidence':
        return evidence_weight
    if mode == 'hybrid':
        return 0.5 * entropy_weight + 0.5 * evidence_weight
    raise ValueError(f'Unsupported reliability_mode: {mode}')


def evidence_consistency_loss(logits1, logits2):
    opinion1 = get_evidence_opinion(logits1)
    opinion2 = get_evidence_opinion(logits2)
    p1 = opinion1['prob']
    p2 = opinion2['prob']
    r1 = opinion1['reliability']
    r2 = opinion2['reliability']
    if is_single_label_task():
        r1 = r1.expand_as(p1)
        r2 = r2.expand_as(p2)
    return (torch.abs(p1 - p2) * r1 * r2).mean()


def weighted_feature_kd(student_feat, teacher_feat, teacher_weight, temperature):
    student_log_prob = F.log_softmax(student_feat / temperature, dim=1)
    teacher_prob = F.softmax(teacher_feat.detach() / temperature, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=1)
    return (loss * teacher_weight.detach()).mean() * (temperature ** 2)


def weighted_logit_kd(student_logits, teacher_logits, teacher_weight, temperature):
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=1)
    return (loss * teacher_weight.detach()).mean() * (temperature ** 2)


def branch_complementarity_loss(logits1, logits2):
    if logits1.size(1) <= 1:
        return logits1.new_tensor(0.0)
    centered1 = logits1 - logits1.mean(dim=1, keepdim=True)
    centered2 = logits2 - logits2.mean(dim=1, keepdim=True)
    norm1 = F.normalize(centered1, p=2, dim=1)
    norm2 = F.normalize(centered2, p=2, dim=1)
    cosine = torch.sum(norm1 * norm2, dim=1)
    return cosine.pow(2).mean()


def get_rampup_weight(epoch, total_epochs):
    if total_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(total_epochs))


def _dirichlet_opinion_aggregation(opinion1, opinion2):
    eps = 1e-6
    u1 = opinion1['uncertainty']
    u2 = opinion2['uncertainty']
    b1 = opinion1['belief']
    b2 = opinion2['belief']
    denom = (u1 + u2 - u1 * u2).clamp_min(eps)
    fused_belief = (b1 * u2 + b2 * u1) / denom
    fused_uncertainty = (u1 * u2) / denom
    base_rate = 1.0 / float(b1.size(1))
    fused_prob = fused_belief + fused_uncertainty * base_rate
    fused_prob = fused_prob.clamp_min(eps)
    return fused_prob / fused_prob.sum(dim=1, keepdim=True).clamp_min(eps)


def fuse_evidence_opinions(out1, out2):
    opinion1 = get_evidence_opinion(out1)
    opinion2 = get_evidence_opinion(out2)
    prob1 = opinion1['prob']
    prob2 = opinion2['prob']
    if is_single_label_task():
        fused_prob = _dirichlet_opinion_aggregation(opinion1, opinion2)
    else:
        reliability1 = opinion1['reliability']
        reliability2 = opinion2['reliability']
        fused_prob = (
            reliability1 * prob1 + reliability2 * prob2
        ) / (reliability1 + reliability2 + 1e-6)
        fused_prob = fused_prob.clamp(1e-6, 1.0 - 1e-6)
    return prob1, prob2, fused_prob


def fuse_branch_predictions(out1, out2, tf_logits=None, category_fusion_logits=None):
    # Per-class uncertainty weighting lets each label prefer a different branch.
    certainty1 = get_uncertainty_weight(out1, min_weight=config.uncertainty_min_weight, reduce=False)
    certainty2 = get_uncertainty_weight(out2, min_weight=config.uncertainty_min_weight, reduce=False)
    weight1 = certainty1.pow(
        config.fusion_conf_power
    ) * config.branch1_fusion_bias
    weight2 = certainty2.pow(
        config.fusion_conf_power
    ) * config.branch2_fusion_bias
    if is_single_label_task():
        prob1 = torch.softmax(out1, dim=1)
        prob2 = torch.softmax(out2, dim=1)
    else:
        prob1 = torch.sigmoid(out1)
        prob2 = torch.sigmoid(out2)
    fusion_mode = getattr(config, 'fusion_mode', 'standard')
    if fusion_mode == 'evidence_opinion':
        return fuse_evidence_opinions(out1, out2)
    if fusion_mode == 'branch1_only':
        return prob1, prob2, prob1
    if fusion_mode == 'branch2_only':
        return prob1, prob2, prob2
    if fusion_mode == 'simple_average':
        fused_prob = 0.5 * (prob1 + prob2)
        if is_single_label_task():
            fused_prob = fused_prob / fused_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return prob1, prob2, fused_prob
    if fusion_mode == 'confidence_select':
        return prob1, prob2, torch.where(weight2 >= weight1, prob2, prob1)

    norm = weight1 + weight2 + 1e-6
    fused_prob = (prob1 * weight1 + prob2 * weight2) / norm
    fused_logit = (out1 * weight1 + out2 * weight2) / norm
    if is_single_label_task():
        fused_logit_prob = torch.softmax(config.fusion_logit_scale * fused_logit, dim=1)
    else:
        fused_logit_prob = torch.sigmoid(config.fusion_logit_scale * fused_logit)
    fused_prob = (
        (1.0 - config.fusion_logit_blend) * fused_prob +
        config.fusion_logit_blend * fused_logit_prob
    )

    stronger_prob = torch.where(weight2 >= weight1, prob2, prob1)
    disagreement = torch.abs(prob2 - prob1)
    positive_conf = torch.clamp(
        (torch.maximum(prob1, prob2) - config.fusion_positive_margin) /
        max(1e-6, 1.0 - config.fusion_positive_margin),
        min=0.0,
        max=1.0
    )
    residual_gate = (disagreement * positive_conf).pow(config.fusion_gate_power)
    fused_prob = fused_prob + config.fusion_residual_weight * residual_gate * (stronger_prob - fused_prob)
    fused_prob = fused_prob.clamp(1e-6, 1.0 - 1e-6)

    if getattr(config, 'use_fusion_f1_boost', False):
        max_prob = torch.maximum(prob1, prob2)
        boost_margin = getattr(config, 'fusion_f1_boost_margin', 0.48)
        boost_gate = torch.clamp(
            (max_prob - boost_margin) / max(1e-6, 1.0 - boost_margin),
            min=0.0,
            max=1.0
        ).pow(getattr(config, 'fusion_f1_boost_power', 2.0))
        fused_prob = fused_prob + getattr(config, 'fusion_f1_boost_weight', 0.03) * boost_gate * (stronger_prob - fused_prob)
        fused_prob = fused_prob.clamp(1e-6, 1.0 - 1e-6)

    if tf_logits is not None:
        if is_single_label_task():
            tf_prob = torch.softmax(tf_logits, dim=1)
        else:
            tf_prob = torch.sigmoid(tf_logits)
        tf_certainty = get_uncertainty_weight(tf_logits, min_weight=config.uncertainty_min_weight, reduce=False)
        tf_disagreement = torch.abs(prob2 - prob1).detach().pow(getattr(config, 'tf_fusion_disagreement_power', 1.0))
        tf_gate = tf_certainty * tf_disagreement
        fused_prob = fused_prob + getattr(config, 'tf_fusion_alpha', 0.25) * tf_gate * (tf_prob - fused_prob)
        fused_prob = fused_prob.clamp(1e-6, 1.0 - 1e-6)

    if category_fusion_logits is not None:
        if is_single_label_task():
            category_prob = torch.softmax(category_fusion_logits, dim=1)
        else:
            category_prob = torch.sigmoid(category_fusion_logits)
        fused_prob = (
            (1.0 - getattr(config, 'category_fusion_alpha', 0.25)) * fused_prob +
            getattr(config, 'category_fusion_alpha', 0.25) * category_prob
        )
        fused_prob = fused_prob.clamp(1e-6, 1.0 - 1e-6)

    return prob1, prob2, fused_prob


def _binarize_predictions(predictions, thresholds=None):
    predictions = np.array(predictions)
    if thresholds is None:
        return (predictions >= 0.5).astype(int)
    thresholds = np.array(thresholds).reshape(1, -1)
    return (predictions >= thresholds).astype(int)


def _compute_thresholded_f1(y_true, y_pred_binary):
    y_true = np.array(y_true).astype(int)
    y_pred_binary = np.array(y_pred_binary).astype(int)
    tp = np.sum((y_true == 1) & (y_pred_binary == 1), axis=1)
    fp = np.sum((y_true == 0) & (y_pred_binary == 1), axis=1)
    fn = np.sum((y_true == 1) & (y_pred_binary == 0), axis=1)
    denom = 2 * tp + fp + fn
    f1_scores = np.divide(
        2 * tp,
        denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=denom > 0
    )
    return float(np.mean(f1_scores))


def _compute_thresholded_acc(y_true, y_pred_binary):
    y_true = np.array(y_true).astype(int)
    y_pred_binary = np.array(y_pred_binary).astype(int)
    return float(np.mean(y_true == y_pred_binary))


def _compute_thresholded_tpr(y_true, y_pred_binary):
    y_true = np.array(y_true).astype(int)
    y_pred_binary = np.array(y_pred_binary).astype(int)
    tp = np.sum((y_true == 1) & (y_pred_binary == 1), axis=1)
    fn = np.sum((y_true == 1) & (y_pred_binary == 0), axis=1)
    denom = tp + fn
    tpr_scores = np.divide(
        tp,
        denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=denom > 0
    )
    return float(np.mean(tpr_scores))


def optimize_class_thresholds(targets, predictions):
    y_true = np.array(targets).astype(int)
    y_pred = np.array(predictions)
    grid = getattr(config, 'threshold_grid', None)
    if not grid:
        grid = np.linspace(0.1, 0.9, 17)
    grid = sorted(set(float(threshold) for threshold in grid) | {0.5})

    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    baseline_score = _compute_thresholded_f1(y_true, _binarize_predictions(y_pred, thresholds))
    best_score = baseline_score

    threshold_mode = getattr(config, 'threshold_optimization_mode', 'classwise')
    if threshold_mode == 'global':
        best_threshold = 0.5
        for threshold in grid:
            candidate = np.full(y_true.shape[1], threshold, dtype=np.float32)
            candidate_score = _compute_thresholded_f1(
                y_true,
                _binarize_predictions(y_pred, candidate)
            )
            if candidate_score > best_score + 1e-8:
                best_score = candidate_score
                best_threshold = threshold

        min_gain = getattr(config, 'threshold_min_f1_gain', 0.0)
        if best_score < baseline_score + min_gain:
            return thresholds
        return np.full(y_true.shape[1], best_threshold, dtype=np.float32)

    # Optimize the same sample-level F1 metric reported in the training log.
    # If no validation gain is found, this safely falls back to the original 0.5 threshold.
    for _ in range(getattr(config, 'threshold_coordinate_passes', 2)):
        improved = False
        for class_idx in range(y_true.shape[1]):
            class_best_threshold = thresholds[class_idx]
            class_best_score = best_score
            for threshold in grid:
                candidate = thresholds.copy()
                candidate[class_idx] = threshold
                candidate_score = _compute_thresholded_f1(
                    y_true,
                    _binarize_predictions(y_pred, candidate)
                )
                if candidate_score > class_best_score + 1e-8:
                    class_best_score = candidate_score
                    class_best_threshold = threshold
            if class_best_score > best_score + 1e-8:
                thresholds[class_idx] = class_best_threshold
                best_score = class_best_score
                improved = True
        if not improved:
            break

    min_gain = getattr(config, 'threshold_min_f1_gain', 0.0)
    if best_score < baseline_score + min_gain:
        return np.full(y_true.shape[1], 0.5, dtype=np.float32)
    return thresholds.astype(np.float32)


def compute_metrics(targets, predictions, thresholds=None):
    targets = np.array(targets)
    predictions = np.array(predictions)
    if not np.isfinite(predictions).all():
        print('Warning: non-finite predictions found before metric computation; replacing NaN/Inf.')
        predictions = np.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0)
    predictions = np.clip(predictions, 0.0, 1.0)

    if is_single_label_task():
        true_labels = np.argmax(targets, axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        auc = roc_auc_score(targets, predictions, average='macro')
        tpr = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        return auc, tpr, acc, f1

    auc = roc_auc_score(targets, predictions)
    if thresholds is None:
        tpr = utils.compute_TPR(targets, predictions.copy())
        acc = utils.compute_ACC(targets, predictions.copy())
        f1 = utils.compute_F1(targets, predictions.copy())
    else:
        pred_binary = _binarize_predictions(predictions, thresholds)
        tpr = _compute_thresholded_tpr(targets, pred_binary)
        acc = _compute_thresholded_acc(targets, pred_binary)
        f1 = _compute_thresholded_f1(targets, pred_binary)
    return auc, tpr, acc, f1


def get_selection_score(auc1, f11, auc2, f12, aucf, f1f):
    mode = getattr(config, 'selection_score_mode', 'single_auc')
    if mode == 'fusion_auc':
        return aucf
    if mode == 'fusion_auc_f1':
        return aucf + f1f
    if mode == 'branch2_auc':
        return auc2
    if mode == 'branch2_auc_f1':
        return auc2 + f12
    if mode == 'branch1_auc':
        return auc1
    if mode == 'branch1_auc_f1':
        return auc1 + f11
    return max(auc1, auc2, aucf)


def collect_outputs(prob1, prob2, probf, target, outputs1, outputs2, outputsf, targets):
    for i in range(len(target)):
        outputs1.append(prob1[i].cpu().detach().numpy())
        outputs2.append(prob2[i].cpu().detach().numpy())
        outputsf.append(probf[i].cpu().detach().numpy())
        targets.append(target[i].cpu().detach().numpy())


def _as_model_list(model_or_models):
    if isinstance(model_or_models, (list, tuple)):
        return list(model_or_models)
    return [model_or_models]


def unpack_model_outputs(outputs):
    if torch.is_tensor(outputs):
        logits = extract_logits(outputs).to(torch.float)
        return logits, logits, logits, logits, None, None
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 6:
            feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits = outputs
            return feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits
        if len(outputs) == 5:
            feat_ins, out_ins, feat_all, out, tf_logits = outputs
            return feat_ins, out_ins, feat_all, out, tf_logits, None
        if len(outputs) == 4:
            feat_ins, out_ins, feat_all, out = outputs
            return feat_ins, out_ins, feat_all, out, None, None
        if len(outputs) == 3 and all(torch.is_tensor(output) for output in outputs):
            out_ins, out, fused = outputs
            return out_ins, out_ins, out, out, None, fused
        logits = extract_logits(outputs).to(torch.float)
        return logits, logits, logits, logits, None, None
    raise ValueError('Unexpected model outputs format')


def _get_georgia_frequency_aux_output(model, inputs):
    module = model.module if hasattr(model, 'module') else model
    backbone = getattr(module, 'backbone', None)
    if backbone is None or getattr(backbone, 'frequency_aux', None) is None:
        return None
    x = inputs.to(device)
    input_channels = getattr(module, 'input_channels', getattr(config, 'input_channels', 12))
    if x.dim() != 3:
        return None
    if x.size(1) == input_channels:
        x = x.permute(0, 2, 1)
    if x.size(2) != input_channels:
        return None
    x = x.permute(0, 2, 1)
    x = backbone.shape_enhance(x)
    x = backbone.lead_relation_graph(x)
    x = backbone.lead_calibration(x)
    return backbone.frequency_aux(x)


def run_georgia_preflight_checks(model, criterion, train_dataloader, num_classes):
    print('Georgia smoke preflight checks:')
    inputs, labels = next(iter(train_dataloader))
    print(f'  dataloader_input_shape: {tuple(inputs.shape)}')
    print(f'  dataloader_label_shape: {tuple(labels.shape)}')
    print(f'  target_mode: {getattr(config, "target_mode", "multilabel")}')
    print(f'  loss: {criterion.__class__.__name__}')
    activation = 'softmax' if is_single_label_task(config) else 'sigmoid'
    print(f'  activation: {activation}')

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.to(device))
        _, out1, _, out2, tf_logits, category_fusion_logits = unpack_model_outputs(outputs)
        freq_out = _get_georgia_frequency_aux_output(model, inputs)
        prob = logits_to_prob(out2, config)
    if was_training:
        model.train()

    print(f'  model_out1_shape: {tuple(out1.shape)}')
    print(f'  model_out2_shape: {tuple(out2.shape)}')
    print(f'  frequency_aux_shape: {tuple(freq_out.shape) if torch.is_tensor(freq_out) else None}')
    print(f'  tf_logits_shape: {tuple(tf_logits.shape) if torch.is_tensor(tf_logits) else None}')
    print(
        f'  category_fusion_shape: '
        f'{tuple(category_fusion_logits.shape) if torch.is_tensor(category_fusion_logits) else None}'
    )
    print(f'  probability_range_after_{activation}: ({float(prob.min()):.6f}, {float(prob.max()):.6f})')

    expected_input_shape = (inputs.size(0), getattr(config, 'signal_length', 1000), getattr(config, 'input_channels', 12))
    expected_label_shape = (labels.size(0), num_classes)
    if tuple(inputs.shape) != expected_input_shape:
        raise ValueError(f'Unexpected Georgia input shape: {tuple(inputs.shape)} != {expected_input_shape}')
    if tuple(labels.shape) != expected_label_shape:
        raise ValueError(f'Unexpected Georgia label shape: {tuple(labels.shape)} != {expected_label_shape}')
    if tuple(out1.shape) != expected_label_shape or tuple(out2.shape) != expected_label_shape:
        raise ValueError(
            f'Unexpected Georgia output shapes: out1={tuple(out1.shape)}, out2={tuple(out2.shape)}, '
            f'expected={expected_label_shape}'
        )
    if getattr(config, 'use_frequency_aux', False):
        if not torch.is_tensor(freq_out) or tuple(freq_out.shape) != expected_label_shape:
            raise ValueError(f'Unexpected Georgia FrequencyAux shape: {None if freq_out is None else tuple(freq_out.shape)}')


def _replicate_single_metrics(auc, tpr, acc, f1):
    """Keep legacy CSV columns compatible for single-output baselines."""
    return auc, tpr, acc, f1, auc, tpr, acc, f1, auc, tpr, acc, f1


def compute_baseline_loss(outputs, labels, criterion):
    logits = extract_logits(outputs).to(torch.float)
    return criterion(logits, labels), logits


def compute_dbfnet_loss(outputs, labels, criterion):
    # DBFNet must use its own outputs only; never apply UFML uncertainty/frequency
    # losses here. If a future official DBFNet adapter returns branch logits, the
    # final tensor is used as the supervised prediction by default.
    logits = extract_logits(outputs).to(torch.float)
    return criterion(logits, labels), logits


def compute_loss(model_name, outputs, labels, criterion, cfg):
    name = str(model_name).lower()
    if is_baseline_model(name):
        return compute_baseline_loss(outputs, labels, criterion)
    if name == 'dbfnet':
        return compute_dbfnet_loss(outputs, labels, criterion)
    if is_external_model(name):
        raise NotImplementedError(f'Loss computation for {model_name} is not implemented.')
    raise NotImplementedError(
        f'Use the UFML/MRM multi-output training path for {model_name}; '
        'do not route it through baseline compute_loss.'
    )


def save_results_to_csv(run_id, experiment, model_name, seed, best_epoch,
                        bauc1, btpr1, bacc1, bf11,
                        bauc2, btpr2, bacc2, bf12,
                        baucf, btprf, baccf, bf1f,
                        dataset_tag='dataset'):
    results_dir = get_results_dir()
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    dataset_tag = get_canonical_result_dataset_tag(dataset_tag)
    csv_path = os.path.join(results_dir, f'results_{dataset_tag}.csv')
    use_safe_append = True

    result_data = {
        'run_id': [run_id],
        'dataset': [dataset_tag],
        'experiment': [experiment],
        'model_name': [model_name],
        'seed': [seed],
        'best_epoch': [best_epoch],
        'selection_score_mode': [getattr(config, 'selection_score_mode', 'single_auc')],
        'training_selection_mode': [getattr(config, 'training_selection_mode', 'legacy_test')],
        'run_tag': [get_run_tag()],
        'active_modules': [get_active_modules_summary(config)],
        'run_note': [getattr(config, 'run_note', '')],
        'branch1_auc': [bauc1],
        'branch1_tpr': [btpr1],
        'branch1_acc': [bacc1],
        'branch1_f1': [bf11],
        'branch2_auc': [bauc2],
        'branch2_tpr': [btpr2],
        'branch2_acc': [bacc2],
        'branch2_f1': [bf12],
        'fusion_auc': [baucf],
        'fusion_tpr': [btprf],
        'fusion_acc': [baccf],
        'fusion_f1': [bf1f],
        'num_params': [getattr(config, 'num_params', '')],
        'checkpoint_path': [getattr(config, 'last_checkpoint_path', '')],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }

    df = pd.DataFrame(result_data)
    if use_safe_append:
        csv_path, write_header = append_dataframe_safely(df, csv_path)
        if write_header:
            print(f'结果已保存到: {csv_path}')
        else:
            print(f'结果已追加到: {csv_path}')
        return

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f'结果已追加到: {csv_path}')
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f'结果已保存到: {csv_path}')


def save_epoch_result_to_csv(run_id, experiment, model_name, seed, epoch,
                             train_auc1, train_tpr1, train_acc1, train_f11,
                             train_auc2, train_tpr2, train_acc2, train_f12,
                             train_aucf, train_tprf, train_accf, train_f1f,
                             val_auc1, val_tpr1, val_acc1, val_f11,
                             val_auc2, val_tpr2, val_acc2, val_f12,
                             val_aucf, val_tprf, val_accf, val_f1f,
                             test_auc1, test_tpr1, test_acc1, test_f11,
                             test_auc2, test_tpr2, test_acc2, test_f12,
                             test_aucf, test_tprf, test_accf, test_f1f,
                             train_loss, val_loss, test_loss,
                             dataset_tag='dataset',
                             csv_file_prefix='training_log'):
    results_dir = get_results_dir()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    dataset_tag = get_canonical_result_dataset_tag(dataset_tag)
    result_data = {
        'run_id': [run_id],
        'dataset': [dataset_tag],
        'experiment': [experiment],
        'model_name': [model_name],
        'seed': [seed],
        'epoch': [epoch],
        'selection_score_mode': [getattr(config, 'selection_score_mode', 'single_auc')],
        'training_selection_mode': [getattr(config, 'training_selection_mode', 'legacy_test')],
        'run_tag': [get_run_tag()],
        'active_modules': [get_active_modules_summary(config)],
        'run_note': [getattr(config, 'run_note', '')],
        'train_branch1_auc': [train_auc1],
        'train_branch1_tpr': [train_tpr1],
        'train_branch1_acc': [train_acc1],
        'train_branch1_f1': [train_f11],
        'train_branch2_auc': [train_auc2],
        'train_branch2_tpr': [train_tpr2],
        'train_branch2_acc': [train_acc2],
        'train_branch2_f1': [train_f12],
        'train_fusion_auc': [train_aucf],
        'train_fusion_tpr': [train_tprf],
        'train_fusion_acc': [train_accf],
        'train_fusion_f1': [train_f1f],
        'val_branch1_auc': [val_auc1],
        'val_branch1_tpr': [val_tpr1],
        'val_branch1_acc': [val_acc1],
        'val_branch1_f1': [val_f11],
        'val_branch2_auc': [val_auc2],
        'val_branch2_tpr': [val_tpr2],
        'val_branch2_acc': [val_acc2],
        'val_branch2_f1': [val_f12],
        'val_fusion_auc': [val_aucf],
        'val_fusion_tpr': [val_tprf],
        'val_fusion_acc': [val_accf],
        'val_fusion_f1': [val_f1f],
        'test_branch1_auc': [test_auc1],
        'test_branch1_tpr': [test_tpr1],
        'test_branch1_acc': [test_acc1],
        'test_branch1_f1': [test_f11],
        'test_branch2_auc': [test_auc2],
        'test_branch2_tpr': [test_tpr2],
        'test_branch2_acc': [test_acc2],
        'test_branch2_f1': [test_f12],
        'test_fusion_auc': [test_aucf],
        'test_fusion_tpr': [test_tprf],
        'test_fusion_acc': [test_accf],
        'test_fusion_f1': [test_f1f],
        'train_loss': [train_loss],
        'val_loss': [val_loss],
        'test_loss': [test_loss],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }

    df = pd.DataFrame(result_data)
    run_tag = get_run_tag()
    tag_part = f"_{get_safe_filename_token(run_tag)}" if run_tag else ""
    csv_filename = (
        f"{csv_file_prefix}_{dataset_tag}_{get_safe_filename_token(model_name)}_"
        f"{get_safe_filename_token(experiment)}{tag_part}_seed{seed}.csv"
    )
    csv_path = os.path.join(results_dir, csv_filename)

    # If an older log file uses a different schema, write to a versioned file
    # instead of silently appending misaligned rows.
    if os.path.exists(csv_path):
        try:
            existing_header = pd.read_csv(csv_path, nrows=0)
            if list(existing_header.columns) != list(df.columns):
                csv_filename = (
                    f"{csv_file_prefix}_v2_{dataset_tag}_{get_safe_filename_token(model_name)}_"
                    f"{get_safe_filename_token(experiment)}{tag_part}_seed{seed}.csv"
                )
                csv_path = os.path.join(results_dir, csv_filename)
        except Exception:
            csv_filename = (
                f"{csv_file_prefix}_v2_{dataset_tag}_{get_safe_filename_token(model_name)}_"
                f"{get_safe_filename_token(experiment)}{tag_part}_seed{seed}.csv"
            )
            csv_path = os.path.join(results_dir, csv_filename)

    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)
    if write_header:
        print(f'Detailed log file created for experiment "{experiment}": {csv_path}')


def save_final_test_to_csv(run_id, experiment, model_name, seed, best_epoch,
                           bauc1, btpr1, bacc1, bf11,
                           bauc2, btpr2, bacc2, bf12,
                           baucf, btprf, baccf, bf1f,
                           dataset_tag='dataset'):
    results_dir = get_results_dir()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    dataset_tag = get_canonical_result_dataset_tag(dataset_tag)
    csv_path = os.path.join(results_dir, f'final_test_{dataset_tag}.csv')
    use_safe_append = True
    result_data = {
        'run_id': [run_id],
        'dataset': [dataset_tag],
        'experiment': [experiment],
        'model_name': [model_name],
        'seed': [seed],
        'best_epoch': [best_epoch],
        'selection_score_mode': [getattr(config, 'selection_score_mode', 'single_auc')],
        'training_selection_mode': [getattr(config, 'training_selection_mode', 'legacy_test')],
        'run_tag': [get_run_tag()],
        'active_modules': [get_active_modules_summary(config)],
        'run_note': [getattr(config, 'run_note', '')],
        'branch1_auc': [bauc1],
        'branch1_tpr': [btpr1],
        'branch1_acc': [bacc1],
        'branch1_f1': [bf11],
        'branch2_auc': [bauc2],
        'branch2_tpr': [btpr2],
        'branch2_acc': [bacc2],
        'branch2_f1': [bf12],
        'fusion_auc': [baucf],
        'fusion_tpr': [btprf],
        'fusion_acc': [baccf],
        'fusion_f1': [bf1f],
        'num_params': [getattr(config, 'num_params', '')],
        'checkpoint_path': [getattr(config, 'last_checkpoint_path', '')],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }

    df = pd.DataFrame(result_data)
    if use_safe_append:
        csv_path, write_header = append_dataframe_safely(df, csv_path)
        if write_header:
            print(f'结果已保存到: {csv_path}')
        else:
            print(f'结果已追加到: {csv_path}')
        return
    csv_path, write_header = append_dataframe_safely(df, csv_path)
    if write_header:
        print(f'结果已保存到: {csv_path}')
    else:
        print(f'结果已追加到: {csv_path}')
    return
    append_dataframe_safely(df, csv_path)


def save_thresholds_to_csv(thresholds, experiment, dataset_tag='dataset', best_epoch=None):
    if thresholds is None:
        return

    results_dir = get_results_dir()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    num_classes = len(thresholds['fusion'])
    threshold_df = pd.DataFrame({
        'class': [f'class_{i}' for i in range(num_classes)],
        'branch1_threshold': thresholds['branch1'],
        'branch2_threshold': thresholds['branch2'],
        'fusion_threshold': thresholds['fusion'],
        'best_epoch': [best_epoch] * num_classes,
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * num_classes,
    })
    csv_path = os.path.join(results_dir, f'thresholds_{dataset_tag}_{get_tagged_experiment_name(experiment)}.csv')
    threshold_df.to_csv(csv_path, index=False)
    print(f'验证集类别阈值已保存到: {csv_path}')


def save_threshold_report_to_csv(run_id, experiment, model_name, seed, best_epoch,
                                 threshold_report, dataset_tag='dataset'):
    if threshold_report is None:
        return

    results_dir = get_results_dir()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    thresholds = threshold_report.get('thresholds', {})
    metrics = threshold_report.get('test_metrics', {})
    nan_metrics = (np.nan, np.nan, np.nan, np.nan)
    result_data = {
        'run_id': [run_id],
        'dataset': [dataset_tag],
        'experiment': [experiment],
        'model_name': [model_name],
        'seed': [seed],
        'best_epoch': [best_epoch],
        'selection_score_mode': [getattr(config, 'selection_score_mode', 'single_auc')],
        'training_selection_mode': [getattr(config, 'training_selection_mode', 'legacy_test')],
        'active_modules': [get_active_modules_summary(config)],
        'run_note': [getattr(config, 'run_note', '')],
        'branch1_threshold_mean': [float(np.mean(thresholds.get('branch1'))) if thresholds.get('branch1') is not None else np.nan],
        'branch2_threshold_mean': [float(np.mean(thresholds.get('branch2'))) if thresholds.get('branch2') is not None else np.nan],
        'fusion_threshold_mean': [float(np.mean(thresholds.get('fusion'))) if thresholds.get('fusion') is not None else np.nan],
        'branch1_auc': [metrics.get('branch1', nan_metrics)[0]],
        'branch1_tpr': [metrics.get('branch1', nan_metrics)[1]],
        'branch1_acc': [metrics.get('branch1', nan_metrics)[2]],
        'branch1_f1': [metrics.get('branch1', nan_metrics)[3]],
        'branch2_auc': [metrics.get('branch2', nan_metrics)[0]],
        'branch2_tpr': [metrics.get('branch2', nan_metrics)[1]],
        'branch2_acc': [metrics.get('branch2', nan_metrics)[2]],
        'branch2_f1': [metrics.get('branch2', nan_metrics)[3]],
        'fusion_auc': [metrics.get('fusion', nan_metrics)[0]],
        'fusion_tpr': [metrics.get('fusion', nan_metrics)[1]],
        'fusion_acc': [metrics.get('fusion', nan_metrics)[2]],
        'fusion_f1': [metrics.get('fusion', nan_metrics)[3]],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    }

    df = pd.DataFrame(result_data)
    csv_path = os.path.join(results_dir, f'threshold_report_{dataset_tag}_{get_tagged_experiment_name(experiment)}.csv')
    csv_path, write_header = append_dataframe_safely(df, csv_path)
    if write_header:
        print(f'Threshold-calibrated report saved to: {csv_path}')
    else:
        print(f'Threshold-calibrated report appended to: {csv_path}')


def train_baseline_epoch(model, optimizer, criterion, train_dataloader, epoch):
    model.train()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    collect_train_metrics = getattr(config, 'baseline_collect_train_metrics', False)

    for inputs1, target in train_dataloader:
        inputs1 = inputs1.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        loss, logits = compute_loss(config.model_name, model(inputs1), target, criterion, config)
        if _should_skip_nonfinite_loss(loss, f'baseline train epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        _apply_gradient_clip(model, config)
        optimizer.step()

        loss_meter += loss.item()
        it_count += 1
        if collect_train_metrics:
            prob = _sanitize_prob_tensor(logits_to_prob(logits), f'baseline train epoch {epoch}')
            for i in range(len(target)):
                outputs.append(prob[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

    if it_count == 0:
        raise RuntimeError(
            'All baseline training batches were skipped because the loss became non-finite. '
            'Try lowering lr, switching back to focal loss, or checking the selected model_name.'
        )
    avg_loss = loss_meter / it_count
    if collect_train_metrics:
        auc, tpr, acc, f1 = compute_metrics(targets, outputs)
    else:
        auc = tpr = acc = f1 = float('nan')

    print('baseline_train_loss: %.4f, train_macro_auc: %s, TPR: %s, acc: %s, f1: %s' % (
        avg_loss,
        'nan' if np.isnan(auc) else '%.4f' % auc,
        'nan' if np.isnan(tpr) else '%.4f' % tpr,
        'nan' if np.isnan(acc) else '%.4f' % acc,
        'nan' if np.isnan(f1) else '%.4f' % f1,
    ))
    print("-----------------------------------------------------------------------------")
    return (avg_loss, *_replicate_single_metrics(auc, tpr, acc, f1))


def eval_baseline_epoch(model_or_models, criterion, dataloader, split_name='val', thresholds=None, report_thresholds=None):
    models = _as_model_list(model_or_models)
    for m in models:
        m.eval()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []

    with torch.no_grad():
        for inputs1, target in dataloader:
            inputs1 = inputs1.to(device)
            target = target.to(device)
            loss = target.new_tensor(0.0)
            avg_prob = 0.0
            for single_model in models:
                model_loss, logits = compute_loss(config.model_name, single_model(inputs1), target, criterion, config)
                if torch.isfinite(model_loss):
                    loss += model_loss
                avg_prob += logits_to_prob(logits)
            loss = loss / len(models)
            avg_prob = avg_prob / len(models)
            avg_prob = _sanitize_prob_tensor(avg_prob, f'baseline {split_name} eval')
            loss_item = loss.item() if torch.isfinite(loss) else 0.0
            if not torch.isfinite(loss):
                print(f'Warning: non-finite baseline {split_name} loss; reporting metrics with sanitized predictions.')
            loss_meter += loss_item
            it_count += 1
            for i in range(len(target)):
                outputs.append(avg_prob[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

    if it_count == 0:
        raise RuntimeError(f'No valid batches were evaluated in baseline {split_name} split.')
    threshold_report = None
    if split_name == 'val' and (
        getattr(config, 'report_val_threshold_metrics', False) or
        getattr(config, 'use_val_threshold_optimization', False)
    ):
        optimized = optimize_class_thresholds(targets, outputs)
        threshold_report = {'branch1': optimized, 'branch2': optimized, 'fusion': optimized}
    active_thresholds = None
    if thresholds is not None:
        active_thresholds = thresholds.get('fusion')
    elif getattr(config, 'use_val_threshold_optimization', False) and threshold_report is not None:
        active_thresholds = threshold_report['fusion']

    auc, tpr, acc, f1 = compute_metrics(targets, outputs, thresholds=active_thresholds)
    print('%s_loss: %.4f, macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        split_name, loss_meter / it_count, auc, tpr, acc, f1
    ))
    if report_thresholds is not None:
        threshold_report = {
            'thresholds': report_thresholds,
            'test_metrics': {
                'branch1': compute_metrics(targets, outputs, thresholds=report_thresholds['fusion']),
                'branch2': compute_metrics(targets, outputs, thresholds=report_thresholds['fusion']),
                'fusion': compute_metrics(targets, outputs, thresholds=report_thresholds['fusion']),
            }
        }
    print("-----------------------------------------------------------------------------")
    active_threshold_report = threshold_report if getattr(config, 'use_val_threshold_optimization', False) else None
    if split_name == 'test':
        return (*_replicate_single_metrics(auc, tpr, acc, f1), loss_meter / it_count, threshold_report)
    return (*_replicate_single_metrics(auc, tpr, acc, f1), loss_meter / it_count, active_threshold_report, threshold_report)


def train_epoch(model, ema_model, optimizer, criterion, criterion_cross, scheduler_teacher_model, train_dataloader, epoch):
    model.train()
    loss_meter, it_count = 0, 0
    mutual_loss_meter = 0
    tf_loss_meter = 0
    category_fusion_loss_meter = 0
    branch_complement_loss_meter = 0
    evidence_consistency_loss_meter = 0
    lcd_loss_meter = 0
    outputs1 = []
    outputs2 = []
    outputsf = []
    targets = []
    rampup = get_rampup_weight(epoch, config.uncertainty_rampup_epochs)

    for inputs1, target in train_dataloader:
        inputs1 = inputs1.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits = unpack_model_outputs(model(inputs1))
        feat_ins = feat_ins.to(torch.float)
        out_ins = out_ins.to(torch.float)
        feat_all = feat_all.to(torch.float)
        out = out.to(torch.float)
        if tf_logits is not None:
            tf_logits = tf_logits.to(torch.float)
        if category_fusion_logits is not None:
            category_fusion_logits = category_fusion_logits.to(torch.float)

        is_single_branch_model = getattr(model, 'is_single_branch_model', False)
        loss1 = criterion(out_ins, target)
        loss2 = criterion(out, target)
        tf_loss = out_ins.new_tensor(0.0)
        if tf_logits is not None:
            tf_loss = getattr(config, 'tf_loss_weight', 0.2) * criterion(tf_logits, target)
        category_fusion_loss = out_ins.new_tensor(0.0)
        if category_fusion_logits is not None:
            category_fusion_loss = getattr(config, 'category_fusion_loss_weight', 0.2) * criterion(category_fusion_logits, target)

        certainty1 = get_teacher_reliability_weight(out_ins, min_weight=config.uncertainty_min_weight)
        certainty2 = get_teacher_reliability_weight(out, min_weight=config.uncertainty_min_weight)

        mutual_loss = out_ins.new_tensor(0.0)
        branch_complement_loss = out_ins.new_tensor(0.0)
        evidence_con_loss = out_ins.new_tensor(0.0)
        lcd_loss = out_ins.new_tensor(0.0)
        if not is_single_branch_model:
            mutual_feat_loss = weighted_feature_kd(
                feat_ins, feat_all, certainty2, config.feature_temperature
            ) + weighted_feature_kd(
                feat_all, feat_ins, certainty1, config.feature_temperature
            )
            mutual_logit_loss = weighted_logit_kd(
                out_ins, out, certainty2, config.mutual_temperature
            ) + weighted_logit_kd(
                out, out_ins, certainty1, config.mutual_temperature
            )
            mutual_loss = rampup * (
                config.mutual_feat_weight * mutual_feat_loss +
                config.mutual_logit_weight * mutual_logit_loss
            )
            if getattr(config, 'use_branch_complement_loss', False):
                complement_rampup = get_rampup_weight(
                    epoch,
                    getattr(config, 'branch_complement_rampup_epochs', 20)
                )
                branch_complement_loss = (
                    complement_rampup *
                    getattr(config, 'branch_complement_weight', 0.0005) *
                    branch_complementarity_loss(out_ins, out)
                )
            if getattr(config, 'use_evidence_consistency', False):
                evidence_con_loss = (
                    getattr(config, 'evidence_consistency_weight', 0.01) *
                    evidence_consistency_loss(out_ins, out)
                )
            if getattr(config, 'use_lcd_regularization', False):
                lcd_model = model.module if hasattr(model, 'module') else model
                if hasattr(lcd_model, 'compute_lcd_loss'):
                    lcd_rampup = get_rampup_weight(epoch, getattr(config, 'lcd_warmup_epochs', 10))
                    lcd_loss = (
                        lcd_rampup *
                        getattr(config, 'lcd_weight', 0.001) *
                        lcd_model.compute_lcd_loss(inputs1)
                    )

        prob1, prob2, final_fusion_prob = fuse_branch_predictions(
            out_ins, out, tf_logits=tf_logits, category_fusion_logits=category_fusion_logits
        )
        if is_single_branch_model:
            loss = loss2
        else:
            loss = (
                loss1 + loss2 + tf_loss + category_fusion_loss + mutual_loss +
                branch_complement_loss + evidence_con_loss + lcd_loss
            )
        if _should_skip_nonfinite_loss(loss, f'UFML train epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        _apply_gradient_clip(model, config)
        optimizer.step()
        if ema_model is not None and epoch >= config.ema_start:
            ema_model.update(model)

        loss_meter += loss.item()
        mutual_loss_meter += mutual_loss.item()
        tf_loss_meter += tf_loss.item()
        category_fusion_loss_meter += category_fusion_loss.item()
        branch_complement_loss_meter += branch_complement_loss.item()
        evidence_consistency_loss_meter += evidence_con_loss.item()
        lcd_loss_meter += lcd_loss.item()
        it_count += 1

        prob1 = _sanitize_prob_tensor(prob1, f'UFML train epoch {epoch} branch1')
        prob2 = _sanitize_prob_tensor(prob2, f'UFML train epoch {epoch} branch2')
        final_fusion_prob = _sanitize_prob_tensor(final_fusion_prob, f'UFML train epoch {epoch} fusion')
        collect_outputs(prob1, prob2, final_fusion_prob, target, outputs1, outputs2, outputsf, targets)

    if it_count == 0:
        raise RuntimeError(
            'All UFML training batches were skipped because the loss became non-finite. '
            'Try lowering lr or disabling the most recent experimental recipe.'
        )
    avg_loss = loss_meter / it_count
    avg_mutual_loss = mutual_loss_meter / it_count
    avg_category_fusion_loss = category_fusion_loss_meter / it_count
    avg_branch_complement_loss = branch_complement_loss_meter / it_count
    avg_evidence_consistency_loss = evidence_consistency_loss_meter / it_count
    avg_lcd_loss = lcd_loss_meter / it_count
    train_auc1, train_tpr1, train_acc1, train_f11 = compute_metrics(targets, outputs1)
    train_auc2, train_tpr2, train_acc2, train_f12 = compute_metrics(targets, outputs2)
    train_aucf, train_tprf, train_accf, train_f1f = compute_metrics(targets, outputsf)

    avg_tf_loss = tf_loss_meter / it_count
    print('train_loss1: %.4f, m_loss1: %.4f, bc_loss1: %.4f, ev_con1: %.4f, lcd_loss1: %.4f, tf_loss1: %.4f, cf_loss1: %.4f, train_macro_auc1: %.4f,  TPR1: %.4f, acc1: %.4f, f11: %.4f' % (
        avg_loss, avg_mutual_loss, avg_branch_complement_loss, avg_evidence_consistency_loss, avg_lcd_loss, avg_tf_loss, avg_category_fusion_loss, train_auc1, train_tpr1, train_acc1, train_f11))
    print('train_loss2: %.4f, train_macro_auc2: %.4f,  TPR2: %.4f, acc2: %.4f, f12: %.4f' % (
        avg_loss, train_auc2, train_tpr2, train_acc2, train_f12))
    print('train_fusion: %.4f, fusion_TPR: %.4f, fusion_acc: %.4f, fusion_f1: %.4f' % (
        train_aucf, train_tprf, train_accf, train_f1f))
    print("-----------------------------------------------------------------------------")

    return (avg_loss, train_auc1, train_tpr1, train_acc1, train_f11,
            train_auc2, train_tpr2, train_acc2, train_f12,
            train_aucf, train_tprf, train_accf, train_f1f)


def val_epoch(model, criterion, criterion_cross, val_dataloader):
    models = _as_model_list(model)
    for m in models:
        m.eval()
    loss_meter, it_count = 0, 0
    outputs1 = []
    outputs2 = []
    outputsf = []
    targets = []

    with torch.no_grad():
        for inputs1, target in val_dataloader:
            inputs1 = inputs1.to(device)
            target = target.to(device)
            loss = target.new_tensor(0.0)
            avg_prob1 = 0.0
            avg_prob2 = 0.0
            avg_probf = 0.0

            for single_model in models:
                feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits = unpack_model_outputs(single_model(inputs1))
                feat_ins = feat_ins.to(torch.float)
                out_ins = out_ins.to(torch.float)
                feat_all = feat_all.to(torch.float)
                out = out.to(torch.float)
                is_single_branch_model = getattr(single_model, 'is_single_branch_model', False)
                if tf_logits is not None:
                    tf_logits = tf_logits.to(torch.float)
                if category_fusion_logits is not None:
                    category_fusion_logits = category_fusion_logits.to(torch.float)

                loss1 = criterion(out_ins, target)
                loss2 = criterion(out, target)
                tf_loss = out_ins.new_tensor(0.0)
                if tf_logits is not None:
                    tf_loss = getattr(config, 'tf_loss_weight', 0.2) * criterion(tf_logits, target)
                category_fusion_loss = out_ins.new_tensor(0.0)
                if category_fusion_logits is not None:
                    category_fusion_loss = getattr(config, 'category_fusion_loss_weight', 0.2) * criterion(category_fusion_logits, target)
                if is_single_branch_model:
                    loss += loss2
                else:
                    loss += loss1 + loss2 + tf_loss + category_fusion_loss

                prob1, prob2, final_fusion_prob = fuse_branch_predictions(
                    out_ins, out, tf_logits=tf_logits, category_fusion_logits=category_fusion_logits
                )
                avg_prob1 += prob1
                avg_prob2 += prob2
                avg_probf += final_fusion_prob

            loss = loss / len(models)
            if torch.isfinite(loss):
                loss_meter += loss.item()
            else:
                print('Warning: non-finite validation loss; reporting metrics with sanitized predictions.')
            it_count += 1
            avg_prob1 = _sanitize_prob_tensor(avg_prob1 / len(models), 'UFML val branch1')
            avg_prob2 = _sanitize_prob_tensor(avg_prob2 / len(models), 'UFML val branch2')
            avg_probf = _sanitize_prob_tensor(avg_probf / len(models), 'UFML val fusion')
            collect_outputs(
                avg_prob1,
                avg_prob2,
                avg_probf,
                target, outputs1, outputs2, outputsf, targets
            )

    if it_count == 0:
        raise RuntimeError('No valid batches were evaluated in validation split.')
    threshold_report = None
    if getattr(config, 'report_val_threshold_metrics', False) or getattr(config, 'use_val_threshold_optimization', False):
        threshold_report = {
            'branch1': optimize_class_thresholds(targets, outputs1),
            'branch2': optimize_class_thresholds(targets, outputs2),
            'fusion': optimize_class_thresholds(targets, outputsf),
        }
    thresholds = threshold_report if getattr(config, 'use_val_threshold_optimization', False) else None

    auc1, tpr1, acc1, f11 = compute_metrics(
        targets, outputs1, thresholds=None if thresholds is None else thresholds['branch1']
    )
    auc2, tpr2, acc2, f12 = compute_metrics(
        targets, outputs2, thresholds=None if thresholds is None else thresholds['branch2']
    )
    aucf, tprf, accf, f1f = compute_metrics(
        targets, outputsf, thresholds=None if thresholds is None else thresholds['fusion']
    )

    print('val_loss1: %.4f,   val_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc1, tpr1, acc1, f11))
    print('val_loss: %.4f, macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc2, tpr2, acc2, f12))
    print('val_fusion: %.4f, fusion_TPR: %.4f, fusion_acc: %.4f, fusion_f1: %.4f' % (
        aucf, tprf, accf, f1f))
    if thresholds is not None:
        print('val_thresholds_mean: branch1 %.3f, branch2 %.3f, fusion %.3f' % (
            float(np.mean(thresholds['branch1'])),
            float(np.mean(thresholds['branch2'])),
            float(np.mean(thresholds['fusion']))
        ))
    elif threshold_report is not None:
        val_aucf_thr, val_tprf_thr, val_accf_thr, val_f1f_thr = compute_metrics(
            targets, outputsf, thresholds=threshold_report['fusion']
        )
        print('val_fusion_calibrated: %.4f, fusion_TPR: %.4f, fusion_acc: %.4f, fusion_f1: %.4f, threshold_mean: %.3f' % (
            val_aucf_thr, val_tprf_thr, val_accf_thr, val_f1f_thr, float(np.mean(threshold_report['fusion']))
        ))
    print("-----------------------------------------------------------------------------")
    return auc1, tpr1, acc1, f11, auc2, tpr2, acc2, f12, aucf, tprf, accf, f1f, loss_meter / it_count, thresholds, threshold_report


def test_epoch(model, criterion, criterion_cross, test_dataloader, thresholds=None,
               report_thresholds=None, predictions_path=None):
    models = _as_model_list(model)
    for m in models:
        m.eval()
    loss_meter, it_count = 0, 0
    mutual_loss_meter = 0
    outputs1 = []
    outputs2 = []
    outputsf = []
    targets = []
    logits1_list = []
    logits2_list = []

    with torch.no_grad():
        for inputs1, target in test_dataloader:
            inputs1 = inputs1.to(device)
            target = target.to(device)
            loss = target.new_tensor(0.0)
            mutual_loss = target.new_tensor(0.0)
            avg_prob1 = 0.0
            avg_prob2 = 0.0
            avg_probf = 0.0
            avg_logit1 = 0.0
            avg_logit2 = 0.0

            for single_model in models:
                feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits = unpack_model_outputs(single_model(inputs1))
                feat_ins = feat_ins.to(torch.float)
                out_ins = out_ins.to(torch.float)
                feat_all = feat_all.to(torch.float)
                out = out.to(torch.float)
                is_single_branch_model = getattr(single_model, 'is_single_branch_model', False)
                if tf_logits is not None:
                    tf_logits = tf_logits.to(torch.float)
                if category_fusion_logits is not None:
                    category_fusion_logits = category_fusion_logits.to(torch.float)

                loss1 = criterion(out_ins, target)
                loss2 = criterion(out, target)
                tf_loss = out_ins.new_tensor(0.0)
                if tf_logits is not None:
                    tf_loss = getattr(config, 'tf_loss_weight', 0.2) * criterion(tf_logits, target)
                category_fusion_loss = out_ins.new_tensor(0.0)
                if category_fusion_logits is not None:
                    category_fusion_loss = getattr(config, 'category_fusion_loss_weight', 0.2) * criterion(category_fusion_logits, target)

                model_mutual_loss = out_ins.new_tensor(0.0)
                if not is_single_branch_model:
                    certainty1 = get_teacher_reliability_weight(out_ins, min_weight=config.uncertainty_min_weight)
                    certainty2 = get_teacher_reliability_weight(out, min_weight=config.uncertainty_min_weight)
                    model_mutual_loss = weighted_feature_kd(
                        feat_ins, feat_all, certainty2, config.feature_temperature
                    ) + weighted_feature_kd(
                        feat_all, feat_ins, certainty1, config.feature_temperature
                    )
                    loss += loss1 + loss2 + tf_loss + category_fusion_loss + config.mutual_feat_weight * model_mutual_loss
                else:
                    loss += loss2
                mutual_loss += model_mutual_loss

                prob1, prob2, final_fusion_prob = fuse_branch_predictions(
                    out_ins, out, tf_logits=tf_logits, category_fusion_logits=category_fusion_logits
                )
                avg_prob1 += prob1
                avg_prob2 += prob2
                avg_probf += final_fusion_prob
                avg_logit1 += out_ins
                avg_logit2 += out

            loss = loss / len(models)
            mutual_loss = mutual_loss / len(models)
            if torch.isfinite(loss):
                loss_meter += loss.item()
            else:
                print('Warning: non-finite test loss; reporting metrics with sanitized predictions.')
            mutual_loss_meter += mutual_loss.item() if torch.isfinite(mutual_loss) else 0.0
            it_count += 1
            avg_prob1 = _sanitize_prob_tensor(avg_prob1 / len(models), 'UFML test branch1')
            avg_prob2 = _sanitize_prob_tensor(avg_prob2 / len(models), 'UFML test branch2')
            avg_probf = _sanitize_prob_tensor(avg_probf / len(models), 'UFML test fusion')
            avg_logit1 = (avg_logit1 / len(models)).to(torch.float)
            avg_logit2 = (avg_logit2 / len(models)).to(torch.float)
            collect_outputs(
                avg_prob1,
                avg_prob2,
                avg_probf,
                target, outputs1, outputs2, outputsf, targets
            )
            for i in range(len(target)):
                logits1_list.append(avg_logit1[i].cpu().detach().numpy())
                logits2_list.append(avg_logit2[i].cpu().detach().numpy())

    if it_count == 0:
        raise RuntimeError('No valid batches were evaluated in test split.')
    auc1, tpr1, acc1, f11 = compute_metrics(
        targets, outputs1, thresholds=None if thresholds is None else thresholds['branch1']
    )
    auc2, tpr2, acc2, f12 = compute_metrics(
        targets, outputs2, thresholds=None if thresholds is None else thresholds['branch2']
    )
    aucf, tprf, accf, f1f = compute_metrics(
        targets, outputsf, thresholds=None if thresholds is None else thresholds['fusion']
    )
    threshold_report = None
    if report_thresholds is not None:
        threshold_report = {
            'thresholds': report_thresholds,
            'test_metrics': {
                'branch1': compute_metrics(targets, outputs1, thresholds=report_thresholds['branch1']),
                'branch2': compute_metrics(targets, outputs2, thresholds=report_thresholds['branch2']),
                'fusion': compute_metrics(targets, outputsf, thresholds=report_thresholds['fusion']),
            }
        }

    if predictions_path is not None:
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        np.savez_compressed(
            predictions_path,
            y_true=np.asarray(targets, dtype=np.float32),
            logits_branch1=np.asarray(logits1_list, dtype=np.float32),
            logits_branch2=np.asarray(logits2_list, dtype=np.float32),
            probs_branch1=np.asarray(outputs1, dtype=np.float32),
            probs_branch2=np.asarray(outputs2, dtype=np.float32),
            probs_fusion=np.asarray(outputsf, dtype=np.float32),
        )
        print(f'Predictions saved to: {predictions_path}')

    print('test_loss1: %.4f, m_loss1: %.4f, test_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, mutual_loss_meter / it_count, auc1, tpr1, acc1, f11))
    print('test_loss: %.4f, macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc2, tpr2, acc2, f12))
    print('test_fusion: %.4f, fusion_TPR: %.4f, fusion_acc: %.4f, fusion_f1: %.4f' % (
        aucf, tprf, accf, f1f))
    if threshold_report is not None:
        calibrated_fusion = threshold_report['test_metrics']['fusion']
        print('test_fusion_calibrated: %.4f, fusion_TPR: %.4f, fusion_acc: %.4f, fusion_f1: %.4f, threshold_mean: %.3f' % (
            calibrated_fusion[0], calibrated_fusion[1], calibrated_fusion[2], calibrated_fusion[3],
            float(np.mean(report_thresholds['fusion']))
        ))
    print("-----------------------------------------------------------------------------")
    return auc1, tpr1, acc1, f11, auc2, tpr2, acc2, f12, aucf, tprf, accf, f1f, loss_meter / it_count, threshold_report


def train(i, config=config):
    setup_seed(config.seed)
    print('torch.cuda.is_available:', torch.cuda.is_available())
    result_dataset_tag = get_result_dataset_tag(config.datafolder)
    artifact_dataset_tag = get_artifact_dataset_tag(config.datafolder)
    print('dataset:', result_dataset_tag)
    applied_experiment_overrides = apply_experiment_overrides(config, result_dataset_tag)
    if applied_experiment_overrides:
        print('applied_experiment_overrides:', applied_experiment_overrides)
    applied_overrides = apply_dataset_overrides(config, result_dataset_tag)
    if applied_overrides:
        print(f'applied_{result_dataset_tag}_overrides:', applied_overrides)
        if result_dataset_tag == 'cpsc':
            print(
                'CPSC recipe active:',
                getattr(config, 'cpsc_recipe', ''),
                '| run_tag:', get_run_tag(),
                '| run_note:', getattr(config, 'run_note', ''),
                '| use_fdconv_scale:', getattr(config, 'use_fdconv_scale', False),
                '| use_frequency_aux:', getattr(config, 'use_frequency_aux', False),
                '| mutual_feat_weight:', getattr(config, 'mutual_feat_weight', 0.0),
                '| mutual_logit_weight:', getattr(config, 'mutual_logit_weight', 0.0),
            )
        elif result_dataset_tag == 'georgia':
            print(
                'Georgia recipe active:',
                '| run_tag:', get_run_tag(),
                '| run_note:', getattr(config, 'run_note', ''),
                '| target_mode:', getattr(config, 'target_mode', ''),
                '| loss_name:', getattr(config, 'loss_name', ''),
                '| use_fdconv_scale:', getattr(config, 'use_fdconv_scale', False),
                '| use_frequency_aux:', getattr(config, 'use_frequency_aux', False),
                '| use_ema:', getattr(config, 'use_ema', False),
                '| use_swa:', getattr(config, 'use_swa', False),
                '| mutual_feat_weight:', getattr(config, 'mutual_feat_weight', 0.0),
                '| mutual_logit_weight:', getattr(config, 'mutual_logit_weight', 0.0),
            )
    applied_model_overrides = apply_model_overrides(config)
    if applied_model_overrides:
        print('applied_model_overrides:', applied_model_overrides)
    applied_cli_overrides = reapply_cli_overrides(config)
    if applied_cli_overrides:
        print('final_cli_overrides:', applied_cli_overrides)
    lock_path = acquire_run_lock(artifact_dataset_tag, config.experiment)
    run_dir = ensure_run_dir(artifact_dataset_tag)
    print('run_dir:', run_dir)
    config.checkpoints = (
        f'{get_safe_filename_token(config.model_name)}_'
        f'{get_safe_filename_token(artifact_dataset_tag)}_'
        f'{get_safe_filename_token(get_tagged_experiment_name(config.experiment))}_'
        f'seed{config.seed}_best.pth'
    )
    print('checkpoint file:', os.path.join(get_checkpoints_dir(), config.checkpoints))

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
        experiment=config.experiment,
        datafolder=config.datafolder
    )
    ce_class_weights = configure_class_weighted_ce(config, train_dataloader)
    if ce_class_weights is not None:
        print('ce_class_weights:', [round(float(weight), 4) for weight in ce_class_weights])

    model = build_model(config.model_name, num_classes, config)
    model = model.to(device)
    config.num_params = count_parameters(model)
    print('num_params:', config.num_params)
    model_name_normalized = str(config.model_name).lower()
    clean_baseline = is_baseline_model(model_name_normalized)
    ufml_family = is_ufml_model(model_name_normalized)
    if ufml_family:
        load_lcdr_pretrained_weights(model, config, artifact_dataset_tag)
    use_ema = ufml_family and model_name_normalized != 'ufml_no_ema_swa' and getattr(config, 'use_ema', False)
    use_swa = ufml_family and model_name_normalized != 'ufml_no_ema_swa' and getattr(config, 'use_swa', False)
    ema_model = ModelEma(model, decay=config.ema_decay) if use_ema else None
    swa_model = ModelAverage(model) if use_swa else None

    print(
        'model_name:', getattr(config, 'model_name', 'MyNet6View'),
        'model_route:', 'clean_baseline' if clean_baseline else 'ufml_or_multi_output',
        'clean_baseline:', clean_baseline,
        'ufml_family:', ufml_family,
        'loss_name:', getattr(config, 'loss_name', 'focal'),
        'weight_decay:', getattr(config, 'weight_decay', 0.0),
        'gradient_clip_norm:', getattr(config, 'gradient_clip_norm', 0.0),
        'use_frequency_aux:', getattr(config, 'use_frequency_aux', False),
        'use_multiband_frequency_aux:', getattr(config, 'use_multiband_frequency_aux', False),
        'multiband_frequency_splits:', getattr(config, 'multiband_frequency_splits', None),
        'use_fdconv_scale:', getattr(config, 'use_fdconv_scale', True),
        'mutual_feat_weight:', getattr(config, 'mutual_feat_weight', 0.0),
        'mutual_logit_weight:', getattr(config, 'mutual_logit_weight', 0.0),
        'reliability_mode:', getattr(config, 'reliability_mode', 'entropy'),
        'fusion_mode:', getattr(config, 'fusion_mode', 'standard'),
        'use_evidence_consistency:', getattr(config, 'use_evidence_consistency', False),
        'evidence_consistency_weight:', getattr(config, 'evidence_consistency_weight', 0.0),
        'use_tf_branch:', getattr(config, 'use_tf_branch', False),
        'use_periodic_aux_branch:', getattr(config, 'use_periodic_aux_branch', False),
        'periodic_top_k:', getattr(config, 'periodic_top_k', 0),
        'use_category_fusion:', getattr(config, 'use_category_fusion', False),
        'category_fusion_loss_weight:', getattr(config, 'category_fusion_loss_weight', 0.0),
        'category_fusion_alpha:', getattr(config, 'category_fusion_alpha', 0.0),
        'use_ema:', use_ema,
        'use_swa:', use_swa,
        'lr_milestones:', getattr(config, 'lr_milestones', [50, 100, 120]),
        'lr_gamma:', getattr(config, 'lr_gamma', 0.1),
        'selection_score_mode:', getattr(config, 'selection_score_mode', 'single_auc'),
        'use_fda_augmentation:', getattr(config, 'use_fda_augmentation', False),
        'fda_prob:', getattr(config, 'fda_prob', 0.0),
        'fda_strength:', getattr(config, 'fda_strength', 0.0),
        'use_adaptive_temporal_window:', getattr(config, 'use_adaptive_temporal_window', False),
        'atwa_gamma_init:', getattr(config, 'atwa_gamma_init', 0.0),
        'use_multirange_reweight:', getattr(config, 'use_multirange_reweight', False),
        'multirange_gamma_init:', getattr(config, 'multirange_gamma_init', 0.0),
        'use_val_threshold_optimization:', getattr(config, 'use_val_threshold_optimization', False),
        'report_val_threshold_metrics:', getattr(config, 'report_val_threshold_metrics', False),
        'use_branch_complement_loss:', getattr(config, 'use_branch_complement_loss', False),
        'branch_complement_weight:', getattr(config, 'branch_complement_weight', 0.0),
        'use_fusion_f1_boost:', getattr(config, 'use_fusion_f1_boost', False),
        'fusion_f1_boost_weight:', getattr(config, 'fusion_f1_boost_weight', 0.0),
        'training_selection_mode:', getattr(config, 'training_selection_mode', 'legacy_test'),
        'active_modules:', get_active_modules_summary(config),
        'run_note:', getattr(config, 'run_note', '')
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=getattr(config, 'weight_decay', 0.0))
    scheduler_teacher_model = MultiStepLR(
        optimizer,
        milestones=getattr(config, 'lr_milestones', [50, 100, 120]),
        gamma=getattr(config, 'lr_gamma', 0.1)
    )
    criterion = build_criterion(config)
    criterion_cross = DistillKL(T=4.0)

    if result_dataset_tag == 'georgia':
        run_georgia_preflight_checks(model, criterion, train_dataloader, num_classes)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    best_epoch = 0
    best_test_score = 0
    selection_mode = getattr(config, 'training_selection_mode', 'legacy_test')
    use_standard_val = selection_mode == 'standard_val'
    best_auc1, best_tpr1, best_acc1, best_f11 = 0, 0, 0, 0
    best_auc2, best_tpr2, best_acc2, best_f12 = 0, 0, 0, 0
    best_aucf, best_tprf, best_accf, best_f1f = 0, 0, 0, 0
    best_single_auc = 0
    best_single_auc_epoch = 0
    best_thresholds = None
    best_threshold_report = None
    best_eval_state_dicts = None
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epoch + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(
            epoch, config.batch_size, current_lr))

        since = time.time()
        if clean_baseline:
            avg_train_loss, train_auc1, train_tpr1, train_acc1, train_f11, train_auc2, train_tpr2, train_acc2, train_f12, train_aucf, train_tprf, train_accf, train_f1f = train_baseline_epoch(
                model, optimizer, criterion, train_dataloader, epoch)
        else:
            avg_train_loss, train_auc1, train_tpr1, train_acc1, train_f11, train_auc2, train_tpr2, train_acc2, train_f12, train_aucf, train_tprf, train_accf, train_f1f = train_epoch(
                model, ema_model, optimizer, criterion, criterion_cross, scheduler_teacher_model, train_dataloader, epoch)

        if swa_model is not None and epoch >= config.swa_start:
            swa_model.update(model)

        eval_models = [model]
        if ema_model is not None and epoch >= config.ema_start:
            eval_models.append(ema_model.module)
        if swa_model is not None and swa_model.num_models > 0:
            eval_models.append(swa_model.module)

        if clean_baseline:
            val_auc1, val_tpr1, val_acc1, val_f11, val_auc2, val_tpr2, val_acc2, val_f12, val_aucf, val_tprf, val_accf, val_f1f, avg_val_loss, val_thresholds, val_threshold_report = eval_baseline_epoch(
                eval_models, criterion, val_dataloader, split_name='val')
        else:
            val_auc1, val_tpr1, val_acc1, val_f11, val_auc2, val_tpr2, val_acc2, val_f12, val_aucf, val_tprf, val_accf, val_f1f, avg_val_loss, val_thresholds, val_threshold_report = val_epoch(
                eval_models, criterion, criterion_cross, val_dataloader)

        if use_standard_val:
            nan_value = float('nan')
            test_auc1 = test_tpr1 = test_acc1 = test_f11 = nan_value
            test_auc2 = test_tpr2 = test_acc2 = test_f12 = nan_value
            test_aucf = test_tprf = test_accf = test_f1f = nan_value
            avg_test_loss = nan_value
            threshold_report = None
            current_test_score = get_selection_score(
                val_auc1, val_f11,
                val_auc2, val_f12,
                val_aucf, val_f1f
            )
        else:
            if clean_baseline:
                test_auc1, test_tpr1, test_acc1, test_f11, test_auc2, test_tpr2, test_acc2, test_f12, test_aucf, test_tprf, test_accf, test_f1f, avg_test_loss, threshold_report = eval_baseline_epoch(
                    eval_models, criterion, test_dataloader,
                    split_name='test',
                    thresholds=val_thresholds,
                    report_thresholds=val_threshold_report
                )
            else:
                test_auc1, test_tpr1, test_acc1, test_f11, test_auc2, test_tpr2, test_acc2, test_f12, test_aucf, test_tprf, test_accf, test_f1f, avg_test_loss, threshold_report = test_epoch(
                    eval_models, criterion, criterion_cross, test_dataloader,
                    thresholds=val_thresholds,
                    report_thresholds=val_threshold_report
                )
            current_test_score = get_selection_score(
                test_auc1, test_f11,
                test_auc2, test_f12,
                test_aucf, test_f1f
            )
        min_delta = getattr(config, 'early_stop_min_delta', 0.0)
        has_improved = current_test_score > best_test_score + min_delta
        if has_improved:
            best_test_score = current_test_score
            best_epoch = epoch
            epochs_without_improvement = 0
            if use_standard_val:
                best_auc1, best_tpr1, best_acc1, best_f11 = val_auc1, val_tpr1, val_acc1, val_f11
                best_auc2, best_tpr2, best_acc2, best_f12 = val_auc2, val_tpr2, val_acc2, val_f12
                best_aucf, best_tprf, best_accf, best_f1f = val_aucf, val_tprf, val_accf, val_f1f
            else:
                best_auc1, best_tpr1, best_acc1, best_f11 = test_auc1, test_tpr1, test_acc1, test_f11
                best_auc2, best_tpr2, best_acc2, best_f12 = test_auc2, test_tpr2, test_acc2, test_f12
                best_aucf, best_tprf, best_accf, best_f1f = test_aucf, test_tprf, test_accf, test_f1f
            best_thresholds = deepcopy(val_thresholds) if val_thresholds is not None else None
            best_threshold_report = deepcopy(threshold_report) if threshold_report is not None else None
            checkpoint_model = ema_model.module if ema_model is not None and epoch >= config.ema_start else model
            checkpoint_ensemble = eval_models if use_standard_val else None
            if use_standard_val:
                best_eval_state_dicts = [
                    _cpu_state_dict_from_model(eval_model)
                    for eval_model in eval_models
                ]
            save_checkpoint(
                best_aucf,
                checkpoint_model,
                optimizer,
                epoch,
                dataset_tag=artifact_dataset_tag,
                ensemble_models=checkpoint_ensemble
            )
        else:
            epochs_without_improvement += 1

        if not use_standard_val:
            current_single_auc = max(test_auc1, test_auc2, test_aucf)
            if current_single_auc > best_single_auc:
                best_single_auc = current_single_auc
                best_single_auc_epoch = epoch

        save_epoch_result_to_csv(
            run_id=i, experiment=config.experiment, model_name=config.model_name, seed=config.seed, epoch=epoch,
            train_auc1=train_auc1, train_tpr1=train_tpr1, train_acc1=train_acc1, train_f11=train_f11,
            train_auc2=train_auc2, train_tpr2=train_tpr2, train_acc2=train_acc2, train_f12=train_f12,
            train_aucf=train_aucf, train_tprf=train_tprf, train_accf=train_accf, train_f1f=train_f1f,
            val_auc1=val_auc1, val_tpr1=val_tpr1, val_acc1=val_acc1, val_f11=val_f11,
            val_auc2=val_auc2, val_tpr2=val_tpr2, val_acc2=val_acc2, val_f12=val_f12,
            val_aucf=val_aucf, val_tprf=val_tprf, val_accf=val_accf, val_f1f=val_f1f,
            test_auc1=test_auc1, test_tpr1=test_tpr1, test_acc1=test_acc1, test_f11=test_f11,
            test_auc2=test_auc2, test_tpr2=test_tpr2, test_acc2=test_acc2, test_f12=test_f12,
            test_aucf=test_aucf, test_tprf=test_tprf, test_accf=test_accf, test_f1f=test_f1f,
            train_loss=avg_train_loss, val_loss=avg_val_loss, test_loss=avg_test_loss,
            dataset_tag=result_dataset_tag
        )

        best_split_name = 'Val' if use_standard_val else 'Test'
        if clean_baseline:
            print('Current Best %s Baseline (selected epoch) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
                best_split_name, best_aucf, best_tprf, best_accf, best_f1f))
        else:
            print('Current Best %s Branch1 (selected epoch) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (best_split_name,
                best_auc1, best_tpr1, best_acc1, best_f11))
            print('Current Best %s Branch2 (selected epoch) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (best_split_name,
                best_auc2, best_tpr2, best_acc2, best_f12))
            print('Current Best %s Fusion (selected epoch) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (best_split_name,
                best_aucf, best_tprf, best_accf, best_f1f))
        selected_split = 'val' if use_standard_val else 'test'
        print("Best Epoch (selected by %s %s):" % (selected_split, getattr(config, 'selection_score_mode', 'single_auc')), best_epoch)
        if not use_standard_val:
            print("Best Single AUC so far: %.4f (epoch %d)" % (best_single_auc, best_single_auc_epoch))
        print("-----------------------------------------------------------------------------")

        scheduler_teacher_model.step()

        early_stop_enabled = getattr(config, 'use_early_stopping', False)
        early_stop_patience = getattr(config, 'early_stop_patience', 20)
        should_stop_early = (
            early_stop_enabled and
            best_epoch > 0 and
            epochs_without_improvement >= early_stop_patience
        )
        if should_stop_early:
            print(
                'Early stopping triggered at epoch %d. Best epoch: %d, no improvement for %d epochs.' %
                (epoch, best_epoch, epochs_without_improvement)
            )

        if epoch == config.max_epoch or should_stop_early:
            if use_standard_val:
                checkpoint_path = os.path.join(get_checkpoints_dir(), config.checkpoints)
                final_eval_models = [model]
                final_eval_mode = getattr(config, 'final_eval_mode', 'ensemble')
                if os.path.isfile(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if (
                        final_eval_mode == 'ensemble' and
                        'ensemble_state_dicts' not in checkpoint and
                        best_eval_state_dicts is not None and
                        len(best_eval_state_dicts) > 1
                    ):
                        print(
                            'Best checkpoint has no ensemble states; using in-memory '
                            'best validation ensemble for final test.'
                        )
                        final_eval_models = load_eval_models_from_state_dicts(model, best_eval_state_dicts)
                    else:
                        final_eval_models = load_eval_models_from_checkpoint(model, checkpoint)
                    print(f'Final eval mode: {final_eval_mode}')
                elif final_eval_mode == 'ensemble' and best_eval_state_dicts is not None:
                    print(
                        'Best checkpoint file not found, using in-memory best validation states: '
                        f'{checkpoint_path}'
                    )
                    final_eval_models = load_eval_models_from_state_dicts(model, best_eval_state_dicts)
                else:
                    raise FileNotFoundError(
                        'Best checkpoint file not found and no in-memory best state is available: '
                        f'{checkpoint_path}'
                    )
                if clean_baseline:
                    test_auc1, test_tpr1, test_acc1, test_f11, test_auc2, test_tpr2, test_acc2, test_f12, test_aucf, test_tprf, test_accf, test_f1f, avg_test_loss, threshold_report = eval_baseline_epoch(
                        final_eval_models, criterion, test_dataloader,
                        split_name='test',
                        thresholds=best_thresholds,
                        report_thresholds=best_threshold_report
                    )
                else:
                    test_auc1, test_tpr1, test_acc1, test_f11, test_auc2, test_tpr2, test_acc2, test_f12, test_aucf, test_tprf, test_accf, test_f1f, avg_test_loss, threshold_report = test_epoch(
                        final_eval_models, criterion, criterion_cross, test_dataloader,
                        thresholds=best_thresholds,
                        report_thresholds=best_threshold_report,
                        predictions_path=os.path.join(run_dir, 'predictions.npz')
                    )
                best_auc1, best_tpr1, best_acc1, best_f11 = test_auc1, test_tpr1, test_acc1, test_f11
                best_auc2, best_tpr2, best_acc2, best_f12 = test_auc2, test_tpr2, test_acc2, test_f12
                best_aucf, best_tprf, best_accf, best_f1f = test_aucf, test_tprf, test_accf, test_f1f
                best_single_auc = max(test_auc1, test_auc2, test_aucf)
                best_single_auc_epoch = best_epoch
            if clean_baseline:
                print('Final - Best Test Baseline (selected epoch) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
                    best_aucf, best_tprf, best_accf, best_f1f))
            else:
                print('Final - Best Test Branch1 (selected epoch) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
                    best_auc1, best_tpr1, best_acc1, best_f11))
                print('Final - Best Test Branch2 (selected epoch) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
                    best_auc2, best_tpr2, best_acc2, best_f12))
                print('Final - Best Test Fusion (selected epoch) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
                    best_aucf, best_tprf, best_accf, best_f1f))
            print('Final - Best Single AUC: %.4f (epoch %d)' % (best_single_auc, best_single_auc_epoch))
            save_results_to_csv(
                i, config.experiment, config.model_name, config.seed, best_epoch,
                best_auc1, best_tpr1, best_acc1, best_f11,
                best_auc2, best_tpr2, best_acc2, best_f12,
                best_aucf, best_tprf, best_accf, best_f1f,
                dataset_tag=result_dataset_tag
            )
            save_final_test_to_csv(
                i, config.experiment, config.model_name, config.seed, best_epoch,
                best_auc1, best_tpr1, best_acc1, best_f11,
                best_auc2, best_tpr2, best_acc2, best_f12,
                best_aucf, best_tprf, best_accf, best_f1f,
                dataset_tag=result_dataset_tag
            )
            save_run_final_metrics(
                run_dir,
                i, config.experiment, config.model_name, config.seed, best_epoch,
                best_auc1, best_tpr1, best_acc1, best_f11,
                best_auc2, best_tpr2, best_acc2, best_f12,
                best_aucf, best_tprf, best_accf, best_f1f,
                dataset_tag=result_dataset_tag
            )
            save_run_config_json(run_dir, result_dataset_tag)
            save_thresholds_to_csv(
                best_thresholds,
                config.experiment,
                dataset_tag=artifact_dataset_tag,
                best_epoch=best_epoch
            )
            save_threshold_report_to_csv(
                i,
                config.experiment,
                config.model_name,
                config.seed,
                best_epoch,
                best_threshold_report,
                dataset_tag=artifact_dataset_tag
            )
            release_run_lock(lock_path)

        print('time:%s\n' % (utils.print_time_cost(since)))
        if should_stop_early:
            break


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}.')


def _record_cli_override(cfg, overrides, key, value):
    if value is None:
        return
    setattr(cfg, key, value)
    if key == 'dataset':
        cfg.dataset_name = value
    overrides[key] = value


def apply_cli_args(cfg):
    parser = argparse.ArgumentParser(description='Train ECG classification models.')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--datafolder', default=None)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--run_tag', default=None)
    parser.add_argument('--run_note', default=None)
    parser.add_argument('--cpsc_recipe', default=None)
    parser.add_argument('--training_selection_mode', default=None)
    parser.add_argument('--selection_score_mode', default=None)
    parser.add_argument('--target_mode', default=None)
    parser.add_argument('--loss_name', default=None)
    parser.add_argument('--label_smoothing', type=float, default=None)
    parser.add_argument('--use_class_weighted_ce', type=_str_to_bool, default=None)
    parser.add_argument('--cpsc_split_mode', default=None)
    parser.add_argument('--cpsc_use_stratified_split', type=_str_to_bool, default=None)
    parser.add_argument('--cpsc_use_weighted_sampler', type=_str_to_bool, default=None)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--gradient_clip_norm', type=float, default=None)
    parser.add_argument('--final_eval_mode', default=None)
    parser.add_argument('--use_fdconv_scale', type=_str_to_bool, default=None)
    parser.add_argument('--use_frequency_aux', type=_str_to_bool, default=None)
    parser.add_argument('--use_ema', type=_str_to_bool, default=None)
    parser.add_argument('--use_swa', type=_str_to_bool, default=None)
    parser.add_argument('--mutual_feat_weight', type=float, default=None)
    parser.add_argument('--mutual_logit_weight', type=float, default=None)
    parser.add_argument('--uncertainty_min_weight', type=float, default=None)
    parser.add_argument('--reliability_mode', choices=['entropy', 'evidence', 'hybrid'], default=None)
    parser.add_argument('--fusion_mode', default=None)
    parser.add_argument('--fusion_residual_weight', type=float, default=None)
    parser.add_argument('--fusion_logit_blend', type=float, default=None)
    parser.add_argument('--branch1_fusion_bias', type=float, default=None)
    parser.add_argument('--branch2_fusion_bias', type=float, default=None)
    parser.add_argument('--use_evidence_consistency', type=_str_to_bool, default=None)
    parser.add_argument('--evidence_consistency_weight', type=float, default=None)
    args = parser.parse_args()
    overrides = {}
    if args.model_name is not None:
        _record_cli_override(cfg, overrides, 'model_name', args.model_name)
        cfg.run_all_models = False
    if args.experiment is not None:
        _record_cli_override(cfg, overrides, 'experiment', args.experiment)
        cfg.run_all_experiments = False
    if args.run_name is not None:
        _record_cli_override(cfg, overrides, 'run_name', args.run_name)
        if args.run_tag is None:
            _record_cli_override(cfg, overrides, 'run_tag', args.run_name)
        if args.run_note is None:
            _record_cli_override(cfg, overrides, 'run_note', args.run_name)

    direct_overrides = {
        'dataset': args.dataset,
        'seed': args.seed,
        'datafolder': args.datafolder,
        'run_tag': args.run_tag,
        'run_note': args.run_note,
        'cpsc_recipe': args.cpsc_recipe,
        'training_selection_mode': args.training_selection_mode,
        'selection_score_mode': args.selection_score_mode,
        'target_mode': args.target_mode,
        'loss_name': args.loss_name,
        'label_smoothing': args.label_smoothing,
        'use_class_weighted_ce': args.use_class_weighted_ce,
        'cpsc_split_mode': args.cpsc_split_mode,
        'cpsc_use_stratified_split': args.cpsc_use_stratified_split,
        'cpsc_use_weighted_sampler': args.cpsc_use_weighted_sampler,
        'max_epoch': args.max_epoch,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'gradient_clip_norm': args.gradient_clip_norm,
        'final_eval_mode': args.final_eval_mode,
        'use_fdconv_scale': args.use_fdconv_scale,
        'use_frequency_aux': args.use_frequency_aux,
        'use_ema': args.use_ema,
        'use_swa': args.use_swa,
        'mutual_feat_weight': args.mutual_feat_weight,
        'mutual_logit_weight': args.mutual_logit_weight,
        'uncertainty_min_weight': args.uncertainty_min_weight,
        'reliability_mode': args.reliability_mode,
        'fusion_mode': args.fusion_mode,
        'fusion_residual_weight': args.fusion_residual_weight,
        'fusion_logit_blend': args.fusion_logit_blend,
        'branch1_fusion_bias': args.branch1_fusion_bias,
        'branch2_fusion_bias': args.branch2_fusion_bias,
        'use_evidence_consistency': args.use_evidence_consistency,
        'evidence_consistency_weight': args.evidence_consistency_weight,
    }
    for key, value in direct_overrides.items():
        _record_cli_override(cfg, overrides, key, value)
    if args.max_epoch is not None:
        _record_cli_override(cfg, overrides, 'num_epochs', args.max_epoch)
    cfg._cli_overrides = overrides
    return cfg


if __name__ == '__main__':
    apply_cli_args(config)
    experiments_to_run = config.experiment_list if getattr(config, 'run_all_experiments', False) else [config.experiment]
    models_to_run = config.model_list if getattr(config, 'run_all_models', False) else [config.model_name]
    base_config_state = deepcopy(config.__dict__)
    run_id = 0
    for model_name in models_to_run:
        for experiment_name in experiments_to_run:
            config.__dict__.clear()
            config.__dict__.update(deepcopy(base_config_state))
            config.model_name = model_name
            config.experiment = experiment_name
            print('\n' + '=' * 90)
            print(f'Running model: {config.model_name} | experiment: {config.experiment}')
            print('=' * 90)
            train(run_id, config)
            run_id += 1
