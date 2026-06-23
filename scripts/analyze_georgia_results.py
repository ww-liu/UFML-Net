# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config
from data_load_georgia import (
    LEAD_ORDER,
    TARGET_FS,
    TARGET_LENGTH,
    get_cache_dir,
    load_data,
)
from main_mul import (
    apply_dataset_overrides,
    fuse_branch_predictions,
    get_canonical_result_dataset_tag,
    unpack_model_outputs,
)
from models.registry import build_model


def set_georgia_config(args):
    config.dataset = 'georgia'
    config.dataset_name = 'georgia'
    config.datafolder = args.datafolder
    config.model_name = args.model_name
    config.experiment = args.experiment
    config.seed = args.seed
    config.batch_size = args.batch_size
    config.run_name = args.run_name
    config.run_tag = args.run_name
    config.target_mode = 'multilabel'
    config.loss_name = 'focal'
    apply_dataset_overrides(config, 'georgia')
    config.use_fdconv_scale = args.use_fdconv_scale
    config.use_frequency_aux = args.use_frequency_aux
    config.use_ema = args.use_ema
    config.use_swa = args.use_swa
    config.mutual_feat_weight = args.mutual_feat_weight
    config.mutual_logit_weight = args.mutual_logit_weight
    config.fusion_mode = args.fusion_mode
    config.selection_score_mode = args.selection_score_mode
    config.training_selection_mode = args.training_selection_mode
    config.max_epoch = args.max_epoch
    config.weight_decay = args.weight_decay
    return config


def load_label_map(cache_dir):
    with open(Path(cache_dir) / 'label_map.json', encoding='utf-8') as f:
        label_map = json.load(f)
    labels = [item['code'] for item in label_map['labels']]
    counts = [int(item['count']) for item in label_map['labels']]
    return label_map, labels, counts


def load_split_payload(cache_dir, seed=1, min_label_count=100):
    path = Path(cache_dir) / f'splits_seed{seed}_min{min_label_count}.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def find_checkpoint(args):
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path

    checkpoint_dir = Path(args.checkpoint_dir)
    patterns = [
        f'{args.model_name}_georgia_{args.experiment}_{args.run_name}_seed{args.seed}_best.pth',
        f'{args.model_name}_georgia_{args.experiment}_{args.run_name}_*seed{args.seed}_best.pth',
        f'*georgia*{args.run_name}*seed{args.seed}*best.pth',
        f'*georgia*seed{args.seed}*best.pth',
    ]
    matches = []
    for pattern in patterns:
        matches.extend(checkpoint_dir.glob(pattern))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(
            f'Could not find Georgia checkpoint in {checkpoint_dir}. '
            'Pass --checkpoint explicitly if the file uses a custom name.'
        )
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def load_model_from_checkpoint(checkpoint_path, num_classes, device):
    model = build_model(config.model_name, num_classes, config).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def make_loader(x, y, batch_size):
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_split(model, loader, device):
    y_true, probs1, probs2, probsf = [], [], [], []
    logits1, logits2 = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, out1, _, out2, tf_logits, category_fusion_logits = unpack_model_outputs(outputs)
            prob1, prob2, fusion_prob = fuse_branch_predictions(
                out1.float(),
                out2.float(),
                tf_logits=tf_logits.float() if torch.is_tensor(tf_logits) else None,
                category_fusion_logits=category_fusion_logits.float() if torch.is_tensor(category_fusion_logits) else None,
            )
            y_true.append(labels.numpy())
            probs1.append(prob1.detach().cpu().numpy())
            probs2.append(prob2.detach().cpu().numpy())
            probsf.append(fusion_prob.detach().cpu().numpy())
            logits1.append(out1.detach().cpu().numpy())
            logits2.append(out2.detach().cpu().numpy())
    return {
        'y_true': np.concatenate(y_true, axis=0),
        'probs_branch1': np.concatenate(probs1, axis=0),
        'probs_branch2': np.concatenate(probs2, axis=0),
        'probs_fusion': np.concatenate(probsf, axis=0),
        'logits_branch1': np.concatenate(logits1, axis=0),
        'logits_branch2': np.concatenate(logits2, axis=0),
    }


def safe_auc(y_true, y_prob, average=None):
    try:
        return float(roc_auc_score(y_true, y_prob, average=average))
    except ValueError:
        return float('nan')


def metrics_at_threshold(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    return {
        'macro_auc': safe_auc(y_true, y_prob, average='macro'),
        'micro_auc': safe_auc(y_true, y_prob, average='micro'),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, pred, average='micro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, pred, average='macro', zero_division=0)),
        'micro_precision': float(precision_score(y_true, pred, average='micro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, pred, average='macro', zero_division=0)),
        'micro_recall': float(recall_score(y_true, pred, average='micro', zero_division=0)),
    }


def per_label_metrics(y_true, y_prob, labels, split_name, threshold=0.5):
    pred = (y_prob >= threshold).astype(int)
    rows = []
    for idx, code in enumerate(labels):
        y_i = y_true[:, idx]
        p_i = y_prob[:, idx]
        pred_i = pred[:, idx]
        rows.append({
            'split': split_name,
            'index': idx,
            'code': code,
            'support': int(y_i.sum()),
            'auc': safe_auc(y_i, p_i),
            'f1': float(f1_score(y_i, pred_i, zero_division=0)),
            'precision': float(precision_score(y_i, pred_i, zero_division=0)),
            'recall': float(recall_score(y_i, pred_i, zero_division=0)),
            'threshold': threshold,
        })
    return rows


def probability_stats(y_prob, labels, split_name):
    rows = []
    quantiles = np.quantile(y_prob, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    for idx, code in enumerate(labels):
        values = y_prob[:, idx]
        rows.append({
            'split': split_name,
            'index': idx,
            'code': code,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'q05': float(quantiles[0, idx]),
            'q25': float(quantiles[1, idx]),
            'q50': float(quantiles[2, idx]),
            'q75': float(quantiles[3, idx]),
            'q95': float(quantiles[4, idx]),
            'max': float(np.max(values)),
        })
    return rows


def optimize_global_threshold(y_val, p_val, thresholds):
    best_threshold, best_score = 0.5, -1.0
    for threshold in thresholds:
        score = metrics_at_threshold(y_val, p_val, threshold)['macro_f1']
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def optimize_per_class_thresholds(y_val, p_val, thresholds):
    per_class = []
    for idx in range(y_val.shape[1]):
        best_threshold, best_score = 0.5, -1.0
        for threshold in thresholds:
            pred = (p_val[:, idx] >= threshold).astype(int)
            score = float(f1_score(y_val[:, idx], pred, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        per_class.append(best_threshold)
    return np.asarray(per_class, dtype=np.float32)


def metrics_at_per_class_thresholds(y_true, y_prob, thresholds):
    pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    return {
        'macro_auc': safe_auc(y_true, y_prob, average='macro'),
        'micro_auc': safe_auc(y_true, y_prob, average='micro'),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, pred, average='micro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, pred, average='macro', zero_division=0)),
        'micro_precision': float(precision_score(y_true, pred, average='micro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, pred, average='macro', zero_division=0)),
        'micro_recall': float(recall_score(y_true, pred, average='micro', zero_division=0)),
    }


def read_final_metrics(run_name):
    run_path = Path('results') / 'georgia' / run_name / 'final_metrics.csv'
    if run_path.is_file():
        return pd.read_csv(run_path).to_dict(orient='records')
    root_path = Path('results') / 'final_test_georgia.csv'
    if root_path.is_file():
        df = pd.read_csv(root_path)
        if 'run_tag' in df.columns:
            matched = df[df['run_tag'].astype(str) == run_name]
            if not matched.empty:
                return matched.tail(1).to_dict(orient='records')
        return df.tail(1).to_dict(orient='records')
    return []


def preprocess_stats(cache_dir, x_splits):
    signals_path = Path(cache_dir) / 'signals.npy'
    raw = np.load(signals_path, mmap_mode='r')
    stats = {
        'cache_signals_shape': list(raw.shape),
        'expected_shape_per_record': [TARGET_LENGTH, len(LEAD_ORDER)],
        'lead_order': LEAD_ORDER,
        'sampling_frequency_after_resample': TARGET_FS,
        'all_final_shape_1000x12': bool(raw.ndim == 3 and raw.shape[1:] == (TARGET_LENGTH, len(LEAD_ORDER))),
        'raw_has_nan': bool(np.isnan(raw).any()),
        'raw_has_inf': bool(np.isinf(raw).any()),
        'raw_global_min': float(np.min(raw)),
        'raw_global_max': float(np.max(raw)),
        'raw_global_mean': float(np.mean(raw)),
        'raw_global_std': float(np.std(raw)),
        'normalization': 'train-split global z-score: (x - train_mean) / train_std',
        'ptbxl_consistency': 'Matches PTB-XL preprocess_signals style: scaler fit on train values, applied to val/test.',
    }
    for split, x in x_splits.items():
        lead_mean = np.mean(x, axis=(0, 1))
        lead_std = np.std(x, axis=(0, 1))
        stats[f'{split}_has_nan'] = bool(np.isnan(x).any())
        stats[f'{split}_has_inf'] = bool(np.isinf(x).any())
        stats[f'{split}_global_min'] = float(np.min(x))
        stats[f'{split}_global_max'] = float(np.max(x))
        stats[f'{split}_lead_mean_min'] = float(np.min(lead_mean))
        stats[f'{split}_lead_mean_max'] = float(np.max(lead_mean))
        stats[f'{split}_lead_std_min'] = float(np.min(lead_std))
        stats[f'{split}_lead_std_max'] = float(np.max(lead_std))
        stats[f'{split}_lead_stats'] = [
            {'lead': lead, 'mean': float(lead_mean[i]), 'std': float(lead_std[i])}
            for i, lead in enumerate(LEAD_ORDER)
        ]
    return stats


def config_summary(num_classes):
    return {
        'target_mode': getattr(config, 'target_mode', None),
        'loss_name': getattr(config, 'loss_name', None),
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'num_classes': int(num_classes),
        'use_fdconv_scale': getattr(config, 'use_fdconv_scale', None),
        'use_frequency_aux': getattr(config, 'use_frequency_aux', None),
        'use_ema': getattr(config, 'use_ema', None),
        'use_swa': getattr(config, 'use_swa', None),
        'mutual_feat_weight': getattr(config, 'mutual_feat_weight', None),
        'mutual_logit_weight': getattr(config, 'mutual_logit_weight', None),
        'fusion_mode': getattr(config, 'fusion_mode', None),
        'selection_score_mode': getattr(config, 'selection_score_mode', None),
        'final_eval_mode': getattr(config, 'final_eval_mode', None),
        'lr': getattr(config, 'lr', None),
        'weight_decay': getattr(config, 'weight_decay', None),
        'max_epoch': getattr(config, 'max_epoch', None),
        'batch_size': getattr(config, 'batch_size', None),
        'activation': 'sigmoid' if getattr(config, 'target_mode', '') == 'multilabel' else 'softmax',
        'is_cpsc_single_label_ce': bool(
            getattr(config, 'target_mode', '') == 'single_label' or
            str(getattr(config, 'loss_name', '')).lower() == 'ce'
        ),
    }


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Analyze Georgia predictions and thresholds.')
    parser.add_argument('--datafolder', default='/root/autodl-tmp/data/georgia')
    parser.add_argument('--run_name', default='georgia_min100_seed1')
    parser.add_argument('--model_name', default='ufml_net')
    parser.add_argument('--experiment', default='exp0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--use_fdconv_scale', action='store_true', default=True)
    parser.add_argument('--use_frequency_aux', action='store_true', default=True)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--use_swa', action='store_true', default=False)
    parser.add_argument('--mutual_feat_weight', type=float, default=0.005)
    parser.add_argument('--mutual_logit_weight', type=float, default=0.05)
    parser.add_argument('--fusion_mode', default='standard')
    parser.add_argument('--selection_score_mode', default='fusion_auc_f1')
    parser.add_argument('--training_selection_mode', default='standard_val')
    args = parser.parse_args()

    set_georgia_config(args)
    output_dir = Path(args.output_dir or Path('results') / 'georgia' / args.run_name / 'diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(
        datafolder=args.datafolder,
        min_label_count=getattr(config, 'georgia_min_label_count', 100),
        seed=args.seed,
    )
    cache_dir = get_cache_dir(getattr(config, 'georgia_min_label_count', 100))
    label_map, labels, label_counts = load_label_map(cache_dir)
    split_payload = load_split_payload(cache_dir, seed=args.seed, min_label_count=getattr(config, 'georgia_min_label_count', 100))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = find_checkpoint(args)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, len(labels), device)

    loaders = {
        'train': make_loader(x_train, y_train, args.batch_size),
        'val': make_loader(x_val, y_val, args.batch_size),
        'test': make_loader(x_test, y_test, args.batch_size),
    }
    predictions = {split: evaluate_split(model, loader, device) for split, loader in loaders.items()}

    np.savez_compressed(
        output_dir / 'predictions.npz',
        **{
            f'{split}_{key}': value
            for split, values in predictions.items()
            for key, value in values.items()
        }
    )

    per_label_rows = []
    prob_rows = []
    split_metrics = {}
    for split, values in predictions.items():
        y_true = values['y_true']
        y_prob = values['probs_fusion']
        per_label_rows.extend(per_label_metrics(y_true, y_prob, labels, split, threshold=0.5))
        prob_rows.extend(probability_stats(y_prob, labels, split))
        split_metrics[split] = {
            'num_samples': int(y_true.shape[0]),
            'default_threshold_0.5': metrics_at_threshold(y_true, y_prob, 0.5),
        }

    per_label_df = pd.DataFrame(per_label_rows)
    prob_df = pd.DataFrame(prob_rows)
    per_label_df.to_csv(output_dir / 'per_label_metrics_georgia.csv', index=False)
    prob_df.to_csv(output_dir / 'probability_stats_georgia.csv', index=False)

    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    val_y = predictions['val']['y_true']
    val_p = predictions['val']['probs_fusion']
    test_y = predictions['test']['y_true']
    test_p = predictions['test']['probs_fusion']
    global_threshold, global_val_macro_f1 = optimize_global_threshold(val_y, val_p, thresholds)
    per_class_thresholds = optimize_per_class_thresholds(val_y, val_p, thresholds)
    threshold_payload = {
        'search_grid': thresholds.tolist(),
        'default_threshold': 0.5,
        'default_test_metrics': metrics_at_threshold(test_y, test_p, 0.5),
        'global_threshold': global_threshold,
        'global_val_macro_f1': global_val_macro_f1,
        'global_test_metrics': metrics_at_threshold(test_y, test_p, global_threshold),
        'per_class_thresholds': [
            {'index': i, 'code': code, 'threshold': float(per_class_thresholds[i])}
            for i, code in enumerate(labels)
        ],
        'per_class_test_metrics': metrics_at_per_class_thresholds(test_y, test_p, per_class_thresholds),
    }
    write_json(output_dir / 'thresholds_georgia.json', threshold_payload)

    preprocess_payload = preprocess_stats(
        cache_dir,
        {'train': x_train, 'val': x_val, 'test': x_test},
    )
    write_json(output_dir / 'preprocess_stats_georgia.json', preprocess_payload)

    final_metrics_records = read_final_metrics(args.run_name)
    if final_metrics_records:
        pd.DataFrame(final_metrics_records).to_csv(output_dir / 'final_metrics.csv', index=False)

    low_f1 = (
        per_label_df[per_label_df['split'] == 'test']
        .sort_values(['f1', 'support'], ascending=[True, True])
        .head(10)
        .to_dict(orient='records')
    )
    low_auc = (
        per_label_df[per_label_df['split'] == 'test']
        .sort_values(['auc', 'support'], ascending=[True, True])
        .head(10)
        .to_dict(orient='records')
    )
    support = np.asarray([row['support'] for row in per_label_rows if row['split'] == 'test'], dtype=np.float32)
    f1_values = np.asarray([row['f1'] for row in per_label_rows if row['split'] == 'test'], dtype=np.float32)
    support_f1_corr = float(np.corrcoef(support, f1_values)[0, 1]) if len(support) > 1 else float('nan')

    summary = {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_epoch': int(checkpoint.get('global_epoch', -1)) if isinstance(checkpoint, dict) else None,
        'cache_dir': str(cache_dir),
        'label_map': label_map,
        'split_summary': {
            split: {
                'num_samples': split_payload['splits'][split]['num_samples'],
                'label_positive_counts': split_payload['splits'][split]['label_positive_counts'],
                'min_positive_count': split_payload['splits'][split]['min_positive_count'],
            }
            for split in ('train', 'val', 'test')
        },
        'split_metrics': split_metrics,
        'threshold_optimization': threshold_payload,
        'lowest_test_f1_labels': low_f1,
        'lowest_test_auc_labels': low_auc,
        'test_support_f1_correlation': support_f1_corr,
        'long_tail_drag_signal': support_f1_corr > 0.25,
        'default_threshold': 0.5,
        'config_summary': config_summary(len(labels)),
        'preprocess_summary': {
            key: value
            for key, value in preprocess_payload.items()
            if key != 'train_lead_stats' and key != 'val_lead_stats' and key != 'test_lead_stats'
        },
        'final_metrics_records': final_metrics_records,
    }
    write_json(output_dir / 'diagnostics_summary_georgia.json', summary)

    print('Georgia diagnostics saved to:', output_dir)
    print('Checkpoint:', checkpoint_path)
    print('Default test metrics:', threshold_payload['default_test_metrics'])
    print('Best global threshold:', global_threshold)
    print('Global-threshold test metrics:', threshold_payload['global_test_metrics'])
    print('Per-class-threshold test metrics:', threshold_payload['per_class_test_metrics'])
    print('Lowest test F1 labels:', [(row['code'], row['support'], round(row['f1'], 4)) for row in low_f1[:5]])
    print('Lowest test AUC labels:', [(row['code'], row['support'], round(row['auc'], 4)) for row in low_auc[:5]])


if __name__ == '__main__':
    main()
