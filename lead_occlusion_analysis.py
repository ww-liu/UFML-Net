# -*- coding: utf-8 -*-

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import config
from dataset import load_datasets
from main_mul import (
    apply_dataset_overrides,
    apply_experiment_overrides,
    compute_metrics,
    fuse_branch_predictions,
    get_artifact_dataset_tag,
    get_result_dataset_tag,
    optimize_class_thresholds,
    setup_seed,
    unpack_model_outputs,
)
from models.duibi import Dui1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def build_model(num_classes):
    model = Dui1(
        num_classes=num_classes,
        use_shape_enhance=getattr(config, 'use_shape_enhance', True),
        shape_gamma_init=getattr(config, 'shape_gamma_init', 0.01),
        use_fdconv_scale=getattr(config, 'use_fdconv_scale', True),
        use_lead_calibration=getattr(config, 'use_lead_calibration', True),
        lead_gamma_init=getattr(config, 'lead_gamma_init', 0.01),
        use_lead_relation_graph=getattr(config, 'use_lead_relation_graph', True),
        lead_relation_graph_dim=getattr(config, 'lead_relation_graph_dim', 16),
        lead_relation_graph_gamma_init=getattr(config, 'lead_relation_graph_gamma_init', 0.003),
        use_frequency_gate=getattr(config, 'use_frequency_gate', True),
        frequency_gamma_init=getattr(config, 'frequency_gamma_init', 0.02),
        use_frequency_aux=getattr(config, 'use_frequency_aux', True),
        frequency_aux_gamma_init=getattr(config, 'frequency_aux_gamma_init', 0.03),
        frequency_aux_uncertainty_power=getattr(config, 'frequency_aux_uncertainty_power', 1.5),
        frequency_aux_disagreement_power=getattr(config, 'frequency_aux_disagreement_power', 1.0),
        use_tf_branch=getattr(config, 'use_tf_branch', True),
        tf_n_fft=getattr(config, 'tf_n_fft', 64),
        tf_hop_length=getattr(config, 'tf_hop_length', 16),
        tf_win_length=getattr(config, 'tf_win_length', 64),
        use_category_fusion=getattr(config, 'use_category_fusion', True),
    )
    return model.to(device)


def load_best_checkpoint(model, artifact_dataset_tag):
    checkpoint_path = os.path.join(
        'checkpoints',
        f'{config.model_name}_{artifact_dataset_tag}_{config.experiment}_checkpoint_best.pth'
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint: {checkpoint_path}')
    return model


def collect_predictions(model, dataloader, mask_lead=None):
    model.eval()
    outputs1, outputs2, outputsf, targets = [], [], [], []
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs = inputs.to(device)
            if mask_lead is not None:
                inputs = inputs.clone()
                inputs[:, :, mask_lead] = 0.0
            target = target.to(device)

            feat1, out1, feat2, out2, tf_logits, category_fusion_logits = unpack_model_outputs(model(inputs))
            prob1, prob2, probf = fuse_branch_predictions(
                out1.float(),
                out2.float(),
                tf_logits=None if tf_logits is None else tf_logits.float(),
                category_fusion_logits=None if category_fusion_logits is None else category_fusion_logits.float()
            )
            outputs1.append(prob1.cpu().numpy())
            outputs2.append(prob2.cpu().numpy())
            outputsf.append(probf.cpu().numpy())
            targets.append(target.cpu().numpy())
    return {
        'branch1': np.concatenate(outputs1, axis=0),
        'branch2': np.concatenate(outputs2, axis=0),
        'fusion': np.concatenate(outputsf, axis=0),
        'target': np.concatenate(targets, axis=0),
    }


def class_auc(y_true, y_pred):
    scores = []
    for class_idx in range(y_true.shape[1]):
        if len(np.unique(y_true[:, class_idx])) < 2:
            scores.append(np.nan)
        else:
            scores.append(roc_auc_score(y_true[:, class_idx], y_pred[:, class_idx]))
    return np.array(scores, dtype=np.float32)


def main():
    setup_seed(config.seed)
    result_dataset_tag = get_result_dataset_tag(config.datafolder)
    artifact_dataset_tag = get_artifact_dataset_tag(config.datafolder)
    apply_dataset_overrides(config, result_dataset_tag)
    apply_experiment_overrides(config)

    train_loader, val_loader, test_loader, num_classes = load_datasets(
        experiment=config.experiment,
        datafolder=config.datafolder
    )
    model = build_model(num_classes)
    model = load_best_checkpoint(model, artifact_dataset_tag)

    val_pred = collect_predictions(model, val_loader)
    thresholds = {
        'branch1': optimize_class_thresholds(val_pred['target'], val_pred['branch1']),
        'branch2': optimize_class_thresholds(val_pred['target'], val_pred['branch2']),
        'fusion': optimize_class_thresholds(val_pred['target'], val_pred['fusion']),
    }

    baseline = collect_predictions(model, test_loader)
    y_true = baseline['target']
    baseline_metrics = {
        name: compute_metrics(y_true, baseline[name], thresholds=thresholds[name])
        for name in ['branch1', 'branch2', 'fusion']
    }
    baseline_class_auc = class_auc(y_true, baseline['fusion'])

    rows = []
    class_drop = []
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        masked = collect_predictions(model, test_loader, mask_lead=lead_idx)
        row = {'lead': lead_name, 'lead_idx': lead_idx}
        for name in ['branch1', 'branch2', 'fusion']:
            auc, tpr, acc, f1 = compute_metrics(y_true, masked[name], thresholds=thresholds[name])
            base_auc, base_tpr, base_acc, base_f1 = baseline_metrics[name]
            row[f'{name}_auc'] = auc
            row[f'{name}_auc_drop'] = base_auc - auc
            row[f'{name}_tpr_drop'] = base_tpr - tpr
            row[f'{name}_acc_drop'] = base_acc - acc
            row[f'{name}_f1_drop'] = base_f1 - f1
        rows.append(row)

        masked_class_auc = class_auc(y_true, masked['fusion'])
        class_drop.append(baseline_class_auc - masked_class_auc)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    prefix = f'lead_occlusion_{result_dataset_tag}_{config.experiment}'

    overall_path = os.path.join(results_dir, f'{prefix}_overall.csv')
    pd.DataFrame(rows).to_csv(overall_path, index=False)

    class_rows = []
    class_drop_array = np.stack(class_drop, axis=1)
    for class_idx in range(num_classes):
        row = {'class': f'class_{class_idx}'}
        for lead_idx, lead_name in enumerate(LEAD_NAMES):
            row[lead_name] = class_drop_array[class_idx, lead_idx]
        class_rows.append(row)
    matrix_path = os.path.join(results_dir, f'{prefix}_class_auc_drop.csv')
    pd.DataFrame(class_rows).to_csv(matrix_path, index=False)

    threshold_path = os.path.join(results_dir, f'thresholds_{result_dataset_tag}_{config.experiment}.csv')
    threshold_df = pd.DataFrame({
        'class': [f'class_{i}' for i in range(num_classes)],
        'branch1_threshold': thresholds['branch1'],
        'branch2_threshold': thresholds['branch2'],
        'fusion_threshold': thresholds['fusion'],
    })
    threshold_df.to_csv(threshold_path, index=False)

    print(f'Saved lead occlusion overall results to: {overall_path}')
    print(f'Saved class-lead AUC drop matrix to: {matrix_path}')
    print(f'Saved validation thresholds to: {threshold_path}')


if __name__ == '__main__':
    main()
