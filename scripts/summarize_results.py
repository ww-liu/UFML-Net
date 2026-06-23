#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from glob import glob

import pandas as pd


SUMMARY_COLUMNS = [
    'dataset',
    'run_name',
    'model_name',
    'target_mode',
    'loss_name',
    'use_fdconv_scale',
    'use_frequency_aux',
    'mutual_feat_weight',
    'mutual_logit_weight',
    'use_ema',
    'use_swa',
    'fusion_mode',
    'fusion_residual_weight',
    'branch1_auc',
    'branch1_f1',
    'branch2_auc',
    'branch2_f1',
    'fusion_auc',
    'fusion_f1',
    'best_epoch',
    'seed',
]


def read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def read_run(run_dir):
    final_metrics_path = os.path.join(run_dir, 'final_metrics.csv')
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(final_metrics_path) or not os.path.exists(config_path):
        return None
    metrics = pd.read_csv(final_metrics_path)
    if metrics.empty:
        return None
    row = metrics.iloc[-1].to_dict()
    cfg = read_json(config_path)
    merged = {**cfg, **row}
    merged['run_name'] = cfg.get('run_name') or row.get('run_name') or os.path.basename(run_dir)
    merged['dataset'] = cfg.get('dataset') or row.get('dataset') or os.path.basename(os.path.dirname(run_dir))
    return merged


def collect_runs(results_dir):
    rows = []
    for final_metrics_path in glob(os.path.join(results_dir, '*', '*', 'final_metrics.csv')):
        run_dir = os.path.dirname(final_metrics_path)
        row = read_run(run_dir)
        if row:
            rows.append(row)
    return rows


def ablation_key(run_name):
    name = str(run_name)
    for token in ('_PTB_', '_CPSC_'):
        name = name.replace(token, '_')
    if name.startswith('E0_'):
        return 'E0'
    parts = name.split('_')
    if len(parts) >= 2 and parts[0].startswith('A'):
        return parts[0] + '_' + '_'.join(parts[1:])
    if len(parts) >= 2 and parts[0].startswith('F'):
        return parts[0] + '_' + '_'.join(parts[1:])
    return name


def first_row(df, dataset, run_name):
    matches = df[(df['dataset'] == dataset) & (df['run_name'] == run_name)]
    if matches.empty:
        return None
    return matches.iloc[-1]


def build_stability(summary):
    rows = []
    ptb_base = first_row(summary, 'ptbxl', 'E0_PTB_FULL')
    cpsc_base = first_row(summary, 'cpsc', 'E0_CPSC_SOFTMAX_CE')
    if ptb_base is None or cpsc_base is None:
        return pd.DataFrame(rows)

    summary = summary.copy()
    summary['ablation_key'] = summary['run_name'].map(ablation_key)
    for key, group in summary.groupby('ablation_key'):
        if key == 'E0':
            continue
        ptb = group[group['dataset'] == 'ptbxl']
        cpsc = group[group['dataset'] == 'cpsc']
        ptb_delta = None
        cpsc_delta = None
        status = 'dataset-specific ablation'
        if not ptb.empty:
            ptb_delta = float(ptb.iloc[-1]['fusion_auc']) - float(ptb_base['fusion_auc'])
        if not cpsc.empty:
            cpsc_delta = float(cpsc.iloc[-1]['fusion_auc']) - float(cpsc_base['fusion_auc'])
        if ptb_delta is not None and cpsc_delta is not None:
            if ptb_delta > 0 and cpsc_delta < -0.001:
                status = 'PTB-specific'
            elif ptb_delta >= -0.001 and cpsc_delta >= -0.001:
                status = 'cross-dataset stable'
            elif cpsc_delta < -0.001:
                status = 'not recommended'
            else:
                status = 'mixed'
        rows.append({
            'ablation_key': key,
            'ptb_run': ptb.iloc[-1]['run_name'] if not ptb.empty else '',
            'cpsc_run': cpsc.iloc[-1]['run_name'] if not cpsc.empty else '',
            'ptb_delta_fusion_auc': ptb_delta,
            'cpsc_delta_fusion_auc': cpsc_delta,
            'recommendation': status,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Summarize run-level ablation results.')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()

    rows = collect_runs(args.results_dir)
    if not rows:
        print('No run-level final_metrics.csv files found.')
        return

    summary = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in summary.columns:
            summary[col] = pd.NA
    summary = summary[SUMMARY_COLUMNS].sort_values(['dataset', 'run_name']).reset_index(drop=True)
    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, 'summary_all_experiments.csv')
    summary.to_csv(summary_path, index=False)
    print(f'Summary saved to: {summary_path}')

    stability = build_stability(summary)
    stability_path = os.path.join(args.results_dir, 'summary_ptb_cpsc_stability.csv')
    stability.to_csv(stability_path, index=False)
    print(f'Stability summary saved to: {stability_path}')


if __name__ == '__main__':
    main()
