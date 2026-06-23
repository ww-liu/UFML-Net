#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from glob import glob

import pandas as pd


SEEDS = [1, 2, 3]
METRICS = [
    'branch1_auc',
    'branch1_f1',
    'branch2_auc',
    'branch2_f1',
    'fusion_auc',
    'fusion_f1',
]

CORE_SPECS = [
    {
        'dataset': 'ptbxl',
        'ablation': 'full',
        'display_name': 'Full UFML-Net',
        'run_patterns': ['PTB_FULL_seed{seed}'],
        'legacy_run_names': ['E0_PTB_FULL'],
        'legacy_run_tags': ['freqaux_ref'],
    },
    {
        'dataset': 'ptbxl',
        'ablation': 'no_fdconv',
        'display_name': 'w/o FDConvScale',
        'run_patterns': ['PTB_no_fdconv_seed{seed}'],
        'legacy_run_names': ['ptb9471_no_fdconv', 'A1_PTB_no_fdconv'],
        'legacy_run_tags': ['ptb9471_no_fdconv'],
    },
    {
        'dataset': 'ptbxl',
        'ablation': 'no_freqaux',
        'display_name': 'w/o FrequencyAux',
        'run_patterns': ['PTB_no_freqaux_seed{seed}'],
        'legacy_run_names': ['ptb9471_no_freqaux', 'A2_PTB_no_freqaux'],
        'legacy_run_tags': ['ptb9471_no_freqaux'],
    },
    {
        'dataset': 'ptbxl',
        'ablation': 'no_mutual',
        'display_name': 'w/o Mutual Learning',
        'run_patterns': ['PTB_no_mutual_seed{seed}'],
        'legacy_run_names': ['ptb9471_no_mutual', 'A4_PTB_no_mutual'],
        'legacy_run_tags': ['ptb9471_no_mutual'],
    },
    {
        'dataset': 'cpsc',
        'ablation': 'full',
        'display_name': 'CPSC default recipe',
        'run_patterns': ['CPSC_DEFAULT_seed{seed}'],
        'legacy_run_names': ['E0_CPSC_SOFTMAX_CE'],
        'legacy_run_tags': ['cpsc_softmax_ce_weighted_full'],
    },
    {
        'dataset': 'cpsc',
        'ablation': 'no_fdconv',
        'display_name': 'w/o FDConvScale',
        'run_patterns': ['CPSC_no_fdconv_seed{seed}'],
        'legacy_run_names': ['A1_CPSC_no_fdconv'],
        'legacy_run_tags': [],
    },
    {
        'dataset': 'cpsc',
        'ablation': 'freqaux_on',
        'display_name': '+ FrequencyAux',
        'run_patterns': ['CPSC_freqaux_on_seed{seed}'],
        'legacy_run_names': ['A3_CPSC_freqaux_on'],
        'legacy_run_tags': [],
    },
    {
        'dataset': 'cpsc',
        'ablation': 'no_mutual',
        'display_name': 'w/o Mutual Learning',
        'run_patterns': ['CPSC_no_mutual_seed{seed}'],
        'legacy_run_names': ['A4_CPSC_no_mutual'],
        'legacy_run_tags': [],
    },
]


def canonical_dataset(value):
    value = str(value or '').lower()
    if 'ptb' in value:
        return 'ptbxl'
    if 'cpsc' in value or 'tmp_data' in value:
        return 'cpsc'
    return value


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float('nan')


def safe_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def collect_run_level(results_dir):
    rows = []
    for final_path in glob(os.path.join(results_dir, '*', '*', 'final_metrics.csv')):
        run_dir = os.path.dirname(final_path)
        config_path = os.path.join(run_dir, 'config.json')
        metrics = pd.read_csv(final_path)
        if metrics.empty:
            continue
        row = metrics.iloc[-1].to_dict()
        if os.path.exists(config_path):
            row = {**read_json(config_path), **row}
        row['source'] = 'run_dir'
        row['run_name'] = row.get('run_name') or os.path.basename(run_dir)
        row['dataset'] = canonical_dataset(row.get('dataset') or os.path.basename(os.path.dirname(run_dir)))
        rows.append(row)
    return rows


def collect_flat_csv(results_dir):
    rows = []
    patterns = [
        'results_ptbxl*.csv',
        'final_test_ptbxl*.csv',
        'results_cpsc*.csv',
        'final_test_cpsc*.csv',
    ]
    for pattern in patterns:
        for path in glob(os.path.join(results_dir, pattern)):
            try:
                df = pd.read_csv(path, on_bad_lines='skip')
            except Exception:
                continue
            for _, row in df.iterrows():
                item = row.to_dict()
                item['source'] = os.path.basename(path)
                item['dataset'] = canonical_dataset(item.get('dataset'))
                rows.append(item)
    return rows


def candidate_score(row, spec, seed):
    if canonical_dataset(row.get('dataset')) != spec['dataset']:
        return None
    if safe_int(row.get('seed')) != seed:
        return None
    if str(row.get('model_name', '')).lower() != 'ufml_net':
        return None
    if str(row.get('experiment', 'exp0')) != 'exp0':
        return None

    run_name = str(row.get('run_name', '') or '')
    run_tag = str(row.get('run_tag', '') or '')
    canonical_names = [pattern.format(seed=seed) for pattern in spec['run_patterns']]
    if run_name in canonical_names:
        return (0, str(row.get('timestamp', '')))
    if run_name in spec['legacy_run_names']:
        return (1, str(row.get('timestamp', '')))
    if run_tag in spec['legacy_run_tags']:
        return (2, str(row.get('timestamp', '')))
    return None


def select_row(rows, spec, seed):
    matches = []
    for row in rows:
        score = candidate_score(row, spec, seed)
        if score is not None:
            matches.append((score, row))
    if not matches:
        return None
    best_rank = min(score[0] for score, _ in matches)
    best_matches = [(score, row) for score, row in matches if score[0] == best_rank]
    best_matches.sort(key=lambda item: item[0][1])
    return best_matches[-1][1]


def mean_std(series):
    series = pd.Series(series, dtype='float64').dropna()
    if series.empty:
        return float('nan'), float('nan')
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    return mean, std


def _fmt_mean_std_legacy(mean, std):
    if pd.isna(mean):
        return ''
    return f'{mean:.6f} ± {std:.6f}'


def fmt_mean_std(mean, std):
    if pd.isna(mean):
        return ''
    return f'{mean:.6f} \u00b1 {std:.6f}'


def build_tables(rows):
    per_seed = []
    for spec in CORE_SPECS:
        for seed in SEEDS:
            row = select_row(rows, spec, seed)
            if row is None:
                continue
            record = {
                'dataset': spec['dataset'],
                'ablation': spec['ablation'],
                'display_name': spec['display_name'],
                'seed': seed,
                'run_name': row.get('run_name', ''),
                'run_tag': row.get('run_tag', ''),
                'source': row.get('source', ''),
                'best_epoch': safe_int(row.get('best_epoch')),
            }
            for metric in METRICS:
                record[metric] = safe_float(row.get(metric))
            per_seed.append(record)

    per_seed_df = pd.DataFrame(per_seed)
    if per_seed_df.empty:
        return per_seed_df, pd.DataFrame()

    full_lookup = {
        (row['dataset'], row['seed']): row
        for _, row in per_seed_df[per_seed_df['ablation'] == 'full'].iterrows()
    }
    for metric in METRICS:
        deltas = []
        for _, row in per_seed_df.iterrows():
            full = full_lookup.get((row['dataset'], row['seed']))
            if full is None:
                deltas.append(float('nan'))
            else:
                deltas.append(float(row[metric]) - float(full[metric]))
        per_seed_df[f'delta_{metric}_vs_full'] = deltas

    summary_rows = []
    for (dataset, ablation, display_name), group in per_seed_df.groupby(['dataset', 'ablation', 'display_name'], sort=False):
        summary = {
            'dataset': dataset,
            'ablation': ablation,
            'display_name': display_name,
            'n': int(group['seed'].nunique()),
            'seeds': ','.join(str(seed) for seed in sorted(group['seed'].unique())),
            'run_names': ';'.join(str(name) for name in group['run_name'].fillna('').tolist()),
        }
        for metric in METRICS:
            mean, std = mean_std(group[metric])
            summary[f'{metric}_mean'] = mean
            summary[f'{metric}_std'] = std
            summary[f'{metric}_mean_std'] = fmt_mean_std(mean, std)
            d_mean, d_std = mean_std(group[f'delta_{metric}_vs_full'])
            summary[f'delta_{metric}_vs_full_mean'] = d_mean
            summary[f'delta_{metric}_vs_full_std'] = d_std
            summary[f'delta_{metric}_vs_full_mean_std'] = fmt_mean_std(d_mean, d_std)
        summary_rows.append(summary)
    return per_seed_df, pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description='Summarize core PTB/CPSC multi-seed runs.')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    rows = collect_run_level(args.results_dir) + collect_flat_csv(args.results_dir)
    per_seed, summary = build_tables(rows)
    os.makedirs(args.results_dir, exist_ok=True)

    output = args.output or os.path.join(args.results_dir, 'summary_core_multiseed.csv')
    summary.to_csv(output, index=False)
    per_seed_path = os.path.join(args.results_dir, 'summary_core_multiseed_per_seed.csv')
    per_seed.to_csv(per_seed_path, index=False)
    print(f'Core multi-seed summary saved to: {output}')
    print(f'Per-seed core rows saved to: {per_seed_path}')

    if not summary.empty:
        print(summary[['dataset', 'ablation', 'n', 'seeds', 'fusion_auc_mean_std', 'fusion_f1_mean_std']].to_string(index=False))


if __name__ == '__main__':
    main()
