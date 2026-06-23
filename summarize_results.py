import argparse
import os
from glob import glob

import pandas as pd


SUMMARY_COLUMNS = [
    'dataset', 'experiment', 'model_name', 'seed', 'best_epoch',
    'selection_score_mode', 'training_selection_mode', 'active_modules', 'run_note',
    'branch1_auc', 'branch1_tpr', 'branch1_acc', 'branch1_f1',
    'branch2_auc', 'branch2_tpr', 'branch2_acc', 'branch2_f1',
    'fusion_auc', 'fusion_tpr', 'fusion_acc', 'fusion_f1',
    'best_single_auc', 'best_single_branch',
    'timestamp', 'source_file',
]


def read_csv_lenient(path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        try:
            return pd.read_csv(path, engine='python', on_bad_lines='skip')
        except Exception as fallback_exc:
            print(f'skip invalid csv: {path} ({exc}; fallback: {fallback_exc})')
            return None


def collect_final_test_rows(results_dir, dataset_tag):
    patterns = [
        os.path.join(results_dir, f'final_test_{dataset_tag}_*.csv'),
        os.path.join(results_dir, f'final_test_v2_{dataset_tag}_*.csv'),
    ]
    rows = []
    for pattern in patterns:
        for path in sorted(glob(pattern)):
            df = read_csv_lenient(path)
            if df is None or len(df) == 0:
                continue
            df = df.copy()
            df['source_file'] = os.path.basename(path)
            rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True, sort=False)


def collect_results_rows(results_dir, dataset_tag):
    paths = [
        os.path.join(results_dir, f'results_{dataset_tag}.csv'),
        os.path.join(results_dir, f'results_{dataset_tag}_v2.csv'),
        os.path.join(results_dir, f'results_v2_{dataset_tag}.csv'),
    ]
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        df = read_csv_lenient(path)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df['source_file'] = os.path.basename(path)
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True, sort=False)


def normalize_result_columns(df, dataset_tag):
    df = df.copy()
    if 'dataset' not in df.columns:
        df['dataset'] = dataset_tag
    else:
        df['dataset'] = df['dataset'].fillna(dataset_tag)
    if 'experiment' in df.columns:
        df = df[df['experiment'].astype(str).str.startswith('exp')].copy()

    metric_cols = [
        'branch1_auc', 'branch1_tpr', 'branch1_acc', 'branch1_f1',
        'branch2_auc', 'branch2_tpr', 'branch2_acc', 'branch2_f1',
        'fusion_auc', 'fusion_tpr', 'fusion_acc', 'fusion_f1',
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['best_epoch', 'seed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['selection_score_mode', 'training_selection_mode', 'active_modules', 'run_note']:
        if col not in df.columns:
            df[col] = ''

    auc_cols = [col for col in ['branch1_auc', 'branch2_auc', 'fusion_auc'] if col in df.columns]
    if auc_cols:
        df['best_single_auc'] = df[auc_cols].max(axis=1)
        best_branch = df[auc_cols].apply(
            lambda row: row.dropna().idxmax() if not row.dropna().empty else '',
            axis=1
        )
        df['best_single_branch'] = best_branch.astype(str).str.replace('_auc', '', regex=False)
    else:
        df['best_single_auc'] = pd.NA
        df['best_single_branch'] = ''

    if 'timestamp' in df.columns:
        df['_timestamp_sort'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['timestamp'] = ''
        df['_timestamp_sort'] = pd.NaT

    return df


def keep_latest_per_experiment(df):
    sort_cols = ['experiment', '_timestamp_sort']
    if 'source_file' in df.columns:
        sort_cols.append('source_file')
    df = df.sort_values(sort_cols, na_position='first')
    return df.groupby('experiment', as_index=False).tail(1).reset_index(drop=True)


def build_summary(df, dataset_tag, latest_only):
    df = normalize_result_columns(df, dataset_tag)
    auc_cols = [col for col in ['branch1_auc', 'branch2_auc', 'fusion_auc'] if col in df.columns]
    if auc_cols:
        df = df.dropna(subset=auc_cols, how='all')
    if latest_only and 'experiment' in df.columns:
        df = keep_latest_per_experiment(df)

    available_cols = [col for col in SUMMARY_COLUMNS if col in df.columns]
    summary = df[available_cols].copy()
    if 'experiment' in summary.columns:
        summary = summary.sort_values(by='experiment').reset_index(drop=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment results into one clean table.')
    parser.add_argument('--dataset', required=True, help='Dataset tag, e.g. ptbxl or cpsc')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--all-runs', action='store_true', help='Keep every run instead of only the latest row per experiment')
    args = parser.parse_args()

    dataset_tag = args.dataset.strip().lower()
    final_df = collect_final_test_rows(args.results_dir, dataset_tag)
    result_df = collect_results_rows(args.results_dir, dataset_tag)
    frames = [frame for frame in [final_df, result_df] if frame is not None]
    df = pd.concat(frames, ignore_index=True, sort=False) if frames else None

    if df is None:
        print(f'No result files found for dataset: {dataset_tag}')
        return

    summary = build_summary(df, dataset_tag, latest_only=not args.all_runs)
    suffix = 'all' if args.all_runs else 'latest'
    out_path = os.path.join(args.results_dir, f'summary_{dataset_tag}_{suffix}.csv')
    summary.to_csv(out_path, index=False)
    print(f'Summary saved to: {out_path}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
