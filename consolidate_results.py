import argparse
import os
import re
from glob import glob

import pandas as pd


FINAL_METRIC_COLUMNS = [
    'branch1_auc', 'branch1_tpr', 'branch1_acc', 'branch1_f1',
    'branch2_auc', 'branch2_tpr', 'branch2_acc', 'branch2_f1',
    'fusion_auc', 'fusion_tpr', 'fusion_acc', 'fusion_f1',
]

TRAINING_METRIC_COLUMNS = [
    'train_branch1_auc', 'train_branch1_tpr', 'train_branch1_acc', 'train_branch1_f1',
    'train_branch2_auc', 'train_branch2_tpr', 'train_branch2_acc', 'train_branch2_f1',
    'train_fusion_auc', 'train_fusion_tpr', 'train_fusion_acc', 'train_fusion_f1',
    'val_branch1_auc', 'val_branch1_tpr', 'val_branch1_acc', 'val_branch1_f1',
    'val_branch2_auc', 'val_branch2_tpr', 'val_branch2_acc', 'val_branch2_f1',
    'val_fusion_auc', 'val_fusion_tpr', 'val_fusion_acc', 'val_fusion_f1',
    'test_branch1_auc', 'test_branch1_tpr', 'test_branch1_acc', 'test_branch1_f1',
    'test_branch2_auc', 'test_branch2_tpr', 'test_branch2_acc', 'test_branch2_f1',
    'test_fusion_auc', 'test_fusion_tpr', 'test_fusion_acc', 'test_fusion_f1',
    'train_loss', 'val_loss', 'test_loss',
]


def read_csv_lenient(path):
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine='python', on_bad_lines='skip')
        except Exception as exc:
            print(f'skip invalid csv: {path} ({exc})')
            return None


def infer_dataset_from_name(filename):
    name = filename.lower()
    if 'ptbxl' in name:
        return 'ptbxl'
    if 'cpsc' in name:
        return 'cpsc'
    if 'tmp_data' in name:
        return 'tmp_data'
    return 'unknown'


def infer_experiment_from_name(filename):
    match = re.search(r'(exp\d(?:\.\d+)*)', filename.lower())
    return match.group(1) if match else ''


def normalize_common_columns(df, path):
    df = df.copy()
    filename = os.path.basename(path)
    if 'dataset' not in df.columns:
        df['dataset'] = infer_dataset_from_name(filename)
    df['dataset'] = df['dataset'].fillna(infer_dataset_from_name(filename)).astype(str)

    if 'experiment' not in df.columns:
        df['experiment'] = infer_experiment_from_name(filename)
    df['experiment'] = df['experiment'].fillna(infer_experiment_from_name(filename)).astype(str)

    if 'source_file' not in df.columns:
        df['source_file'] = filename

    if 'timestamp' not in df.columns:
        df['timestamp'] = ''
    df['_timestamp_sort'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df


def normalize_final_results(df, path):
    df = normalize_common_columns(df, path)
    for col in FINAL_METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['run_id', 'seed', 'best_epoch']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['selection_score_mode', 'training_selection_mode', 'active_modules', 'run_note']:
        if col not in df.columns:
            df[col] = ''

    auc_cols = [col for col in ['branch1_auc', 'branch2_auc', 'fusion_auc'] if col in df.columns]
    if auc_cols:
        df['best_single_auc'] = df[auc_cols].max(axis=1)
        df['best_single_branch'] = df[auc_cols].apply(
            lambda row: row.dropna().idxmax().replace('_auc', '') if not row.dropna().empty else '',
            axis=1,
        )
    else:
        df['best_single_auc'] = pd.NA
        df['best_single_branch'] = ''
    return df


def normalize_training_log(df, path):
    df = normalize_common_columns(df, path)
    for col in TRAINING_METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['run_id', 'seed', 'epoch']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def collect_final_results(results_dir):
    patterns = [
        os.path.join(results_dir, 'results*.csv'),
        os.path.join(results_dir, 'final_test*.csv'),
    ]
    frames = []
    for pattern in patterns:
        for path in sorted(glob(pattern)):
            filename = os.path.basename(path)
            if filename.startswith('summary_') or filename.startswith('thresholds_'):
                continue
            df = read_csv_lenient(path)
            if df is None or df.empty:
                continue
            frames.append(normalize_final_results(df, path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    auc_cols = [col for col in ['branch1_auc', 'branch2_auc', 'fusion_auc'] if col in df.columns]
    if auc_cols:
        df = df.dropna(subset=auc_cols, how='all')
    return df.reset_index(drop=True)


def collect_training_logs(results_dir):
    frames = []
    for path in sorted(glob(os.path.join(results_dir, 'training_log*.csv'))):
        df = read_csv_lenient(path)
        if df is None or df.empty:
            continue
        frames.append(normalize_training_log(df, path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).reset_index(drop=True)


def build_best_final_tables(final_df):
    if final_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sort_df = final_df.copy()
    sort_df['_fusion_auc_sort'] = pd.to_numeric(sort_df.get('fusion_auc'), errors='coerce')
    sort_df['_best_auc_sort'] = pd.to_numeric(sort_df.get('best_single_auc'), errors='coerce')

    by_fusion = (
        sort_df.sort_values(
            ['dataset', 'experiment', '_fusion_auc_sort', 'fusion_f1', '_timestamp_sort'],
            ascending=[True, True, False, False, False],
            na_position='last',
        )
        .groupby(['dataset', 'experiment'], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    by_single = (
        sort_df.sort_values(
            ['dataset', 'experiment', '_best_auc_sort', 'fusion_f1', '_timestamp_sort'],
            ascending=[True, True, False, False, False],
            na_position='last',
        )
        .groupby(['dataset', 'experiment'], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return cleanup_output(by_fusion), cleanup_output(by_single)


def build_training_best_table(training_df):
    if training_df.empty:
        return pd.DataFrame()

    score_cols = [
        'test_fusion_auc', 'test_fusion_f1', 'val_fusion_auc', 'val_fusion_f1',
        'test_branch1_auc', 'test_branch2_auc',
    ]
    available = [col for col in score_cols if col in training_df.columns]
    if not available:
        return pd.DataFrame()

    df = training_df.copy()
    if 'test_fusion_auc' in df.columns:
        primary = 'test_fusion_auc'
    elif 'val_fusion_auc' in df.columns:
        primary = 'val_fusion_auc'
    else:
        primary = available[0]

    group_cols = ['dataset', 'experiment', 'model_name', 'source_file']
    group_cols = [col for col in group_cols if col in df.columns]
    best = (
        df.sort_values(group_cols + [primary, '_timestamp_sort'], ascending=[True] * len(group_cols) + [False, False])
        .groupby(group_cols, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    keep = [
        'dataset', 'experiment', 'model_name', 'run_id', 'seed', 'epoch',
        'test_branch1_auc', 'test_branch1_f1',
        'test_branch2_auc', 'test_branch2_f1',
        'test_fusion_auc', 'test_fusion_f1',
        'val_fusion_auc', 'val_fusion_f1',
        'train_loss', 'val_loss', 'test_loss',
        'timestamp', 'source_file',
    ]
    keep = [col for col in keep if col in best.columns]
    return cleanup_output(best[keep])


def cleanup_output(df):
    drop_cols = [
        col for col in df.columns
        if col.startswith('_') or col.startswith('Unnamed:')
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    preferred = [
        'dataset', 'experiment', 'model_name', 'run_id', 'seed', 'best_epoch', 'epoch',
        'selection_score_mode', 'training_selection_mode', 'active_modules', 'run_note',
        'branch1_auc', 'branch1_tpr', 'branch1_acc', 'branch1_f1',
        'branch2_auc', 'branch2_tpr', 'branch2_acc', 'branch2_f1',
        'fusion_auc', 'fusion_tpr', 'fusion_acc', 'fusion_f1',
        'best_single_auc', 'best_single_branch',
        'test_branch1_auc', 'test_branch1_tpr', 'test_branch1_acc', 'test_branch1_f1',
        'test_branch2_auc', 'test_branch2_tpr', 'test_branch2_acc', 'test_branch2_f1',
        'test_fusion_auc', 'test_fusion_tpr', 'test_fusion_acc', 'test_fusion_f1',
        'val_fusion_auc', 'val_fusion_f1',
        'train_loss', 'val_loss', 'test_loss',
        'timestamp', 'source_file',
    ]
    ordered = [col for col in preferred if col in df.columns]
    rest = [col for col in df.columns if col not in ordered]
    return df[ordered + rest]


def build_paper_ready_table(best_single_df):
    if best_single_df.empty:
        return pd.DataFrame()
    df = best_single_df.copy()
    df = df[df['dataset'].isin(['ptbxl', 'cpsc', 'tmp_data'])].copy()
    keep = [
        'dataset', 'experiment', 'model_name', 'best_epoch',
        'branch1_auc', 'branch1_tpr', 'branch1_acc', 'branch1_f1',
        'branch2_auc', 'branch2_tpr', 'branch2_acc', 'branch2_f1',
        'fusion_auc', 'fusion_tpr', 'fusion_acc', 'fusion_f1',
        'best_single_auc', 'best_single_branch',
        'active_modules', 'run_note', 'timestamp', 'source_file',
    ]
    keep = [col for col in keep if col in df.columns]
    return df[keep].sort_values(['dataset', 'experiment']).reset_index(drop=True)


def build_file_inventory(results_dir):
    rows = []
    for path in sorted(glob(os.path.join(results_dir, '*.csv'))):
        filename = os.path.basename(path)
        df = read_csv_lenient(path)
        rows.append({
            'file': filename,
            'category': classify_file(filename),
            'dataset_guess': infer_dataset_from_name(filename),
            'experiment_guess': infer_experiment_from_name(filename),
            'rows': 0 if df is None else len(df),
            'columns': 0 if df is None else len(df.columns),
            'size_kb': round(os.path.getsize(path) / 1024, 2),
            'last_modified': pd.to_datetime(os.path.getmtime(path), unit='s').strftime('%Y-%m-%d %H:%M:%S'),
        })
    return pd.DataFrame(rows)


def classify_file(filename):
    if filename.startswith('training_log'):
        return 'training_log'
    if filename.startswith('final_test'):
        return 'final_test'
    if filename.startswith('results'):
        return 'final_result'
    if filename.startswith('summary'):
        return 'old_summary'
    if filename.startswith('thresholds'):
        return 'thresholds'
    return 'other'


def save_outputs(final_df, training_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    final_all = cleanup_output(final_df)
    training_all = cleanup_output(training_df)
    best_fusion, best_single = build_best_final_tables(final_df)
    training_best = build_training_best_table(training_df)
    paper_ready = build_paper_ready_table(best_single)

    outputs = {
        'all_final_results.csv': final_all,
        'best_final_by_fusion_auc.csv': best_fusion,
        'best_final_by_single_auc.csv': best_single,
        'paper_ready_best_results.csv': paper_ready,
        'all_training_logs.csv': training_all,
        'best_epoch_from_training_logs.csv': training_best,
    }
    saved_paths = []
    for filename, df in outputs.items():
        path = os.path.join(out_dir, filename)
        df.to_csv(path, index=False)
        saved_paths.append(path)
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description='Consolidate scattered ECG experiment CSV files.')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, 'consolidated')
    final_df = collect_final_results(args.results_dir)
    training_df = collect_training_logs(args.results_dir)
    inventory_df = build_file_inventory(args.results_dir)

    saved_paths = save_outputs(final_df, training_df, out_dir)
    inventory_path = os.path.join(out_dir, 'file_inventory.csv')
    os.makedirs(out_dir, exist_ok=True)
    inventory_df.to_csv(inventory_path, index=False)
    saved_paths.append(inventory_path)

    print('Consolidated files saved:')
    for path in saved_paths:
        print(' -', path)
    print(f'Final result rows: {len(final_df)}')
    print(f'Training log rows: {len(training_df)}')


if __name__ == '__main__':
    main()
