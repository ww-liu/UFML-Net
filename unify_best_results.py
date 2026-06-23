import argparse
import os
import shutil
from datetime import datetime
from glob import glob

import pandas as pd


RESULT_COLUMNS = [
    'run_id', 'dataset', 'experiment', 'model_name', 'seed', 'best_epoch',
    'selection_score_mode', 'training_selection_mode', 'run_tag',
    'active_modules', 'run_note',
    'branch1_auc', 'branch1_tpr', 'branch1_acc', 'branch1_f1',
    'branch2_auc', 'branch2_tpr', 'branch2_acc', 'branch2_f1',
    'fusion_auc', 'fusion_tpr', 'fusion_acc', 'fusion_f1',
    'timestamp',
]


TRAINING_COLUMNS = [
    'run_id', 'dataset', 'experiment', 'model_name', 'seed', 'epoch',
    'selection_score_mode', 'training_selection_mode', 'run_tag',
    'active_modules', 'run_note',
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
    'timestamp',
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


def canonical_dataset(value, filename):
    text = f'{value} {filename}'.lower()
    if 'ptb' in text:
        return 'ptbxl'
    if 'cpsc' in text or 'tmp_data' in text:
        return 'cpsc'
    return 'unknown'


def infer_experiment(filename):
    lower = filename.lower()
    for token in ['exp1.1.1', 'exp1.1', 'exp0', 'exp1', 'exp2', 'exp3']:
        if token in lower:
            return token
    return ''


def infer_run_tag(filename, dataset, experiment):
    stem = os.path.splitext(filename)[0]
    prefixes = (
        f'training_log_v2_{dataset}', f'training_log_{dataset}',
        'training_log_v2', 'training_log',
        f'results_{dataset}', f'final_test_{dataset}',
        'results', 'final_test',
    )
    for prefix in prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    stem = stem.strip('_')
    if experiment and stem.startswith(experiment):
        stem = stem[len(experiment):].strip('_')
    return stem


def normalize_result_frame(df, path):
    filename = os.path.basename(path)
    df = df.copy()
    if 'dataset' not in df.columns:
        df['dataset'] = ''
    df['dataset'] = [
        canonical_dataset(value, filename)
        for value in df['dataset'].fillna('').astype(str)
    ]
    if 'experiment' not in df.columns:
        df['experiment'] = infer_experiment(filename)
    df['experiment'] = df['experiment'].fillna(infer_experiment(filename)).astype(str)
    if 'run_tag' not in df.columns:
        df['run_tag'] = [
            infer_run_tag(filename, dataset, experiment)
            for dataset, experiment in zip(df['dataset'], df['experiment'])
        ]
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    return df[RESULT_COLUMNS]


def normalize_training_frame(df, path):
    filename = os.path.basename(path)
    df = df.copy()
    if 'dataset' not in df.columns:
        df['dataset'] = ''
    df['dataset'] = [
        canonical_dataset(value, filename)
        for value in df['dataset'].fillna('').astype(str)
    ]
    if 'experiment' not in df.columns:
        df['experiment'] = infer_experiment(filename)
    df['experiment'] = df['experiment'].fillna(infer_experiment(filename)).astype(str)
    if 'model_name' not in df.columns:
        df['model_name'] = 'MyNet6View'
    df['model_name'] = df['model_name'].fillna('MyNet6View').astype(str)
    if 'run_tag' not in df.columns:
        df['run_tag'] = [
            infer_run_tag(filename, dataset, experiment)
            for dataset, experiment in zip(df['dataset'], df['experiment'])
        ]
    for col in TRAINING_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    return df[TRAINING_COLUMNS]


def collect_result_files(results_dir, prefix):
    paths = []
    for path in sorted(glob(os.path.join(results_dir, f'{prefix}*.csv'))):
        filename = os.path.basename(path)
        if filename.startswith('summary_'):
            continue
        paths.append(path)
    return paths


def collect_result_frames(paths):
    frames = []
    for path in paths:
        df = read_csv_lenient(path)
        if df is None or df.empty:
            continue
        frame = normalize_result_frame(df, path)
        frame = frame[frame['dataset'].isin(['ptbxl', 'cpsc'])]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates()
    merged['_timestamp_sort'] = pd.to_datetime(merged['timestamp'], errors='coerce')
    merged = merged.sort_values(
        ['dataset', 'experiment', '_timestamp_sort'],
        ascending=[True, True, True],
        na_position='first',
    )
    return merged.drop(columns=['_timestamp_sort'])


def collect_training_frames(paths):
    frames = []
    for path in paths:
        df = read_csv_lenient(path)
        if df is None or df.empty:
            continue
        frame = normalize_training_frame(df, path)
        frame = frame[frame['dataset'].isin(['ptbxl', 'cpsc'])]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=TRAINING_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates()
    merged['_epoch_sort'] = pd.to_numeric(merged['epoch'], errors='coerce')
    merged['_timestamp_sort'] = pd.to_datetime(merged['timestamp'], errors='coerce')
    merged = merged.sort_values(
        ['dataset', 'model_name', 'experiment', 'run_tag', '_epoch_sort', '_timestamp_sort'],
        ascending=[True, True, True, True, True, True],
        na_position='first',
    )
    return merged.drop(columns=['_epoch_sort', '_timestamp_sort'])


def safe_filename_token(value):
    token = str(value or 'unknown').strip()
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in token) or 'unknown'


def backup_and_archive(paths, results_dir, target_names):
    archive_dir = os.path.join(
        results_dir,
        'archived_best_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    os.makedirs(archive_dir, exist_ok=True)
    target_paths = {os.path.abspath(os.path.join(results_dir, name)) for name in target_names}
    for path in paths:
        abs_path = os.path.abspath(path)
        if not os.path.exists(path):
            continue
        if abs_path in target_paths:
            shutil.copy2(path, os.path.join(archive_dir, 'backup_' + os.path.basename(path)))
            continue
        shutil.move(path, os.path.join(archive_dir, os.path.basename(path)))
    return archive_dir


def write_by_dataset(df, results_dir, prefix):
    saved = []
    for dataset in ['ptbxl', 'cpsc']:
        dataset_df = df[df['dataset'] == dataset].copy()
        if dataset_df.empty:
            continue
        path = os.path.join(results_dir, f'{prefix}_{dataset}.csv')
        dataset_df.to_csv(path, index=False)
        saved.append(path)
    return saved


def write_training_by_dataset_model(df, results_dir):
    saved = []
    if df.empty:
        return saved
    for (dataset, model_name), group_df in df.groupby(['dataset', 'model_name'], sort=True):
        if dataset not in ['ptbxl', 'cpsc']:
            continue
        path = os.path.join(
            results_dir,
            f'training_log_{dataset}_{safe_filename_token(model_name)}.csv'
        )
        group_df.to_csv(path, index=False)
        saved.append(path)
    return saved


def get_training_target_names(df):
    if df.empty:
        return []
    names = []
    for dataset, model_name in df[['dataset', 'model_name']].drop_duplicates().itertuples(index=False):
        if dataset in ['ptbxl', 'cpsc']:
            names.append(f'training_log_{dataset}_{safe_filename_token(model_name)}.csv')
    return names


def main():
    parser = argparse.ArgumentParser(description='Unify best-result CSV files by dataset.')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    result_paths = collect_result_files(args.results_dir, 'results')
    final_paths = collect_result_files(args.results_dir, 'final_test')
    training_paths = collect_result_files(args.results_dir, 'training_log')
    result_df = collect_result_frames(result_paths)
    final_df = collect_result_frames(final_paths)
    training_df = collect_training_frames(training_paths)

    target_names = [
        'results_ptbxl.csv', 'results_cpsc.csv',
        'final_test_ptbxl.csv', 'final_test_cpsc.csv',
    ] + get_training_target_names(training_df)
    archive_dir = None
    if not args.no_archive:
        archive_dir = backup_and_archive(result_paths + final_paths + training_paths, args.results_dir, target_names)

    saved = []
    saved.extend(write_by_dataset(result_df, args.results_dir, 'results'))
    saved.extend(write_by_dataset(final_df, args.results_dir, 'final_test'))
    saved.extend(write_training_by_dataset_model(training_df, args.results_dir))

    print('Unified files saved:')
    for path in saved:
        print(' -', path)
    print(f'Result rows: {len(result_df)}')
    print(f'Final-test rows: {len(final_df)}')
    print(f'Training-log rows: {len(training_df)}')
    if archive_dir:
        print(f'Archived scattered result files to: {archive_dir}')


if __name__ == '__main__':
    main()
