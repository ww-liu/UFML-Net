import argparse
import os
from glob import glob

import pandas as pd


def load_final_test_rows(results_dir, dataset_tag):
    pattern = os.path.join(results_dir, f'final_test_{dataset_tag}_*.csv')
    rows = []
    for path in sorted(glob(pattern)):
        try:
            df = pd.read_csv(path)
            if len(df) == 0:
                continue
            latest = df.iloc[-1].copy()
            latest['source_file'] = os.path.basename(path)
            rows.append(latest)
        except Exception as exc:
            print(f'skip invalid final test file: {path} ({exc})')
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_results_rows(results_dir, dataset_tag):
    path = os.path.join(results_dir, f'results_{dataset_tag}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    df = df.copy()
    df['source_file'] = os.path.basename(path)
    return df


def build_summary(df, use_final_test):
    if use_final_test:
        keep_cols = [
            'experiment', 'model_name', 'seed', 'best_epoch',
            'best_val_fusion_auc', 'best_val_fusion_f1',
            'test_branch1_auc', 'test_branch1_f1',
            'test_branch2_auc', 'test_branch2_f1',
            'test_fusion_auc', 'test_fusion_f1',
            'test_loss', 'timestamp', 'source_file'
        ]
        rename_map = {}
    else:
        keep_cols = [
            'experiment', 'model_name', 'seed', 'best_epoch',
            'branch1_auc', 'branch1_f1',
            'branch2_auc', 'branch2_f1',
            'fusion_auc', 'fusion_f1',
            'timestamp', 'source_file'
        ]
        rename_map = {
            'branch1_auc': 'test_branch1_auc',
            'branch1_f1': 'test_branch1_f1',
            'branch2_auc': 'test_branch2_auc',
            'branch2_f1': 'test_branch2_f1',
            'fusion_auc': 'test_fusion_auc',
            'fusion_f1': 'test_fusion_f1',
        }

    available_cols = [col for col in keep_cols if col in df.columns]
    summary = df[available_cols].copy()
    summary = summary.rename(columns=rename_map)
    if 'experiment' in summary.columns:
        summary = summary.sort_values(by='experiment').reset_index(drop=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment results into one table.')
    parser.add_argument('--dataset', required=True, help='Dataset tag, e.g. ptbxl or cpsc')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    args = parser.parse_args()

    results_dir = args.results_dir
    dataset_tag = args.dataset.strip().lower()

    df = load_final_test_rows(results_dir, dataset_tag)
    use_final_test = df is not None
    if df is None:
        df = load_results_rows(results_dir, dataset_tag)

    if df is None:
        print(f'No result files found for dataset: {dataset_tag}')
        return

    summary = build_summary(df, use_final_test)
    out_path = os.path.join(results_dir, f'summary_{dataset_tag}.csv')
    summary.to_csv(out_path, index=False)
    print(f'Summary saved to: {out_path}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
