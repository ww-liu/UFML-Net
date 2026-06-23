#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


KEEP_CONFIG_FIELDS = [
    "dataset",
    "datafolder",
    "cpsc_recipe",
    "cpsc_split_mode",
    "target_mode",
    "loss_name",
    "fusion_mode",
    "reliability_mode",
    "use_fdconv_scale",
    "use_frequency_aux",
    "use_ema",
    "use_swa",
    "mutual_feat_weight",
    "mutual_logit_weight",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epoch",
    "use_early_stopping",
    "early_stop_patience",
    "final_eval_mode",
]

METRIC_FIELDS = [
    "branch1_auc",
    "branch1_tpr",
    "branch1_acc",
    "branch1_f1",
    "branch2_auc",
    "branch2_tpr",
    "branch2_acc",
    "branch2_f1",
    "fusion_auc",
    "fusion_tpr",
    "fusion_acc",
    "fusion_f1",
]

OUTPUT_FIELDS = [
    "dataset",
    "result_group",
    "run_name",
    "seed",
    "experiment",
    "model_name",
    "best_epoch",
    "selection_score_mode",
    "training_selection_mode",
    "active_modules",
    "run_note",
    "cpsc_recipe",
    "cpsc_split_mode",
    "target_mode",
    "loss_name",
    "fusion_mode",
    "reliability_mode",
    "use_fdconv_scale",
    "use_frequency_aux",
    "use_ema",
    "use_swa",
    "mutual_feat_weight",
    "mutual_logit_weight",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epoch",
    "final_eval_mode",
    *METRIC_FIELDS,
    "best_single_auc",
    "best_single_source",
    "num_params",
    "checkpoint_path",
    "timestamp",
    "run_dir",
]


def read_last_csv_row(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def read_json(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_number(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def dataset_from_run_dir(path):
    if path.name == "diagnostics" and path.parent.parent.name.lower() in {"georgia"}:
        return path.parent.parent.name.lower()
    parent = path.parent.name.lower()
    if parent in {"cpsc", "ptbxl", "georgia"}:
        return parent
    return parent or "unknown"


def classify_run(dataset, run_name):
    if dataset == "cpsc":
        if re.fullmatch(r"CPSC_Baseline_MRM_seed\d+_branch2only", run_name):
            return "cpsc_reference_mrm"
        if re.fullmatch(r"CPSC_LEGACY_E[0-4]_.+_seed\d+", run_name):
            return "cpsc_formal_ablation"
        if re.fullmatch(r"CPSC_LEGACY_E[56]_.+_seed\d+", run_name):
            return "cpsc_fusion_sanity"
        return "cpsc_exploration"
    if dataset == "ptbxl":
        if re.fullmatch(r"PTB_seed\d+_exp(?:0|1|11|111|2|3)", run_name):
            return "ptbxl_formal"
        if run_name.startswith("debug_"):
            return "ptbxl_debug"
        return "ptbxl_exploration"
    if dataset == "georgia":
        return "georgia_exploration"
    return "other"


def best_single(row):
    scores = {
        "branch1": to_number(row.get("branch1_auc")),
        "branch2": to_number(row.get("branch2_auc")),
        "fusion": to_number(row.get("fusion_auc")),
    }
    scores = {key: value for key, value in scores.items() if value is not None}
    if not scores:
        return "", ""
    source, value = max(scores.items(), key=lambda item: item[1])
    return value, source


def collect_runs(results_root):
    rows = []
    for metrics_path in sorted(results_root.glob("**/final_metrics.csv")):
        if "consolidated" in metrics_path.parts:
            continue
        run_dir = metrics_path.parent
        config_dir = run_dir
        if run_dir.name == "diagnostics" and run_dir.parent.parent.name.lower() == "georgia":
            run_dir = run_dir.parent
            config_dir = metrics_path.parent
        run_name = run_dir.name
        dataset = dataset_from_run_dir(run_dir)
        metrics = read_last_csv_row(metrics_path)
        config = read_json(config_dir / "config.json")
        if not config:
            config = read_json(run_dir / "config.json")
        best_auc, best_source = best_single(metrics)

        row = {
            "dataset": metrics.get("dataset") or dataset,
            "result_group": classify_run(dataset, run_name),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "best_single_auc": best_auc,
            "best_single_source": best_source,
        }
        for key in OUTPUT_FIELDS:
            if key in row:
                continue
            if key in metrics and metrics[key] != "":
                row[key] = metrics[key]
            elif key in KEEP_CONFIG_FIELDS:
                row[key] = config.get(key, "")
            else:
                row[key] = ""
        rows.append(row)
    return rows


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved: {path} ({len(df)} rows)")


def write_readme(out_dir):
    text = """# Consolidated Results

This directory is generated by `scripts/consolidate_run_results.py`.

Source of truth: `results/<dataset>/<run_name>/final_metrics.csv` plus each run's `config.json`.

Files:
- `all_runs.csv`: every run directory with `final_metrics.csv`.
- `paper_main_candidates.csv`: PTB formal runs plus CPSC MRM reference, CPSC formal ablation, and CPSC fusion sanity runs.
- `cpsc_formal_ablation.csv`: CPSC E0-E4 formal ablation; E4 is the current full model.
- `cpsc_fusion_sanity.csv`: CPSC E5/E6 fusion sanity checks.
- `cpsc_exploration.csv`: earlier CPSC tuning/exploration runs.
- `ptbxl_formal.csv`: PTB-XL seed/experiment formal run directories.
- `ptbxl_exploration.csv`: PTB-XL debug or exploratory runs.
- `georgia_exploration.csv`: Georgia exploratory diagnostics.

No original result files are moved or deleted by this script.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")
    print(f"saved: {out_dir / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate run-directory final metrics.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-dir", default="results/consolidated")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    rows = collect_runs(results_root)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No run final_metrics.csv found under {results_root}")

    for col in METRIC_FIELDS + ["best_single_auc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["seed", "best_epoch", "num_params"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ordered = [col for col in OUTPUT_FIELDS if col in df.columns]
    extra = [col for col in df.columns if col not in ordered]
    df = df[ordered + extra].sort_values(["dataset", "result_group", "run_name"]).reset_index(drop=True)

    write_csv(df, out_dir / "all_runs.csv")
    write_csv(
        df[df["result_group"].isin(["ptbxl_formal", "cpsc_reference_mrm", "cpsc_formal_ablation", "cpsc_fusion_sanity"])],
        out_dir / "paper_main_candidates.csv",
    )
    write_csv(df[df["result_group"] == "cpsc_formal_ablation"], out_dir / "cpsc_formal_ablation.csv")
    write_csv(df[df["result_group"] == "cpsc_fusion_sanity"], out_dir / "cpsc_fusion_sanity.csv")
    write_csv(df[df["result_group"] == "cpsc_exploration"], out_dir / "cpsc_exploration.csv")
    write_csv(df[df["result_group"] == "ptbxl_formal"], out_dir / "ptbxl_formal.csv")
    write_csv(df[df["result_group"].isin(["ptbxl_debug", "ptbxl_exploration"])], out_dir / "ptbxl_exploration.csv")
    write_csv(df[df["result_group"] == "georgia_exploration"], out_dir / "georgia_exploration.csv")
    write_readme(out_dir)


if __name__ == "__main__":
    main()
