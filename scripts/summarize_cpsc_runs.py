#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path):
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_run(results_root, run_name):
    run_dir = Path(results_root) / "cpsc" / run_name
    row = {"run_name": run_name, "run_dir": str(run_dir), "status": "missing"}
    metrics_path = run_dir / "final_metrics.csv"
    config_path = run_dir / "config.json"
    if not metrics_path.is_file():
        return row

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        row["status"] = "empty_metrics"
        return row

    cfg = read_json(config_path)
    merged = {**cfg, **metrics.iloc[-1].to_dict()}
    merged["run_name"] = run_name
    merged["run_dir"] = str(run_dir)
    merged["status"] = "ok"
    return merged


def get(row, key, default=np.nan):
    return row.get(key, default)


def fair_rows(rows):
    output = []
    for row in rows:
        output.append({
            "dataset": get(row, "dataset", "cpsc"),
            "run_name": get(row, "run_name", ""),
            "model": get(row, "model_name", ""),
            "seed": get(row, "seed", ""),
            "target_mode": get(row, "target_mode", ""),
            "loss_name": get(row, "loss_name", ""),
            "branch1_auc": get(row, "branch1_auc"),
            "branch2_auc": get(row, "branch2_auc"),
            "fusion_auc": get(row, "fusion_auc"),
            "branch1_f1": get(row, "branch1_f1"),
            "branch2_f1": get(row, "branch2_f1"),
            "fusion_f1": get(row, "fusion_f1"),
            "fusion_acc": get(row, "fusion_acc"),
            "fusion_TPR": get(row, "fusion_tpr"),
            "best_epoch": get(row, "best_epoch"),
            "params": get(row, "num_params"),
            "run_dir": get(row, "run_dir", ""),
            "status": get(row, "status", "ok"),
        })
    return output


def ablation_conclusion(run_name, delta_auc, delta_f1):
    if not np.isfinite(delta_auc) or not np.isfinite(delta_f1):
        return "missing"
    if "FULL" in run_name:
        return "reference full recipe"
    if "freqaux_on" in run_name:
        return "keep only if it improves CPSC; otherwise leave FrequencyAux off"
    if delta_auc < -0.002 or delta_f1 < -0.005:
        return "worse than full; not recommended"
    if delta_auc > 0.002 or delta_f1 > 0.005:
        return "improves over full; candidate change"
    return "similar to full"


def ablation_rows(rows, full_run_name):
    full = next((row for row in rows if row.get("run_name") == full_run_name and row.get("status") == "ok"), None)
    full_auc = float(get(full, "fusion_auc")) if full else np.nan
    full_f1 = float(get(full, "fusion_f1")) if full else np.nan

    output = []
    for row in rows:
        auc = float(get(row, "fusion_auc")) if row.get("status") == "ok" else np.nan
        f1 = float(get(row, "fusion_f1")) if row.get("status") == "ok" else np.nan
        delta_auc = auc - full_auc if np.isfinite(auc) and np.isfinite(full_auc) else np.nan
        delta_f1 = f1 - full_f1 if np.isfinite(f1) and np.isfinite(full_f1) else np.nan
        run_name = get(row, "run_name", "")
        output.append({
            "method": run_name,
            "auc": auc,
            "f1": f1,
            "acc": get(row, "fusion_acc"),
            "TPR": get(row, "fusion_tpr"),
            "delta_auc_vs_full": delta_auc,
            "delta_f1_vs_full": delta_f1,
            "best_epoch": get(row, "best_epoch"),
            "params": get(row, "num_params"),
            "conclusion": ablation_conclusion(run_name, delta_auc, delta_f1),
            "run_dir": get(row, "run_dir", ""),
            "status": get(row, "status", "ok"),
        })
    return output


def tuning_note(row):
    if row.get("status") != "ok":
        return "missing"
    loss_name = str(get(row, "loss_name", "")).lower()
    if loss_name == "ce" and bool(get(row, "use_class_weighted_ce", False)):
        return "weighted CE single-label recipe"
    if loss_name == "focal":
        return "check focal implementation before using for CPSC"
    return "candidate recipe"


def tuning_rows(rows):
    output = []
    for row in rows:
        output.append({
            "run_name": get(row, "run_name", ""),
            "lr": get(row, "lr"),
            "loss_name": get(row, "loss_name", ""),
            "label_smoothing": get(row, "label_smoothing"),
            "use_class_weighted_ce": get(row, "use_class_weighted_ce"),
            "fusion_auc": get(row, "fusion_auc"),
            "fusion_f1": get(row, "fusion_f1"),
            "fusion_acc": get(row, "fusion_acc"),
            "fusion_TPR": get(row, "fusion_tpr"),
            "best_epoch": get(row, "best_epoch"),
            "note": tuning_note(row),
            "run_dir": get(row, "run_dir", ""),
            "status": get(row, "status", "ok"),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description="Summarize CPSC run directories.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--mode", choices=["fair", "ablation", "tuning"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--full-run-name", default="CPSC_UFML_FULL_seed1")
    args = parser.parse_args()

    rows = [read_run(args.results_root, run_name) for run_name in args.run_names]
    if args.mode == "fair":
        out_rows = fair_rows(rows)
    elif args.mode == "ablation":
        out_rows = ablation_rows(rows, args.full_run_name)
    else:
        out_rows = tuning_rows(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns
    df[float_cols] = df[float_cols].round(6)
    df.to_csv(output_path, index=False)
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
