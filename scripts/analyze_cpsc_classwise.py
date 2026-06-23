#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_CLASS_NAMES = ["N", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def load_npz_array(payload, names):
    for name in names:
        if name in payload:
            return payload[name]
    return None


def load_predictions(path):
    payload = np.load(path)
    y_true = load_npz_array(payload, ["y_true", "test_y_true", "labels", "targets"])
    if y_true is None:
        raise KeyError(f"No y_true array found in {path}")

    arrays = {"y_true": np.asarray(y_true)}
    branch_specs = {
        "branch1": (["probs_branch1", "test_probs_branch1"], ["logits_branch1", "test_logits_branch1"]),
        "branch2": (["probs_branch2", "test_probs_branch2"], ["logits_branch2", "test_logits_branch2"]),
        "fusion": (["probs_fusion", "test_probs_fusion", "fusion_probs"], ["logits_fusion", "test_logits_fusion"]),
    }
    for branch, (prob_keys, logit_keys) in branch_specs.items():
        prob = load_npz_array(payload, prob_keys)
        if prob is None:
            logits = load_npz_array(payload, logit_keys)
            if logits is not None:
                prob = softmax(np.asarray(logits))
        if prob is not None:
            arrays[branch] = np.asarray(prob)
    if "fusion" not in arrays and "branch1" in arrays and "branch2" in arrays:
        arrays["fusion"] = 0.5 * (arrays["branch1"] + arrays["branch2"])
    return arrays


def normalize_y_true(y_true, num_classes):
    y_true = np.asarray(y_true)
    if y_true.ndim == 1:
        out = np.zeros((len(y_true), num_classes), dtype=np.int64)
        out[np.arange(len(y_true)), y_true.astype(int)] = 1
        return out
    return y_true.astype(int)


def binary_confusion(yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def safe_auc(yt, score):
    if len(np.unique(yt)) < 2:
        return np.nan
    return float(roc_auc_score(yt, score))


def per_class_metrics(y_true, probs, class_names):
    pred_labels = np.argmax(probs, axis=1)
    rows = []
    for idx, class_name in enumerate(class_names):
        yt = y_true[:, idx].astype(int)
        yp = (pred_labels == idx).astype(int)
        tn, fp, fn, tp = binary_confusion(yt, yp)
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        note = []
        support = int(yt.sum())
        if support < 5:
            note.append("very low support")
        elif support < 20:
            note.append("low support")
        rows.append({
            "class_name": class_name,
            "support": support,
            "AUC": safe_auc(yt, probs[:, idx]),
            "F1": f1_score(yt, yp, zero_division=0),
            "Precision": precision_score(yt, yp, zero_division=0),
            "Recall": recall_score(yt, yp, zero_division=0),
            "Specificity": specificity,
            "Accuracy": accuracy_score(yt, yp),
            "note": "; ".join(note),
        })
    return pd.DataFrame(rows)


def merge_branch_tables(y_true, arrays, class_names):
    tables = {}
    for branch in ("branch1", "branch2", "fusion"):
        if branch in arrays:
            tables[branch] = per_class_metrics(y_true, arrays[branch], class_names)

    if "fusion" not in tables:
        raise KeyError("No fusion probabilities found. Expected probs_fusion or branch probabilities.")

    base = tables["fusion"][["class_name", "support", "AUC", "F1", "Precision", "Recall", "Specificity", "Accuracy", "note"]].copy()
    for branch in ("branch1", "branch2", "fusion"):
        if branch in tables:
            base[f"{branch}_auc"] = tables[branch]["AUC"].values
            base[f"{branch}_f1"] = tables[branch]["F1"].values
        else:
            base[f"{branch}_auc"] = np.nan
            base[f"{branch}_f1"] = np.nan
    return base


def plot_confusion_matrix(y_true, probs, class_names, output_path):
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CPSC Fusion Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_auc_f1(table, output_path):
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, table["fusion_auc"], width=0.36, label="Fusion AUC")
    ax.bar(x + 0.18, table["fusion_f1"], width=0.36, label="Fusion F1")
    ax.set_xticks(x)
    ax.set_xticklabels(table["class_name"], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("CPSC Per-Class Fusion AUC and F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def find_predictions(results_root, run_name):
    root = Path(results_root)
    candidates = []
    if run_name:
        candidates.append(root / "cpsc" / run_name / "predictions.npz")
    for name in ("CPSC_UFML_FINAL_seed1", "CPSC_UFML_FULL_seed1"):
        candidates.append(root / "cpsc" / name / "predictions.npz")
    candidates.extend(sorted((root / "cpsc").glob("*/predictions.npz"), key=lambda p: p.stat().st_mtime, reverse=True))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No CPSC predictions.npz found. Pass --predictions or --run-name.")


def write_analysis(table, output_path):
    hardest_f1 = table.sort_values("fusion_f1", ascending=True).head(3)
    hardest_auc = table.sort_values("fusion_auc", ascending=True).head(3)
    fusion_wins = int(((table["fusion_auc"] >= table["branch1_auc"]) & (table["fusion_auc"] >= table["branch2_auc"])).sum())
    branch1_wins = int(((table["branch1_auc"] > table["branch2_auc"]) & (table["branch1_auc"] > table["fusion_auc"])).sum())
    branch2_wins = int(((table["branch2_auc"] > table["branch1_auc"]) & (table["branch2_auc"] > table["fusion_auc"])).sum())
    lines = [
        "# CPSC Classwise Analysis",
        "",
        "## Hard Classes",
        "",
        "- Lowest fusion F1: " + ", ".join(f"{r.class_name} (F1={r.fusion_f1:.4f}, support={int(r.support)})" for r in hardest_f1.itertuples()),
        "- Lowest fusion AUC: " + ", ".join(f"{r.class_name} (AUC={r.fusion_auc:.4f}, support={int(r.support)})" for r in hardest_auc.itertuples()),
        "",
        "## Branch Comparison",
        "",
        f"- Best AUC counts: branch1={branch1_wins}, branch2={branch2_wins}, fusion={fusion_wins}.",
        f"- Mean AUC: branch1={table['branch1_auc'].mean():.4f}, branch2={table['branch2_auc'].mean():.4f}, fusion={table['fusion_auc'].mean():.4f}.",
        f"- Mean F1: branch1={table['branch1_f1'].mean():.4f}, branch2={table['branch2_f1'].mean():.4f}, fusion={table['fusion_f1'].mean():.4f}.",
        "",
        "## Interpretation",
        "",
        "Low-support classes should be interpreted together with F1 and recall. If AUC is acceptable but F1 is low, the model may rank the class reasonably while the argmax decision still misses minority classes.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze CPSC 9-class predictions.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--class-names", default=",".join(DEFAULT_CLASS_NAMES))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.predictions) if args.predictions else find_predictions(args.results_root, args.run_name)
    arrays = load_predictions(pred_path)
    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    y_true = normalize_y_true(arrays["y_true"], len(class_names))
    table = merge_branch_tables(y_true, arrays, class_names)
    table = table[[
        "class_name", "support", "AUC", "F1", "Precision", "Recall", "Specificity", "Accuracy",
        "branch1_auc", "branch2_auc", "fusion_auc", "branch1_f1", "branch2_f1", "fusion_f1", "note",
    ]]
    float_cols = table.select_dtypes(include=["float", "float64", "float32"]).columns
    table[float_cols] = table[float_cols].round(6)

    metrics_path = output_dir / "cpsc_classwise_metrics.csv"
    table.to_csv(metrics_path, index=False)
    plot_confusion_matrix(y_true, arrays["fusion"], class_names, output_dir / "cpsc_confusion_matrix.png")
    plot_auc_f1(table, output_dir / "cpsc_per_class_f1_auc.png")
    write_analysis(table, output_dir / "cpsc_classwise_analysis.md")

    metadata = {
        "predictions": str(pred_path),
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(y_true.shape[1]),
        "class_names": class_names,
    }
    (output_dir / "cpsc_classwise_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Classwise metrics saved to: {metrics_path}")
    print(f"Plots saved to: {output_dir / 'cpsc_confusion_matrix.png'} and {output_dir / 'cpsc_per_class_f1_auc.png'}")


if __name__ == "__main__":
    main()
