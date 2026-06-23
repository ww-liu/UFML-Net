# -*- coding: utf-8 -*-

import argparse
import ast
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


EXPERIMENTS = {
    "exp0": "all",
    "exp1": "diagnostic",
    "exp1.1": "subdiagnostic",
    "exp1.1.1": "superdiagnostic",
    "exp2": "form",
    "exp3": "rhythm",
}

TASK_FILE_PREFIX = {
    "exp0": "exp0_all",
    "exp1": "exp1_diagnostic",
    "exp1.1": "exp11_subdiagnostic",
    "exp1.1.1": "exp111_superdiagnostic",
    "exp2": "exp2_form",
    "exp3": "exp3_rhythm",
}

BRANCHES = {
    "branch1": {
        "prob": ["test_probs_branch1", "probs_branch1", "branch1_probs"],
        "logit": ["test_logits_branch1", "logits_branch1", "out1"],
    },
    "branch2": {
        "prob": ["test_probs_branch2", "probs_branch2", "branch2_probs"],
        "logit": ["test_logits_branch2", "logits_branch2", "out2"],
    },
    "fusion": {
        "prob": ["test_probs_fusion", "probs_fusion", "fusion_probs", "fusion", "y_score", "preds", "test_predictions"],
        "logit": ["test_logits_fusion", "logits_fusion", "fusion_logits"],
    },
}

Y_TRUE_KEYS = ["test_y_true", "y_true", "labels", "targets", "test_labels"]
VAL_Y_TRUE_KEYS = ["val_y_true", "validation_y_true"]
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)


def safe_float(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def round_df(df, decimals=4):
    out = df.copy()
    float_cols = out.select_dtypes(include=["float", "float32", "float64"]).columns
    out[float_cols] = out[float_cols].round(decimals)
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_json(path):
    if path and Path(path).is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def sanitize_experiment_token(experiment):
    return TASK_FILE_PREFIX.get(experiment, experiment.replace(".", ""))


def get_task_name(experiment):
    return EXPERIMENTS.get(experiment, "unknown")


def infer_experiment_from_path(path):
    text = str(path).lower()
    for exp in sorted(EXPERIMENTS, key=len, reverse=True):
        token = exp.lower()
        if token in text:
            return exp
    return ""


def find_datafolder(explicit):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        Path("/root/autodl-tmp/data/ptbxl"),
        Path("D:/pro/data/ptbxl"),
        Path("../data/ptbxl"),
    ])
    for candidate in candidates:
        if (candidate / "raw" / "ptbxl_database.csv").is_file() and (candidate / "raw" / "scp_statements.csv").is_file():
            return candidate
    return None


def aggregate_labels(df, scp_df, task):
    if task in ("diagnostic", "subdiagnostic", "superdiagnostic"):
        diag_df = scp_df[scp_df["diagnostic"] == 1.0]

        def aggregate(codes):
            labels = []
            for key in codes.keys():
                if key not in diag_df.index:
                    continue
                if task == "diagnostic":
                    value = key
                elif task == "subdiagnostic":
                    value = diag_df.loc[key, "diagnostic_subclass"]
                else:
                    value = diag_df.loc[key, "diagnostic_class"]
                if str(value) != "nan":
                    labels.append(value)
            return list(set(labels))

    elif task == "form":
        form_df = scp_df[scp_df["form"] == 1.0]

        def aggregate(codes):
            return list(set(key for key in codes.keys() if key in form_df.index))

    elif task == "rhythm":
        rhythm_df = scp_df[scp_df["rhythm"] == 1.0]

        def aggregate(codes):
            return list(set(key for key in codes.keys() if key in rhythm_df.index))

    else:
        def aggregate(codes):
            return list(set(codes.keys()))

    return df["scp_codes"].apply(aggregate)


def recover_class_names(datafolder, experiment, expected_num_classes=None):
    task = get_task_name(experiment)
    if datafolder is None:
        return [f"class_{i}" for i in range(expected_num_classes or 0)], "no PTB-XL datafolder found"

    try:
        db = pd.read_csv(datafolder / "raw" / "ptbxl_database.csv", index_col="ecg_id")
        db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
        scp = pd.read_csv(datafolder / "raw" / "scp_statements.csv", index_col=0)
        labels = aggregate_labels(db, scp, task)
        if task != "all":
            counts = pd.Series(np.concatenate(labels.values)).value_counts()
            keep = set(counts[counts > 0].index.values)
            labels = labels.apply(lambda x: list(set(x).intersection(keep)))
            labels = labels[labels.apply(len) > 0]
        else:
            counts = pd.Series(np.concatenate(labels.values)).value_counts()
            keep = set(counts[counts > 0].index.values)
            labels = labels.apply(lambda x: list(set(x).intersection(keep)))
            labels = labels[labels.apply(len) > 0]
        class_names = sorted(set(np.concatenate(labels.values).tolist()))
        note = f"recovered from {datafolder}"
        if expected_num_classes is not None and len(class_names) != int(expected_num_classes):
            note += f"; expected {expected_num_classes}, recovered {len(class_names)}"
            if len(class_names) < int(expected_num_classes):
                class_names.extend([f"class_{i}" for i in range(len(class_names), int(expected_num_classes))])
            else:
                class_names = class_names[: int(expected_num_classes)]
        return class_names, note
    except Exception as exc:
        return [f"class_{i}" for i in range(expected_num_classes or 0)], f"failed to recover class names: {exc}"


def load_class_names_from_run(run_dir, num_classes):
    for name in ("label_map.json", "classes.json", "labels.json"):
        path = run_dir / name
        if path.is_file():
            payload = load_json(path)
            if isinstance(payload, list):
                return [str(x) for x in payload], str(path)
            if isinstance(payload, dict):
                for key in ("classes", "labels", "class_names"):
                    if key in payload and isinstance(payload[key], list):
                        items = payload[key]
                        if items and isinstance(items[0], dict):
                            return [str(item.get("code") or item.get("name") or item.get("label")) for item in items], str(path)
                        return [str(x) for x in items], str(path)
    return None, ""


def array_role(name):
    low = name.lower()
    if low in Y_TRUE_KEYS or low.endswith("_y_true") or "label" in low or "target" in low:
        return "ground_truth"
    if "branch1" in low and ("prob" in low or "pred" in low):
        return "branch1_prob"
    if "branch2" in low and ("prob" in low or "pred" in low):
        return "branch2_prob"
    if "fusion" in low and ("prob" in low or "pred" in low):
        return "fusion_prob"
    if "logit" in low or low in ("out1", "out2"):
        return "logits"
    return "unknown"


def collect_npz_field_rows(experiment, npz_path):
    rows = []
    if not npz_path or not npz_path.is_file():
        rows.append({
            "experiment": experiment,
            "key_name": "",
            "shape": "",
            "dtype": "",
            "inferred_role": "missing_predictions_file",
            "note": "predictions.npz not found",
        })
        return rows
    data = np.load(npz_path, allow_pickle=True)
    for key in data.files:
        arr = data[key]
        rows.append({
            "experiment": experiment,
            "key_name": key,
            "shape": str(tuple(arr.shape)),
            "dtype": str(arr.dtype),
            "inferred_role": array_role(key),
            "note": "",
        })
    return rows


def find_flat_training_logs(results_dir, experiment):
    matches = []
    for path in results_dir.glob("training_log_ptbxl_*.csv"):
        try:
            df = pd.read_csv(path, nrows=1)
            if "experiment" in df.columns and str(df.loc[0, "experiment"]) == experiment:
                matches.append(path)
        except Exception:
            if f"_{experiment}_" in path.name or path.name.endswith(f"_{experiment}.csv"):
                matches.append(path)
    return sorted(matches)


def is_probability(arr):
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return float(np.nanmin(finite)) >= 0.0 and float(np.nanmax(finite)) <= 1.0


def find_first(data, keys):
    for key in keys:
        if key in data.files:
            return key, np.asarray(data[key])
    return None, None


def infer_arrays(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    y_key, y_true = find_first(data, Y_TRUE_KEYS)
    if y_true is None:
        return None, {"missing": ["ground_truth"]}
    y_true = np.asarray(y_true).astype(int)
    arrays = {"y_true": y_true}
    sources = {"y_true": y_key}
    missing = []

    for branch, names in BRANCHES.items():
        key, arr = find_first(data, names["prob"])
        used_logits = False
        if arr is None:
            key, arr = find_first(data, names["logit"])
            used_logits = arr is not None
        if arr is None:
            missing.append(branch)
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if used_logits or not is_probability(arr):
            arr = sigmoid(arr)
        arrays[branch] = arr
        sources[branch] = key

    if y_true.ndim != 2:
        missing.append("ground_truth_shape_[N,C]")
    for branch in BRANCHES:
        if branch in arrays and arrays[branch].shape != y_true.shape:
            missing.append(f"{branch}_shape_mismatch:{arrays[branch].shape}!={y_true.shape}")
    return arrays, {"missing": missing, "sources": sources}


def confusion_values(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def compute_classwise_metrics(y_true, y_prob, class_names, threshold=0.5):
    rows = []
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if np.isscalar(threshold):
        thresholds = np.full(y_true.shape[1], float(threshold), dtype=np.float64)
    else:
        thresholds = np.asarray(threshold, dtype=np.float64)

    for i, name in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        thr = thresholds[i]
        yhat = (yp >= thr).astype(int)
        support_pos = int(np.sum(yt == 1))
        support_neg = int(np.sum(yt == 0))
        note = ""
        if support_pos == 0 or support_neg == 0:
            auc = np.nan
            note = "AUC undefined because test labels are all positive or all negative"
        else:
            auc = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp) if support_pos > 0 else np.nan
        tp, tn, fp, fn = confusion_values(yt, yhat)
        precision = precision_score(yt, yhat, zero_division=0)
        recall = recall_score(yt, yhat, zero_division=0)
        f1 = f1_score(yt, yhat, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        accuracy = accuracy_score(yt, yhat)
        balanced_acc = balanced_accuracy_score(yt, yhat) if support_pos > 0 and support_neg > 0 else np.nan
        rows.append({
            "class_index": i,
            "class_name": name,
            "support_pos": support_pos,
            "support_neg": support_neg,
            "positive_rate": support_pos / len(yt) if len(yt) else np.nan,
            "AUC": auc,
            "AUPRC": auprc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Sensitivity": recall,
            "TPR": recall,
            "Specificity": specificity,
            "TNR": specificity,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "threshold": thr,
            "predicted_positive_count": int(np.sum(yhat == 1)),
            "predicted_positive_rate": float(np.mean(yhat == 1)),
            "note": note,
        })
    return pd.DataFrame(rows)


def summary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int) if np.isscalar(threshold) else (y_prob >= np.asarray(threshold)).astype(int)
    per_auc = []
    per_auprc = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 2:
            per_auc.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
        per_auprc.append(average_precision_score(y_true[:, i], y_prob[:, i]) if np.sum(y_true[:, i]) > 0 else np.nan)
    return {
        "macro_auc": float(np.nanmean(per_auc)) if per_auc else np.nan,
        "macro_auprc": float(np.nanmean(per_auprc)) if per_auprc else np.nan,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def optimize_global_threshold(y_true, y_prob):
    best_thr = 0.5
    best_score = -1.0
    for thr in THRESHOLDS:
        score = f1_score(y_true, y_prob >= thr, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def optimize_per_class_thresholds(y_true, y_prob):
    thresholds = []
    for i in range(y_true.shape[1]):
        best_thr = 0.5
        best_score = -1.0
        for thr in THRESHOLDS:
            score = f1_score(y_true[:, i], y_prob[:, i] >= thr, zero_division=0)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        thresholds.append(best_thr)
    return np.asarray(thresholds, dtype=np.float64)


def build_full_table(branch_tables):
    base = branch_tables["fusion"][["class_name", "support_pos", "positive_rate"]].copy()
    for branch in ("branch1", "branch2", "fusion"):
        table = branch_tables[branch]
        base[f"{branch}_auc"] = table["AUC"].values
        base[f"{branch}_f1"] = table["F1"].values
    fusion = branch_tables["fusion"]
    base["fusion_precision"] = fusion["Precision"].values
    base["fusion_recall"] = fusion["Recall"].values
    base["fusion_specificity"] = fusion["Specificity"].values
    base["fusion_auprc"] = fusion["AUPRC"].values

    auc_cols = ["branch1_auc", "branch2_auc", "fusion_auc"]
    f1_cols = ["branch1_f1", "branch2_f1", "fusion_f1"]
    base["best_branch_by_auc"] = base[auc_cols].idxmax(axis=1).str.replace("_auc", "", regex=False)
    base["best_branch_by_f1"] = base[f1_cols].idxmax(axis=1).str.replace("_f1", "", regex=False)
    base["fusion_gain_vs_branch1_auc"] = base["fusion_auc"] - base["branch1_auc"]
    base["fusion_gain_vs_branch2_auc"] = base["fusion_auc"] - base["branch2_auc"]
    base["fusion_gain_vs_branch1_f1"] = base["fusion_f1"] - base["branch1_f1"]
    base["fusion_gain_vs_branch2_f1"] = base["fusion_f1"] - base["branch2_f1"]
    base["note"] = ""
    return base


def write_markdown(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def top_names(df, column, n=5, ascending=False):
    sub = df.sort_values(column, ascending=ascending).head(n)
    return ", ".join(f"{r.class_name} ({getattr(r, column):.4f})" for r in sub.itertuples())


def write_superdiagnostic_analysis(path, table):
    best_auc = table.sort_values("fusion_auc", ascending=False).iloc[0]
    best_f1 = table.sort_values("fusion_f1", ascending=False).iloc[0]
    hard = table.sort_values(["fusion_f1", "fusion_auc"], ascending=[True, True]).iloc[0]
    fusion_auc_gain = table[(table["fusion_gain_vs_branch1_auc"] > 0) & (table["fusion_gain_vs_branch2_auc"] > 0)]
    branch2 = table[(table["branch2_auc"] > table["branch1_auc"]) & (table["branch2_auc"] >= table["fusion_auc"])]
    branch1 = table[(table["branch1_auc"] > table["branch2_auc"]) & (table["branch1_auc"] >= table["fusion_auc"])]
    text = f"""
# exp1.1.1 Superdiagnostic Classwise Analysis

- Highest fusion AUC: {best_auc.class_name} ({best_auc.fusion_auc:.4f}).
- Highest fusion F1: {best_f1.class_name} ({best_f1.fusion_f1:.4f}).
- Hardest class by fusion F1/AUC: {hard.class_name} (F1={hard.fusion_f1:.4f}, AUC={hard.fusion_auc:.4f}, support={int(hard.support_pos)}).
- Fusion clearly better than both branches by AUC: {', '.join(fusion_auc_gain.class_name.tolist()) or 'None'}.
- Branch2 stronger than branch1 and fusion by AUC: {', '.join(branch2.class_name.tolist()) or 'None'}.
- Branch1 stronger than branch2 and fusion by AUC: {', '.join(branch1.class_name.tolist()) or 'None'}.
- Low-F1 classes should be interpreted together with support and positive rate. Rare labels usually make thresholded F1 less stable even when AUC remains high.
- These results support the UFML-Net two-branch design when fusion improves or stabilizes per-class metrics over either branch alone.
"""
    write_markdown(path, text)


def write_subdiagnostic_analysis(path, table):
    scarce = table.sort_values("support_pos", ascending=True).head(10)
    common = table.sort_values("support_pos", ascending=False).head(10)
    high_auc_low_f1 = table[(table["fusion_auc"] >= table["fusion_auc"].quantile(0.75)) & (table["fusion_f1"] <= table["fusion_f1"].quantile(0.25))]
    low_precision = table.sort_values("fusion_precision", ascending=True).head(10)
    fusion_better = table[(table["fusion_gain_vs_branch1_auc"] > 0) & (table["fusion_gain_vs_branch2_auc"] > 0)]
    fusion_down = table[(table["fusion_gain_vs_branch1_auc"] < 0) | (table["fusion_gain_vs_branch2_auc"] < 0)]
    text = f"""
# exp1.1 Subdiagnostic Top/Bottom Analysis

- fusion_auc top 5: {top_names(table, 'fusion_auc', 5, False)}
- fusion_auc bottom 5: {top_names(table, 'fusion_auc', 5, True)}
- fusion_f1 top 5: {top_names(table, 'fusion_f1', 5, False)}
- fusion_f1 bottom 5: {top_names(table, 'fusion_f1', 5, True)}
- 10 least supported classes: {', '.join(f'{r.class_name} ({int(r.support_pos)})' for r in scarce.itertuples())}
- 10 most supported classes: {', '.join(f'{r.class_name} ({int(r.support_pos)})' for r in common.itertuples())}
- High AUC but low F1 classes: {', '.join(high_auc_low_f1.class_name.tolist()) or 'None'}.
- Low precision classes: {', '.join(low_precision.class_name.tolist())}.
- Fusion clearly better than both branches by AUC: {', '.join(fusion_better.class_name.tolist()) or 'None'}.
- Fusion did not improve over both branches by AUC: {', '.join(fusion_down.class_name.tolist()) or 'None'}.
- Paper-ready conclusion: fine-grained subdiagnostic recognition shows larger class-level variance than coarse superdiagnostic recognition. Low F1 is often tied to sparse positive support and fixed thresholding, so validation-based per-class thresholding is useful as a supplementary analysis.
"""
    write_markdown(path, text)


def write_task_analysis(path, table, task):
    branch2_wins = int((table["best_branch_by_auc"] == "branch2").sum())
    branch1_wins = int((table["best_branch_by_auc"] == "branch1").sum())
    fusion_wins = int((table["best_branch_by_auc"] == "fusion").sum())
    hardest = table.sort_values(["fusion_f1", "fusion_auc"], ascending=[True, True]).head(5)
    text = f"""
# {task.title()} Classwise Analysis

- Best by AUC counts: branch1={branch1_wins}, branch2={branch2_wins}, fusion={fusion_wins}.
- Hardest classes by fusion F1: {', '.join(f'{r.class_name} (F1={r.fusion_f1:.4f}, AUC={r.fusion_auc:.4f})' for r in hardest.itertuples())}.
- Fusion average AUC: {table['fusion_auc'].mean():.4f}; branch1 average AUC: {table['branch1_auc'].mean():.4f}; branch2 average AUC: {table['branch2_auc'].mean():.4f}.
"""
    if task == "rhythm":
        text += "\nRhythm labels are expected to benefit from global temporal context; branch2/fusion dominance would support that interpretation.\n"
    if task == "form":
        text += "\nForm labels are morphology-sensitive; branch1/fusion dominance would support local waveform modeling benefits.\n"
    write_markdown(path, text)


def plot_auc_f1_bar(path, table, title):
    fig, ax = plt.subplots(figsize=(max(7, len(table) * 0.8), 4.5))
    x = np.arange(len(table))
    ax.bar(x - 0.18, table["fusion_auc"], width=0.36, label="Fusion AUC")
    ax.bar(x + 0.18, table["fusion_f1"], width=0.36, label="Fusion F1")
    ax.set_xticks(x)
    ax.set_xticklabels(table["class_name"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_branch_auc(path, table, title):
    fig, ax = plt.subplots(figsize=(max(7, len(table) * 0.8), 4.5))
    x = np.arange(len(table))
    ax.bar(x - 0.25, table["branch1_auc"], width=0.25, label="Branch1 AUC")
    ax.bar(x, table["branch2_auc"], width=0.25, label="Branch2 AUC")
    ax.bar(x + 0.25, table["fusion_auc"], width=0.25, label="Fusion AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(table["class_name"], rotation=45, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_top_bottom(path, table, column, title):
    top = table.sort_values(column, ascending=False).head(10)
    bottom = table.sort_values(column, ascending=True).head(10)
    data = pd.concat([top, bottom]).drop_duplicates("class_name")
    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.45), 5))
    ax.bar(np.arange(len(data)), data[column])
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["class_name"], rotation=60, ha="right")
    ax.set_ylabel(column)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_support_vs_f1(path, table):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(table["support_pos"], table["fusion_f1"])
    ax.set_xlabel("support_pos")
    ax.set_ylabel("fusion_f1")
    ax.set_title("Subdiagnostic Support vs Fusion F1")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def discover_runs(results_dir):
    rows = []
    run_dirs = set()
    for p in results_dir.rglob("config.json"):
        run_dirs.add(p.parent)
    for p in results_dir.rglob("predictions.npz"):
        run_dirs.add(p.parent)

    for run_dir in sorted(run_dirs):
        cfg_path = run_dir / "config.json"
        cfg = load_json(cfg_path)
        experiment = str(cfg.get("experiment") or infer_experiment_from_path(run_dir))
        if experiment not in EXPERIMENTS:
            continue
        pred_path = run_dir / "predictions.npz"
        metrics_path = run_dir / "final_metrics.csv"
        num_classes = ""
        has_branch1 = has_branch2 = has_fusion = False
        has_label_names = any((run_dir / name).is_file() for name in ("label_map.json", "classes.json", "labels.json"))
        note = ""
        if pred_path.is_file():
            try:
                arrays, info = infer_arrays(pred_path)
                if arrays is not None:
                    num_classes = arrays["y_true"].shape[1]
                    has_branch1 = "branch1" in arrays
                    has_branch2 = "branch2" in arrays
                    has_fusion = "fusion" in arrays
                    if info["missing"]:
                        note = "missing fields: " + ";".join(info["missing"])
            except Exception as exc:
                note = f"failed to inspect predictions: {exc}"
        else:
            note = "predictions.npz not found"
        rows.append({
            "experiment": experiment,
            "task_name": get_task_name(experiment),
            "run_dir": str(run_dir),
            "predictions_path": str(pred_path) if pred_path.is_file() else "",
            "final_metrics_path": str(metrics_path) if metrics_path.is_file() else "",
            "config_path": str(cfg_path) if cfg_path.is_file() else "",
            "num_classes": num_classes,
            "has_branch1": has_branch1,
            "has_branch2": has_branch2,
            "has_fusion": has_fusion,
            "has_label_names": has_label_names,
            "note": note,
        })

    # Add expected experiments found only as flat training logs.
    existing = set(row["experiment"] for row in rows)
    for exp in EXPERIMENTS:
        if exp in existing:
            continue
        specific_logs = find_flat_training_logs(results_dir, exp)
        rows.append({
            "experiment": exp,
            "task_name": get_task_name(exp),
            "run_dir": "",
            "predictions_path": "",
            "final_metrics_path": "",
            "config_path": "",
            "num_classes": "",
            "has_branch1": False,
            "has_branch2": False,
            "has_fusion": False,
            "has_label_names": False,
            "note": "no run directory with predictions.npz found" + (f"; flat logs: {len(specific_logs)}" if specific_logs else ""),
        })
    return pd.DataFrame(rows)


def choose_analysis_runs(found_df):
    selected = {}
    for exp in EXPERIMENTS:
        candidates = found_df[(found_df["experiment"] == exp) & (found_df["predictions_path"].astype(str) != "")]
        if candidates.empty:
            continue
        non_debug = candidates[~candidates["run_dir"].astype(str).str.contains("debug", case=False, na=False)]
        pool = non_debug if not non_debug.empty else candidates
        selected[exp] = pool.iloc[-1].to_dict()
    return selected


def threshold_analysis(out_dir, prefix, experiment, task_name, arrays, class_names):
    rows = []
    rows.append({
        "experiment": experiment,
        "task_name": task_name,
        "mode": "default_0.5",
        "threshold_summary": "0.5",
        **summary_metrics(arrays["y_true"], arrays["fusion"], 0.5),
    })
    if "val_y_true" not in arrays or "val_fusion" not in arrays:
        pd.DataFrame(rows).round(4).to_csv(out_dir / f"{prefix}_threshold_comparison.csv", index=False)
        (out_dir / f"{prefix}_per_class_thresholds.json").write_text(
            json.dumps({"note": "missing val predictions; threshold optimization skipped"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return False
    global_thr = optimize_global_threshold(arrays["val_y_true"], arrays["val_fusion"])
    per_class_thr = optimize_per_class_thresholds(arrays["val_y_true"], arrays["val_fusion"])
    rows.append({
        "experiment": experiment,
        "task_name": task_name,
        "mode": "global_val_opt",
        "threshold_summary": str(global_thr),
        **summary_metrics(arrays["y_true"], arrays["fusion"], global_thr),
    })
    rows.append({
        "experiment": experiment,
        "task_name": task_name,
        "mode": "per_class_val_opt",
        "threshold_summary": f"mean={float(np.mean(per_class_thr)):.4f}",
        **summary_metrics(arrays["y_true"], arrays["fusion"], per_class_thr),
    })
    round_df(pd.DataFrame(rows)).to_csv(out_dir / f"{prefix}_threshold_comparison.csv", index=False)
    payload = {
        "search_grid": THRESHOLDS.tolist(),
        "per_class_thresholds": [
            {"class_index": i, "class_name": class_names[i], "threshold": float(per_class_thr[i])}
            for i in range(len(class_names))
        ],
    }
    (out_dir / f"{prefix}_per_class_thresholds.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def analyze_run(row, out_dir, datafolder):
    experiment = row["experiment"]
    task_name = get_task_name(experiment)
    prefix = sanitize_experiment_token(experiment)
    pred_path = Path(row["predictions_path"])
    arrays, info = infer_arrays(pred_path)
    if arrays is None or info["missing"]:
        return None, f"missing fields: {info.get('missing')}"
    num_classes = arrays["y_true"].shape[1]
    class_names, source = load_class_names_from_run(Path(row["run_dir"]), num_classes)
    if not class_names:
        class_names, source = recover_class_names(datafolder, experiment, num_classes)
    if len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]
        source += "; fallback class_i due to length mismatch"

    branch_tables = {}
    for branch in ("branch1", "branch2", "fusion"):
        table = compute_classwise_metrics(arrays["y_true"], arrays[branch], class_names, threshold=0.5)
        branch_tables[branch] = table
        round_df(table).to_csv(out_dir / f"{prefix}_{branch}_classwise.csv", index=False)

    full = build_full_table(branch_tables)
    full_path = out_dir / f"{prefix}_full_table.csv"
    round_df(full).to_csv(full_path, index=False)
    threshold_done = threshold_analysis(out_dir, prefix, experiment, task_name, arrays, class_names)

    if experiment == "exp1.1.1":
        round_df(full).to_csv(out_dir / "exp111_superdiagnostic_full_table.csv", index=False)
        write_superdiagnostic_analysis(out_dir / "exp111_superdiagnostic_analysis.md", full)
        plot_auc_f1_bar(out_dir / "exp111_superdiagnostic_fusion_auc_f1_bar.png", full, "Superdiagnostic Fusion AUC and F1")
        plot_branch_auc(out_dir / "exp111_superdiagnostic_branch_auc_compare.png", full, "Superdiagnostic Branch AUC Comparison")
    elif experiment == "exp1.1":
        round_df(full).to_csv(out_dir / "exp11_subdiagnostic_full_table.csv", index=False)
        write_subdiagnostic_analysis(out_dir / "exp11_subdiagnostic_top_bottom.md", full)
        plot_top_bottom(out_dir / "exp11_subdiagnostic_fusion_auc_top_bottom.png", full, "fusion_auc", "Subdiagnostic Fusion AUC Top/Bottom")
        plot_top_bottom(out_dir / "exp11_subdiagnostic_fusion_f1_top_bottom.png", full, "fusion_f1", "Subdiagnostic Fusion F1 Top/Bottom")
        plot_support_vs_f1(out_dir / "exp11_subdiagnostic_support_vs_f1.png", full)
    elif experiment == "exp3":
        round_df(full).to_csv(out_dir / "exp3_rhythm_full_table.csv", index=False)
        write_task_analysis(out_dir / "exp3_rhythm_analysis.md", full, "rhythm")
        plot_branch_auc(out_dir / "exp3_rhythm_branch_auc_compare.png", full, "Rhythm Branch AUC Comparison")
    elif experiment == "exp2":
        round_df(full).to_csv(out_dir / "exp2_form_full_table.csv", index=False)
        write_task_analysis(out_dir / "exp2_form_analysis.md", full, "form")
        plot_branch_auc(out_dir / "exp2_form_branch_auc_compare.png", full, "Form Branch AUC Comparison")

    return {
        "experiment": experiment,
        "task_name": task_name,
        "num_classes": num_classes,
        "class_source": source,
        "threshold_optimization_done": threshold_done,
        "best_class_auc": full.sort_values("fusion_auc", ascending=False).iloc[0]["class_name"],
        "worst_class_auc": full.sort_values("fusion_auc", ascending=True).iloc[0]["class_name"],
        "best_class_f1": full.sort_values("fusion_f1", ascending=False).iloc[0]["class_name"],
        "worst_class_f1": full.sort_values("fusion_f1", ascending=True).iloc[0]["class_name"],
    }, ""


def write_missing_note(out_dir, filename, title, experiment, task):
    write_markdown(
        out_dir / filename,
        f"""# {title}

This analysis was not generated because no usable `predictions.npz` was found for {experiment} ({task}).
Please copy the run directory containing test predictions into `results/ptbxl/` and rerun `scripts/analyze_ptbxl_classwise.py`.
""",
    )


def write_paper_summary(out_dir, analyzed, missing):
    lines = ["# Paper-ready PTB-XL Classwise Summary", ""]
    if "exp1.1.1" in analyzed:
        lines.append("## Superdiagnostic")
        lines.append("The superdiagnostic task was analyzed from saved test predictions. See `exp111_superdiagnostic_full_table.csv` for per-class branch and fusion metrics.")
    else:
        lines.append("## Superdiagnostic")
        lines.append("Superdiagnostic per-class analysis is pending because `predictions.npz` for exp1.1.1 is missing.")
    if "exp1.1" in analyzed:
        lines.append("\n## Subdiagnostic")
        lines.append("The subdiagnostic task was analyzed from saved test predictions. Fine-grained classes should be interpreted with support and thresholded F1 together.")
    else:
        lines.append("\n## Subdiagnostic")
        lines.append("Subdiagnostic per-class analysis is pending because `predictions.npz` for exp1.1 is missing.")
    if "exp3" in analyzed:
        lines.append("\n## Rhythm")
        lines.append("Rhythm classwise analysis is available in `exp3_rhythm_full_table.csv`.")
    else:
        lines.append("\n## Rhythm")
        lines.append("Rhythm supplementary analysis was skipped because exp3 predictions are missing.")
    if "exp2" in analyzed:
        lines.append("\n## Form")
        lines.append("Form classwise analysis is available in `exp2_form_full_table.csv`.")
    else:
        lines.append("\n## Form")
        lines.append("Form supplementary analysis was skipped because exp2 predictions are missing.")

    lines.append("\n## Chinese Draft Paragraphs")
    lines.append(
        "从类别级结果分析来看，UFML-Net 的双分支结构能够为不同标签提供互补表征：局部形态相关分支更关注波形细节，全局时序分支更关注长程节律模式，融合输出则在多数类别上提供更稳健的综合预测。"
    )
    lines.append(
        "对于细粒度诊断标签，类别间样本数差异会明显影响固定阈值下的 F1 表现。部分类别即使具有较高 AUC，仍可能因为阳性样本较少或预测分数分布偏移而出现较低 F1，因此本文将验证集阈值优化作为补充分析，而不替代默认 0.5 阈值结果。"
    )
    lines.append(
        "上述类别级分析有助于解释模型整体性能来源：当融合结果优于任一单分支时，说明双分支信息具有互补性；当某些类别由单分支占优时，也提示后续可针对特定任务类型进一步优化分支权重或融合策略。"
    )
    if missing:
        lines.append("\n## Missing Predictions")
        lines.extend(f"- {exp}: {reason}" for exp, reason in missing.items())
    write_markdown(out_dir / "paper_ready_classwise_summary.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Analyze saved PTB-XL classwise predictions without retraining.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--datafolder", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "classwise_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    datafolder = find_datafolder(args.datafolder)

    found = discover_runs(results_dir)
    found = found.sort_values(["experiment", "run_dir"]).reset_index(drop=True)
    found.to_csv(out_dir / "found_experiments.csv", index=False)

    field_rows = []
    for row in found.to_dict(orient="records"):
        pred = Path(row["predictions_path"]) if row["predictions_path"] else None
        field_rows.extend(collect_npz_field_rows(row["experiment"], pred))
    pd.DataFrame(field_rows).to_csv(out_dir / "predictions_field_report.csv", index=False)

    selected = choose_analysis_runs(found)
    analyzed = {}
    missing = {}
    for exp in EXPERIMENTS:
        if exp not in selected:
            missing[exp] = "predictions.npz not found"
            if exp == "exp1.1.1":
                write_missing_note(out_dir, "exp111_superdiagnostic_analysis.md", "exp1.1.1 Superdiagnostic Analysis", exp, get_task_name(exp))
            elif exp == "exp1.1":
                write_missing_note(out_dir, "exp11_subdiagnostic_top_bottom.md", "exp1.1 Subdiagnostic Analysis", exp, get_task_name(exp))
            elif exp == "exp3":
                write_missing_note(out_dir, "exp3_rhythm_analysis.md", "exp3 Rhythm Analysis", exp, get_task_name(exp))
            elif exp == "exp2":
                write_missing_note(out_dir, "exp2_form_analysis.md", "exp2 Form Analysis", exp, get_task_name(exp))
            continue
        result, note = analyze_run(selected[exp], out_dir, datafolder)
        if result is None:
            missing[exp] = note
        else:
            analyzed[exp] = result

    write_paper_summary(out_dir, analyzed, missing)

    print("Found experiments:")
    for exp in EXPERIMENTS:
        rows = found[found["experiment"] == exp]
        print(f"- {exp}: {len(rows)} candidate rows")
    print("Successfully analyzed:")
    for exp, info in analyzed.items():
        print(f"- {exp} ({info['task_name']}): {info['num_classes']} classes; best AUC={info['best_class_auc']}; worst AUC={info['worst_class_auc']}; threshold_opt={info['threshold_optimization_done']}")
    print("Missing or skipped:")
    for exp, reason in missing.items():
        print(f"- {exp}: {reason}")
    print("Output directory:", out_dir)
    print("Missing val predictions:", ", ".join(exp for exp, info in analyzed.items() if not info["threshold_optimization_done"]) or "none among analyzed")
    print("Next step: copy predictions.npz for exp1.1.1, exp1.1, exp2, and exp3 into their run directories, then rerun this script.")


if __name__ == "__main__":
    main()
