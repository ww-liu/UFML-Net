#!/bin/bash
set -euo pipefail

DATAFOLDER="${DATAFOLDER:-/root/autodl-tmp/tmp_data}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-1}"
MAX_EPOCH="${MAX_EPOCH:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FORCE="${FORCE:-0}"

RUN_NAMES=(
  CPSC_recipe_ls000_seed1
  CPSC_recipe_ls002_seed1
  CPSC_recipe_ls005_seed1
  CPSC_recipe_ls010_seed1
  CPSC_recipe_lr0001_seed1
  CPSC_recipe_lr0002_seed1
  CPSC_recipe_lr0003_seed1
  CPSC_recipe_weighted_ce_ls002_seed1
)

COMMON_ARGS=(
  --dataset cpsc
  --datafolder "${DATAFOLDER}"
  --experiment exp0
  --model_name ufml_net
  --cpsc_recipe softmax_ce
  --target_mode single_label
  --loss_name ce
  --use_class_weighted_ce true
  --batch_size "${BATCH_SIZE}"
  --max_epoch "${MAX_EPOCH}"
  --seed "${SEED}"
  --use_fdconv_scale true
  --use_frequency_aux false
  --use_ema true
  --use_swa false
  --mutual_feat_weight 0.002
  --mutual_logit_weight 0.02
  --fusion_mode standard
  --selection_score_mode fusion_auc_f1
  --final_eval_mode ensemble
)

run_one() {
  local run_name="$1"
  shift
  local run_dir="${RESULTS_DIR}/cpsc/${run_name}"
  if [ "${FORCE}" != "1" ] && [ -f "${run_dir}/final_metrics.csv" ]; then
    echo "[skip] ${run_name}: ${run_dir}/final_metrics.csv already exists. Set FORCE=1 to rerun."
    return
  fi
  echo "[run] ${run_name}"
  "${PYTHON_BIN}" -u main_mul.py "${COMMON_ARGS[@]}" --run_name "${run_name}" "$@"
}

# A. Label smoothing grid, lr fixed at 0.003.
run_one CPSC_recipe_ls000_seed1 --lr 0.003 --label_smoothing 0.00 --run_note CPSC_recipe_label_smoothing_0.00
run_one CPSC_recipe_ls002_seed1 --lr 0.003 --label_smoothing 0.02 --run_note CPSC_recipe_label_smoothing_0.02
run_one CPSC_recipe_ls005_seed1 --lr 0.003 --label_smoothing 0.05 --run_note CPSC_recipe_label_smoothing_0.05
run_one CPSC_recipe_ls010_seed1 --lr 0.003 --label_smoothing 0.10 --run_note CPSC_recipe_label_smoothing_0.10

# B. Learning-rate grid, label smoothing fixed at 0.02.
run_one CPSC_recipe_lr0001_seed1 --lr 0.001 --label_smoothing 0.02 --run_note CPSC_recipe_lr_0.001
run_one CPSC_recipe_lr0002_seed1 --lr 0.002 --label_smoothing 0.02 --run_note CPSC_recipe_lr_0.002
run_one CPSC_recipe_lr0003_seed1 --lr 0.003 --label_smoothing 0.02 --run_note CPSC_recipe_lr_0.003

# C. Loss recipe anchor. The current focal implementation is multilabel/sigmoid,
# so the safe CPSC loss candidate is weighted CE with label smoothing.
run_one CPSC_recipe_weighted_ce_ls002_seed1 --lr 0.003 --label_smoothing 0.02 --run_note CPSC_weighted_CE_label_smoothing_anchor

"${PYTHON_BIN}" scripts/summarize_cpsc_runs.py \
  --results-root "${RESULTS_DIR}" \
  --mode tuning \
  --output "${RESULTS_DIR}/cpsc_recipe_tuning_summary.csv" \
  --run-names "${RUN_NAMES[@]}"
