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
  CPSC_UFML_FULL_seed1
  CPSC_UFML_no_fdconv_seed1
  CPSC_UFML_no_mutual_seed1
  CPSC_UFML_simple_average_seed1
  CPSC_UFML_freqaux_on_seed1
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
  --label_smoothing 0.02
  --batch_size "${BATCH_SIZE}"
  --max_epoch "${MAX_EPOCH}"
  --seed "${SEED}"
  --use_ema true
  --use_swa false
  --selection_score_mode fusion_auc_f1
  --final_eval_mode ensemble
  --lr 0.003
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

run_one CPSC_UFML_FULL_seed1 \
  --use_fdconv_scale true \
  --use_frequency_aux false \
  --mutual_feat_weight 0.002 \
  --mutual_logit_weight 0.02 \
  --fusion_mode standard \
  --run_note CPSC_UFML_FULL_no_freqaux_weighted_ce

run_one CPSC_UFML_no_fdconv_seed1 \
  --use_fdconv_scale false \
  --use_frequency_aux false \
  --mutual_feat_weight 0.002 \
  --mutual_logit_weight 0.02 \
  --fusion_mode standard \
  --run_note CPSC_UFML_no_fdconv

run_one CPSC_UFML_no_mutual_seed1 \
  --use_fdconv_scale true \
  --use_frequency_aux false \
  --mutual_feat_weight 0.0 \
  --mutual_logit_weight 0.0 \
  --fusion_mode standard \
  --run_note CPSC_UFML_no_mutual

run_one CPSC_UFML_simple_average_seed1 \
  --use_fdconv_scale true \
  --use_frequency_aux false \
  --mutual_feat_weight 0.002 \
  --mutual_logit_weight 0.02 \
  --fusion_mode simple_average \
  --run_note CPSC_UFML_simple_average

run_one CPSC_UFML_freqaux_on_seed1 \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --mutual_feat_weight 0.002 \
  --mutual_logit_weight 0.02 \
  --fusion_mode standard \
  --run_note CPSC_UFML_freqaux_on_check

"${PYTHON_BIN}" scripts/summarize_cpsc_runs.py \
  --results-root "${RESULTS_DIR}" \
  --mode ablation \
  --full-run-name CPSC_UFML_FULL_seed1 \
  --output "${RESULTS_DIR}/cpsc_ablation_summary.csv" \
  --run-names "${RUN_NAMES[@]}"
