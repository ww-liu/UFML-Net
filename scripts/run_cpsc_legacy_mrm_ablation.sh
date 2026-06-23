#!/usr/bin/env bash
set -euo pipefail

DATAFOLDER="${DATAFOLDER:-/root/autodl-tmp/tmp_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-1}"
MAX_EPOCH="${MAX_EPOCH:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FORCE="${FORCE:-0}"

COMMON_ARGS=(
  --dataset cpsc
  --datafolder "${DATAFOLDER}"
  --experiment exp0
  --model_name ufml_net
  --seed "${SEED}"
  --cpsc_recipe legacy_bce
  --target_mode multilabel
  --loss_name bce
  --training_selection_mode standard_val
  --selection_score_mode single_auc
  --fusion_mode branch2_only
  --final_eval_mode checkpoint
  --max_epoch "${MAX_EPOCH}"
  --batch_size "${BATCH_SIZE}"
  --lr 0.001
  --weight_decay 0.0
  --gradient_clip_norm 0.0
  --cpsc_split_mode legacy_fold
  --cpsc_use_stratified_split false
  --cpsc_use_weighted_sampler false
  --use_ema false
  --use_swa false
)

run_one() {
  local run_name="$1"
  shift
  local metrics_path="results/cpsc/${run_name}/final_metrics.csv"
  if [[ "${FORCE}" != "1" && -f "${metrics_path}" ]]; then
    echo "[skip] ${run_name}: ${metrics_path} already exists. Set FORCE=1 to rerun."
    return
  fi

  echo "[run] ${run_name}"
  "${PYTHON_BIN}" -u main_mul.py \
    "${COMMON_ARGS[@]}" \
    --run_name "${run_name}" \
    "$@"
}

run_one "CPSC_LEGACY_E0_clean_mrm_seed${SEED}" \
  --run_note legacy_clean_mrm_no_ufml_modules \
  --mutual_feat_weight 0.0 \
  --mutual_logit_weight 0.0 \
  --use_fdconv_scale false \
  --use_frequency_aux false

run_one "CPSC_LEGACY_E1_uiml_seed${SEED}" \
  --run_note legacy_plus_uncertainty_informed_mutual_learning \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --use_fdconv_scale false \
  --use_frequency_aux false

run_one "CPSC_LEGACY_E2_fdconv_seed${SEED}" \
  --run_note legacy_plus_fdconv_scale_only \
  --mutual_feat_weight 0.0 \
  --mutual_logit_weight 0.0 \
  --use_fdconv_scale true \
  --use_frequency_aux false

run_one "CPSC_LEGACY_E3_uiml_fdconv_seed${SEED}" \
  --run_note legacy_plus_uiml_and_fdconv_scale \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --use_fdconv_scale true \
  --use_frequency_aux false

run_one "CPSC_LEGACY_E4_uiml_fdconv_freqaux_seed${SEED}" \
  --run_note legacy_plus_uiml_fdconv_scale_and_frequency_aux \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --use_fdconv_scale true \
  --use_frequency_aux true

"${PYTHON_BIN}" scripts/summarize_cpsc_runs.py \
  --mode ablation \
  --output results/cpsc_legacy_mrm_ablation_summary.csv \
  --full-run-name "CPSC_LEGACY_E0_clean_mrm_seed${SEED}" \
  --run-names \
    "CPSC_LEGACY_E0_clean_mrm_seed${SEED}" \
    "CPSC_LEGACY_E1_uiml_seed${SEED}" \
    "CPSC_LEGACY_E2_fdconv_seed${SEED}" \
    "CPSC_LEGACY_E3_uiml_fdconv_seed${SEED}" \
    "CPSC_LEGACY_E4_uiml_fdconv_freqaux_seed${SEED}"
