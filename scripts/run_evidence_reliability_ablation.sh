#!/usr/bin/env bash
set -e

SEED=${SEED:-1}
MAX_EPOCH=${MAX_EPOCH:-1}
PTB_DATAFOLDER=${PTB_DATAFOLDER:-/root/autodl-tmp/data/ptbxl}
CPSC_DATAFOLDER=${CPSC_DATAFOLDER:-/root/autodl-tmp/tmp_data}

COMMON_ARGS=(
  --model_name ufml_net
  --experiment exp0
  --seed "$SEED"
  --max_epoch "$MAX_EPOCH"
)

run_case() {
  local dataset_name="$1"
  local datafolder="$2"
  local run_name="$3"
  shift 3

  echo "============================================================"
  echo "[RUN] ${dataset_name} | ${run_name} | seed=${SEED} | max_epoch=${MAX_EPOCH}"
  echo "============================================================"

  python -u main_mul.py \
    "${COMMON_ARGS[@]}" \
    --datafolder "$datafolder" \
    --run_name "$run_name" \
    "$@"
}

run_ptb() {
  local run_name="$1"
  shift
  run_case "PTB-XL" "$PTB_DATAFOLDER" "$run_name" \
    --target_mode multilabel \
    --loss_name focal \
    --use_fdconv_scale true \
    --use_frequency_aux true \
    --use_ema true \
    --use_swa true \
    --mutual_feat_weight 0.005 \
    --mutual_logit_weight 0.05 \
    "$@"
}

run_cpsc() {
  local run_name="$1"
  shift
  run_case "CPSC" "$CPSC_DATAFOLDER" "$run_name" \
    --cpsc_recipe softmax_ce \
    "$@"
}

run_ptb  "E0_PTB_entropy_baseline"      --reliability_mode entropy  --fusion_mode standard
run_ptb  "E1_PTB_evidence_reliability"  --reliability_mode evidence --fusion_mode standard
run_ptb  "E2_PTB_hybrid_reliability"    --reliability_mode hybrid   --fusion_mode standard
run_ptb  "E3_PTB_evidence_opinion_fuse" --reliability_mode entropy  --fusion_mode evidence_opinion
run_ptb  "E4_PTB_evidence_consistency"  --reliability_mode evidence --fusion_mode standard --use_evidence_consistency true --evidence_consistency_weight 0.01

run_cpsc "E0_CPSC_entropy_baseline"      --reliability_mode entropy  --fusion_mode standard
run_cpsc "E1_CPSC_evidence_reliability"  --reliability_mode evidence --fusion_mode standard
run_cpsc "E2_CPSC_hybrid_reliability"    --reliability_mode hybrid   --fusion_mode standard
run_cpsc "E3_CPSC_evidence_opinion_fuse" --reliability_mode entropy  --fusion_mode evidence_opinion
run_cpsc "E4_CPSC_evidence_consistency"  --reliability_mode evidence --fusion_mode standard --use_evidence_consistency true --evidence_consistency_weight 0.01
