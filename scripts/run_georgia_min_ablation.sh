#!/usr/bin/env bash
set -euo pipefail

GEORGIA_ROOT="${GEORGIA_ROOT:-/root/autodl-tmp/data/georgia}"
MAX_EPOCH="${MAX_EPOCH:-80}"
SEED="${SEED:-1}"

run_one() {
  local run_name="$1"
  shift
  python main_mul.py \
    --dataset georgia \
    --datafolder "$GEORGIA_ROOT" \
    --model_name ufml_net \
    --run_name "$run_name" \
    --target_mode multilabel \
    --loss_name focal \
    --use_ema true \
    --use_swa false \
    --selection_score_mode fusion_auc_f1 \
    --training_selection_mode standard_val \
    --max_epoch "$MAX_EPOCH" \
    --seed "$SEED" \
    "$@"
}

run_one G0_Georgia_FULL \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --fusion_mode standard

run_one G1_Georgia_no_fdconv \
  --use_fdconv_scale false \
  --use_frequency_aux true \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --fusion_mode standard

run_one G2_Georgia_no_freqaux \
  --use_fdconv_scale true \
  --use_frequency_aux false \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --fusion_mode standard

run_one G3_Georgia_no_mutual \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --mutual_feat_weight 0.0 \
  --mutual_logit_weight 0.0 \
  --fusion_mode standard

run_one G4_Georgia_simple_average \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --fusion_mode simple_average
