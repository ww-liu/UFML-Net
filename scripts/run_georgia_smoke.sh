#!/usr/bin/env bash
set -euo pipefail

GEORGIA_ROOT="${GEORGIA_ROOT:-/root/autodl-tmp/data/georgia}"

python main_mul.py \
  --dataset georgia \
  --datafolder "$GEORGIA_ROOT" \
  --model_name ufml_net \
  --run_name smoke_georgia_min100_seed1 \
  --target_mode multilabel \
  --loss_name focal \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --use_ema true \
  --use_swa false \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --fusion_mode standard \
  --selection_score_mode fusion_auc_f1 \
  --training_selection_mode standard_val \
  --max_epoch 1 \
  --seed 1
