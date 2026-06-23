#!/bin/bash
set -e

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name E0_PTB_FULL \
  --target_mode multilabel \
  --loss_name focal \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --use_ema true \
  --use_swa true \
  --fusion_mode standard \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name A1_PTB_no_fdconv \
  --target_mode multilabel \
  --loss_name focal \
  --use_fdconv_scale false \
  --use_frequency_aux true \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --use_ema true \
  --use_swa true \
  --fusion_mode standard \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name A2_PTB_no_freqaux \
  --target_mode multilabel \
  --loss_name focal \
  --use_fdconv_scale true \
  --use_frequency_aux false \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --use_ema true \
  --use_swa true \
  --fusion_mode standard \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name A4_PTB_no_mutual \
  --target_mode multilabel \
  --loss_name focal \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --mutual_feat_weight 0.0 \
  --mutual_logit_weight 0.0 \
  --use_ema true \
  --use_swa true \
  --fusion_mode standard \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name A8_PTB_no_fusion_residual \
  --target_mode multilabel \
  --loss_name focal \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --fusion_residual_weight 0.0 \
  --use_ema true \
  --use_swa true \
  --fusion_mode standard \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name A9_PTB_simple_average \
  --target_mode multilabel \
  --loss_name focal \
  --use_fdconv_scale true \
  --use_frequency_aux true \
  --mutual_feat_weight 0.005 \
  --mutual_logit_weight 0.05 \
  --use_ema true \
  --use_swa true \
  --fusion_mode simple_average \
  --seed 1
