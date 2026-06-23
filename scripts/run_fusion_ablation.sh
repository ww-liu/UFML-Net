#!/bin/bash
set -e

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name A8_PTB_no_fusion_residual \
  --target_mode multilabel \
  --loss_name focal \
  --fusion_residual_weight 0.0 \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name A9_PTB_simple_average \
  --target_mode multilabel \
  --loss_name focal \
  --fusion_mode simple_average \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/tmp_data \
  --model_name ufml_net \
  --cpsc_recipe softmax_ce \
  --run_name A8_CPSC_no_fusion_residual \
  --fusion_residual_weight 0.0 \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/tmp_data \
  --model_name ufml_net \
  --cpsc_recipe softmax_ce \
  --run_name A9_CPSC_simple_average \
  --fusion_mode simple_average \
  --seed 1
