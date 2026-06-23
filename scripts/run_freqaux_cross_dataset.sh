#!/bin/bash
set -e

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name F0_PTB_freqaux_on \
  --target_mode multilabel \
  --loss_name focal \
  --use_frequency_aux true \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/data/ptbxl \
  --experiment exp0 \
  --model_name ufml_net \
  --run_name F1_PTB_freqaux_off \
  --target_mode multilabel \
  --loss_name focal \
  --use_frequency_aux false \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/tmp_data \
  --model_name ufml_net \
  --cpsc_recipe softmax_ce \
  --run_name F2_CPSC_freqaux_off \
  --use_frequency_aux false \
  --seed 1

python main_mul.py \
  --datafolder /root/autodl-tmp/tmp_data \
  --model_name ufml_net \
  --cpsc_recipe softmax_ce \
  --run_name F3_CPSC_freqaux_on \
  --use_frequency_aux true \
  --seed 1
