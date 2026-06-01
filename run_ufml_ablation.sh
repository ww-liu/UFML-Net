#!/bin/bash

models=(
  "mrm_baseline"
  "ufml_no_uncertainty"
  "ufml_no_frequency"
  "ufml_no_fdconv"
  "ufml_no_mutual"
  "ufml_no_ema_swa"
  "ufml_net"
)

for model in "${models[@]}"
do
  echo "======================================"
  echo "Running UFML ablation model: $model"
  echo "======================================"
  python main_mul.py --model_name "$model" --experiment exp1.1.1 --seed 42
done
