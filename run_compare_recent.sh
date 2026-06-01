#!/bin/bash

models=(
  "dbfnet"
  # "ecg_gtmd"  # External baseline: graph construction and multi-domain preprocessing required.
  # "bmirc"     # External baseline: self-supervised pretraining/fine-tuning required.
  # "lcd"       # External baseline: self-supervised pretraining/fine-tuning required.
)

for model in "${models[@]}"
do
  echo "======================================"
  echo "Running recent baseline: $model"
  echo "======================================"
  python main_mul.py --model_name "$model" --experiment exp1.1.1 --seed 42
done
