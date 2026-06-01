#!/bin/bash

models=(
  "cnn1d"
  "resnet1d"
  "inceptiontime1d"
  "bilstm"
  "cnn_bilstm"
  "cnn_transformer"
)

experiments=(
  "exp0"
  "exp1"
  "exp1.1"
  "exp1.1.1"
  "exp2"
  "exp3"
)

for model in "${models[@]}"
do
  for experiment in "${experiments[@]}"
  do
    echo "======================================"
    echo "Running clean baseline model: $model | PTB-XL $experiment"
    echo "======================================"
    python main_mul.py --model_name "$model" --experiment "$experiment" --seed 42
  done
done
