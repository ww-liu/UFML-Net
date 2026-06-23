#!/bin/bash
set -e

# Core multi-seed reruns. Seed 1 has already been run, so the default only
# executes missing seed 2 and 3 jobs. Override with: SEEDS="1 2 3" bash ...
SEEDS=${SEEDS:-"2 3"}
PTB_DATAFOLDER=${PTB_DATAFOLDER:-"/root/autodl-tmp/data/ptbxl"}
CPSC_DATAFOLDER=${CPSC_DATAFOLDER:-"/root/autodl-tmp/tmp_data"}

run_if_missing() {
  dataset="$1"
  run_name="$2"
  shift 2
  final_metrics="results/${dataset}/${run_name}/final_metrics.csv"
  if [ -f "$final_metrics" ]; then
    echo "[skip] ${run_name} already has ${final_metrics}"
    return
  fi
  echo "[run] ${run_name}"
  python main_mul.py "$@"
}

for seed in $SEEDS; do
  run_if_missing ptbxl "PTB_FULL_seed${seed}" \
    --datafolder "$PTB_DATAFOLDER" \
    --experiment exp0 \
    --model_name ufml_net \
    --run_name "PTB_FULL_seed${seed}" \
    --run_tag "PTB_FULL_seed${seed}" \
    --target_mode multilabel \
    --loss_name focal \
    --use_fdconv_scale true \
    --use_frequency_aux true \
    --mutual_feat_weight 0.005 \
    --mutual_logit_weight 0.05 \
    --use_ema true \
    --use_swa true \
    --fusion_mode standard \
    --seed "$seed"

  run_if_missing ptbxl "PTB_no_fdconv_seed${seed}" \
    --datafolder "$PTB_DATAFOLDER" \
    --experiment exp0 \
    --model_name ufml_net \
    --run_name "PTB_no_fdconv_seed${seed}" \
    --run_tag "PTB_no_fdconv_seed${seed}" \
    --target_mode multilabel \
    --loss_name focal \
    --use_fdconv_scale false \
    --use_frequency_aux true \
    --mutual_feat_weight 0.005 \
    --mutual_logit_weight 0.05 \
    --use_ema true \
    --use_swa true \
    --fusion_mode standard \
    --seed "$seed"

  run_if_missing ptbxl "PTB_no_freqaux_seed${seed}" \
    --datafolder "$PTB_DATAFOLDER" \
    --experiment exp0 \
    --model_name ufml_net \
    --run_name "PTB_no_freqaux_seed${seed}" \
    --run_tag "PTB_no_freqaux_seed${seed}" \
    --target_mode multilabel \
    --loss_name focal \
    --use_fdconv_scale true \
    --use_frequency_aux false \
    --mutual_feat_weight 0.005 \
    --mutual_logit_weight 0.05 \
    --use_ema true \
    --use_swa true \
    --fusion_mode standard \
    --seed "$seed"

  run_if_missing ptbxl "PTB_no_mutual_seed${seed}" \
    --datafolder "$PTB_DATAFOLDER" \
    --experiment exp0 \
    --model_name ufml_net \
    --run_name "PTB_no_mutual_seed${seed}" \
    --run_tag "PTB_no_mutual_seed${seed}" \
    --target_mode multilabel \
    --loss_name focal \
    --use_fdconv_scale true \
    --use_frequency_aux true \
    --mutual_feat_weight 0.0 \
    --mutual_logit_weight 0.0 \
    --use_ema true \
    --use_swa true \
    --fusion_mode standard \
    --seed "$seed"

  run_if_missing cpsc "CPSC_DEFAULT_seed${seed}" \
    --datafolder "$CPSC_DATAFOLDER" \
    --experiment exp0 \
    --model_name ufml_net \
    --cpsc_recipe softmax_ce \
    --run_name "CPSC_DEFAULT_seed${seed}" \
    --run_tag "CPSC_DEFAULT_seed${seed}" \
    --fusion_mode standard \
    --seed "$seed"

  run_if_missing cpsc "CPSC_no_fdconv_seed${seed}" \
    --datafolder "$CPSC_DATAFOLDER" \
    --experiment exp0 \
    --model_name ufml_net \
    --cpsc_recipe softmax_ce \
    --run_name "CPSC_no_fdconv_seed${seed}" \
    --run_tag "CPSC_no_fdconv_seed${seed}" \
    --fusion_mode standard \
    --use_fdconv_scale false \
    --seed "$seed"

  run_if_missing cpsc "CPSC_freqaux_on_seed${seed}" \
    --datafolder "$CPSC_DATAFOLDER" \
    --experiment exp0 \
    --model_name ufml_net \
    --cpsc_recipe softmax_ce \
    --run_name "CPSC_freqaux_on_seed${seed}" \
    --run_tag "CPSC_freqaux_on_seed${seed}" \
    --fusion_mode standard \
    --use_frequency_aux true \
    --seed "$seed"

  run_if_missing cpsc "CPSC_no_mutual_seed${seed}" \
    --datafolder "$CPSC_DATAFOLDER" \
    --experiment exp0 \
    --model_name ufml_net \
    --cpsc_recipe softmax_ce \
    --run_name "CPSC_no_mutual_seed${seed}" \
    --run_tag "CPSC_no_mutual_seed${seed}" \
    --fusion_mode standard \
    --mutual_feat_weight 0.0 \
    --mutual_logit_weight 0.0 \
    --seed "$seed"
done

python scripts/summarize_core_multiseed.py --results-dir results --output results/summary_core_multiseed.csv
