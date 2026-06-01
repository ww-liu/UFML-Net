#!/bin/bash

# CPSC-focused recipe sweep. Keep the seed fixed when comparing with MRM.
seed=${1:-1}

recipes=(
  "stable"
  "stable_branch2"
  "stable_single_auc"
  "fdconv_only"
  "uiml_only"
  "mrm_like"
)

for recipe in "${recipes[@]}"
do
  echo "======================================"
  echo "Running CPSC recipe: $recipe | seed: $seed"
  echo "======================================"
  python main_mul.py --model_name ufml_net --experiment exp0 --seed "$seed" --cpsc_recipe "$recipe"
done

