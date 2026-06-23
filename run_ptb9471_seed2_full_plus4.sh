#!/usr/bin/env bash
set -e

SEED=${SEED:-2}
PTB_DATAFOLDER=${PTB_DATAFOLDER:-/root/autodl-tmp/data/ptbxl}

run_case () {
  NAME="$1"
  TAG="$2"
  USE_FDCONV="$3"
  USE_FREQAUX="$4"
  MUTUAL_FEAT="$5"
  MUTUAL_LOGIT="$6"
  FUSION_MODE="$7"

  echo "============================================================"
  echo "[RUN] ${NAME} | seed=${SEED} | tag=${TAG}"
  echo "============================================================"

  python -u - "$PTB_DATAFOLDER" "$SEED" "$NAME" "$TAG" "$USE_FDCONV" "$USE_FREQAUX" "$MUTUAL_FEAT" "$MUTUAL_LOGIT" "$FUSION_MODE" <<'PY'
import sys
from config import config
from main_mul import train

datafolder, seed, name, tag, use_fdconv, use_freqaux, mutual_feat, mutual_logit, fusion_mode = sys.argv[1:]

config.datafolder = datafolder
config.model_name = 'ufml_net'
config.experiment = 'exp0'
config.seed = int(seed)
config.max_epoch = 100

config.run_tag = tag
config.run_note = name

config.use_fdconv_scale = use_fdconv.lower() == 'true'
config.use_frequency_aux = use_freqaux.lower() == 'true'
config.mutual_feat_weight = float(mutual_feat)
config.mutual_logit_weight = float(mutual_logit)
config.use_ema = True
config.use_swa = True
config.fusion_mode = fusion_mode

train(0, config)
PY
}

run_case "PTB_FULL_seed${SEED}"           "freqaux_ref"             true  true  0.005 0.05 standard
run_case "PTB_no_fdconv_seed${SEED}"      "ptb9471_no_fdconv"       false true  0.005 0.05 standard
run_case "PTB_no_freqaux_seed${SEED}"     "ptb9471_no_freqaux"      true  false 0.005 0.05 standard
run_case "PTB_no_mutual_seed${SEED}"      "ptb9471_no_mutual"       true  true  0.0   0.0  standard
run_case "PTB_simple_average_seed${SEED}" "ptb9471_simple_average"  true  true  0.005 0.05 simple_average
