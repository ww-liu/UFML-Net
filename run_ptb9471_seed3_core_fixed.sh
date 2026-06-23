#!/usr/bin/env bash
set -e

SEED=3
PTB_DATAFOLDER=${PTB_DATAFOLDER:-/root/autodl-tmp/data/ptbxl}

run_case () {
  TAG="$1"
  NOTE="$2"
  USE_FDCONV="$3"
  USE_FREQAUX="$4"
  MUTUAL_FEAT="$5"
  MUTUAL_LOGIT="$6"

  echo "============================================================"
  echo "[RUN] ${TAG} | seed=${SEED}"
  echo "============================================================"

  python -u - "$PTB_DATAFOLDER" "$SEED" "$TAG" "$NOTE" "$USE_FDCONV" "$USE_FREQAUX" "$MUTUAL_FEAT" "$MUTUAL_LOGIT" <<'PY'
import sys
from copy import deepcopy
from config import config
from main_mul import train

datafolder, seed, tag, note, use_fdconv, use_freqaux, mutual_feat, mutual_logit = sys.argv[1:]

config.datafolder = datafolder
config.model_name = 'ufml_net'
config.experiment = 'exp0'
config.seed = int(seed)
config.max_epoch = 100

ov = deepcopy(config.experiment_overrides['exp0'])
ov.update({
    'run_tag': tag,
    'run_note': note,
    'selection_score_mode': 'fusion_auc_f1',
    'training_selection_mode': 'standard_val',
    'use_fdconv_scale': use_fdconv.lower() == 'true',
    'use_frequency_aux': use_freqaux.lower() == 'true',
    'use_multiband_frequency_aux': False,
    'use_classwise_frequency_aux': False,
    'mutual_feat_weight': float(mutual_feat),
    'mutual_logit_weight': float(mutual_logit),
    'use_ema': True,
    'use_swa': True,
    'fusion_mode': 'standard',
})
config.experiment_overrides['exp0'] = ov

train(0, config)
PY
}

run_case "ptb9471_full_fixed"      "UIML+FreqAux+FDConvScale+EMA+SWA" true  true  0.005 0.05
run_case "ptb9471_no_fdconv_fixed" "UIML+FreqAux+EMA+SWA"             false true  0.005 0.05
run_case "ptb9471_no_freqaux"      "UIML+FDConvScale+EMA+SWA"         true  false 0.005 0.05
run_case "ptb9471_no_mutual_fixed" "FreqAux+FDConvScale+EMA+SWA"      true  true  0.0   0.0
