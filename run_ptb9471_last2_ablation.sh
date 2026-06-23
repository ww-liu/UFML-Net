#!/usr/bin/env bash
set -e

python -c "from config import config; from main_mul import train; config.datafolder='/root/autodl-tmp/data/ptbxl'; config.model_name='ufml_net'; config.experiment='exp0'; config.seed=1; config.max_epoch=100; exp=config.experiment_overrides['exp0']; exp.update({'run_tag':'ptb9471_no_mutual','run_note':'ptb9471_no_mutual','mutual_feat_weight':0.0,'mutual_logit_weight':0.0}); train(0, config)"

python -c "from config import config; from main_mul import train; config.datafolder='/root/autodl-tmp/data/ptbxl'; config.model_name='ufml_net'; config.experiment='exp0'; config.seed=1; config.max_epoch=100; exp=config.experiment_overrides['exp0']; exp.update({'run_tag':'ptb9471_no_fusion_residual','run_note':'ptb9471_no_fusion_residual','fusion_residual_weight':0.0}); train(0, config)"
