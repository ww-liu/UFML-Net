# -*- coding: utf-8 -*-

from copy import copy

from .ufml_net import UFMLNet


def build_ufml_ablation(model_name, num_classes, config):
    cfg = copy(config)
    name = model_name.lower()

    if name == 'ufml_no_uncertainty':
        cfg.uncertainty_min_weight = 1.0
        cfg.mutual_feat_weight = 0.0
        cfg.mutual_logit_weight = 0.0
    elif name == 'ufml_no_frequency':
        cfg.use_frequency_aux = False
        cfg.use_multiband_frequency_aux = False
        cfg.use_classwise_frequency_aux = False
        cfg.use_tf_branch = False
        cfg.use_periodic_aux_branch = False
    elif name == 'ufml_no_fdconv':
        cfg.use_fdconv_scale = False
    elif name == 'ufml_no_dswin':
        cfg.use_adaptive_temporal_window = False
        cfg.use_multirange_reweight = False
    elif name == 'ufml_no_mutual':
        cfg.mutual_feat_weight = 0.0
        cfg.mutual_logit_weight = 0.0
    elif name == 'ufml_no_ema_swa':
        cfg.use_ema = False
        cfg.use_swa = False
        cfg.final_eval_mode = 'checkpoint'
    elif name == 'ufml_fdconv_only':
        cfg.use_fdconv_scale = True
        cfg.use_frequency_aux = False
        cfg.use_multiband_frequency_aux = False
        cfg.use_classwise_frequency_aux = False
        cfg.mutual_feat_weight = 0.0
        cfg.mutual_logit_weight = 0.0
    elif name == 'ufml_dswin_only':
        cfg.use_fdconv_scale = False
        cfg.use_frequency_aux = False
        cfg.use_adaptive_temporal_window = True
        cfg.use_multirange_reweight = False
        cfg.mutual_feat_weight = 0.0
        cfg.mutual_logit_weight = 0.0
    elif name == 'ufml_frequency_only':
        cfg.use_fdconv_scale = False
        cfg.use_frequency_aux = True
        cfg.use_multiband_frequency_aux = False
        cfg.mutual_feat_weight = 0.0
        cfg.mutual_logit_weight = 0.0
    elif name == 'ufml_fdconv_dswin':
        cfg.use_fdconv_scale = True
        cfg.use_adaptive_temporal_window = True
        cfg.use_frequency_aux = False
    elif name == 'ufml_dswin':
        cfg.use_fdconv_scale = False
        cfg.use_adaptive_temporal_window = True
        cfg.use_frequency_aux = False
    elif name == 'ufml_fdconv':
        cfg.use_fdconv_scale = True
        cfg.use_frequency_aux = False
    else:
        raise ValueError(f'Unknown UFML ablation model_name: {model_name}')

    return UFMLNet(num_classes=num_classes, config=cfg)
