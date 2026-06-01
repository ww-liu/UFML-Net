# -*- coding: utf-8 -*-

class Config:

    seed = 1

    # path
    #datafolder = '/root/autodl-tmp/data/ptbxl'
    datafolder = '/root/autodl-tmp/tmp_data'
    # datafolder = '../data/hf/'

    #
    '''
    experiment = exp0, exp1, exp1.1, exp1.1.1, exp2, exp3
    '''
    experiment = 'exp0'
    experiment_list = ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']
    run_all_experiments = False

    # for train
    '''
    ufml_net, mrm_baseline, cnn1d, resnet1d, inceptiontime1d, bilstm, cnn_bilstm, cnn_transformer, dbfnet
    '''
    model_name = 'ufml_net'
    model_list = [
        'ufml_net',
        'mrm_baseline',
        'cnn1d',
        'resnet1d',
        'inceptiontime1d',
        'cnn_bilstm',
    ]
    run_all_models = False

    model_name2 = 'MyNet'

    batch_size = 64

    max_epoch = 100
    num_epochs = 100
    use_early_stopping = False
    early_stop_patience = 20
    early_stop_min_delta = 0.0001

    lr = 0.001
    lr_milestones = [50, 100, 120]
    lr_gamma = 0.1
    sampling_frequency = 100
    dataset_name = 'auto'
    # selection_score_mode:
    # 'single_auc' -> select the epoch with the highest AUC among branch1/branch2/fusion.
    # 'fusion_auc' -> select the epoch with the highest fusion AUC.
    # 'fusion_auc_f1' -> old behavior, select by fusion AUC + fusion F1.
    selection_score_mode = 'fusion_auc_f1'
    # Keep the old test-set-selection workflow by default. Set this to
    # 'standard_val' only when you want a stricter paper-style protocol.
    training_selection_mode = 'standard_val'
    run_note = 'uiml_freqaux_stable'
    allow_parallel_same_experiment = True
    run_lock_stale_hours = 12

    device_num = 1

    # eg: MyNet6View_all_checkpoint_best_tpr.pth
    checkpoints = 'MyNet6View_exp0_checkpoint_best_auc.pth'
    # Keep checkpoint files small and robust. The best validation ensemble is
    # kept in memory for the final test, so saving all ensemble weights is not
    # necessary during a normal uninterrupted run.
    save_ensemble_checkpoints = False
    # 'ensemble' uses model + EMA + SWA for final testing when available.
    # 'checkpoint' uses only the saved best checkpoint.
    final_eval_mode = 'ensemble'

    # knowledge distillation param
    alpha = 0.5
    temperature = 2

    # uncertainty-informed mutual learning
    mutual_temperature = 2.0
    feature_temperature = 2.0
    mutual_feat_weight = 0.005
    mutual_logit_weight = 0.05
    use_branch_complement_loss = False
    branch_complement_weight = 0.0005
    branch_complement_rampup_epochs = 20
    use_selective_mutual_learning = False
    selective_mutual_threshold = 0.65
    selective_mutual_min_keep_ratio = 0.25
    selective_mutual_power = 1.0
    selective_mutual_label_agreement = True
    uncertainty_rampup_epochs = 25
    uncertainty_min_weight = 0.3
    fusion_conf_power = 1.5
    branch1_fusion_bias = 1.0
    branch2_fusion_bias = 1.08
    fusion_residual_weight = 0.18
    fusion_positive_margin = 0.55
    fusion_gate_power = 1.5
    fusion_logit_blend = 0.35
    fusion_logit_scale = 1.1
    use_fusion_f1_boost = False
    fusion_f1_boost_weight = 0.03
    fusion_f1_boost_margin = 0.48
    fusion_f1_boost_power = 2.0
    fusion_mode = 'standard'
    use_ema = True
    use_swa = True
    ema_decay = 0.999
    ema_start = 10
    swa_start = 45
    # UFML-Net/MRM-style ECG experiments were tuned without Adam weight decay.
    # A global 1e-4 value made PTB-XL exp0 underfit badly after the registry
    # refactor, so keep the strong reference setting here.
    weight_decay = 0.0
    gradient_clip_norm = 0.0
    skip_nonfinite_batches = True
    loss_name = 'focal'
    target_mode = 'multilabel'
    label_smoothing = 0.0
    use_class_weighted_ce = False
    ce_class_weights = None
    save_dir = './checkpoints'
    result_dir = './results'

    # Paper-inspired lightweight enhancements.
    # These modules are residual gates on top of the restored strong baseline.
    use_shape_enhance = False
    shape_gamma_init = 0.01
    use_fdconv_scale = True
    use_lead_calibration = False
    lead_gamma_init = 0.005
    use_lead_relation_graph = False
    lead_relation_graph_dim = 32
    lead_relation_graph_gamma_init = 0.0015
    use_lcd_regularization = False
    lcd_weight = 0.001
    lcd_gamma = 0.005
    lcd_warmup_epochs = 10
    lcd_projection_dim = 64
    lcd_crop_ratio_low = 0.85
    lcd_mask_ratio_up = 0.10
    use_lcdr_pretrained = False
    lcdr_pretrained_path = ''
    # 'shallow' loads only low-level ECG representation layers from LCDR.
    # 'all' restores the earlier full checkpoint loading behavior.
    lcdr_pretrained_load_mode = 'shallow'
    lcdr_pretrained_prefixes = ['shape_enhance', 'lead_calibration', 'lead_relation_graph', 'BaseBlock']
    lcdr_pretrain_epochs = 20
    lcdr_pretrain_lr = 0.001
    lcdr_pretrain_weight_decay = 0.0001
    lcdr_pretrain_projection_dim = 128
    lcdr_pretrain_hidden_dim = 256
    lcdr_pretrain_inter_weight = 0.5
    lcdr_pretrain_decor_weight = 0.02
    lcdr_pretrain_context_scale = 0.05
    use_frequency_gate = False
    frequency_gamma_init = 0.02
    use_adaptive_temporal_window = False
    atwa_gamma_init = 0.003
    use_multirange_reweight = False
    multirange_gamma_init = 0.002
    use_frequency_aux = True
    frequency_aux_gamma_init = 0.015
    use_multiband_frequency_aux = False
    multiband_frequency_splits = (0.18, 0.42, 1.0)
    use_classwise_frequency_aux = False
    classwise_frequency_aux_init = 1.0
    frequency_aux_uncertainty_power = 1.5
    frequency_aux_disagreement_power = 1.0
    use_auc_ranking_loss = False
    auc_ranking_loss_weight = 0.005
    auc_ranking_margin = 0.1
    use_tf_branch = False
    use_periodic_aux_branch = False
    periodic_top_k = 3
    periodic_min_period = 8
    periodic_max_period = 128
    tf_n_fft = 64
    tf_hop_length = 16
    tf_win_length = 64
    tf_loss_weight = 0.2
    tf_fusion_alpha = 0.25
    tf_fusion_disagreement_power = 1.0
    use_category_fusion = False
    category_fusion_loss_weight = 0.1
    category_fusion_alpha = 0.15
    use_fda_augmentation = False
    fda_prob = 0.5
    fda_strength = 0.08
    fda_importance_power = 1.5
    use_val_threshold_optimization = False
    report_val_threshold_metrics = False
    threshold_optimization_mode = 'global'
    threshold_grid = [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]
    threshold_coordinate_passes = 2
    threshold_min_f1_gain = 0.003

    # CPSC is a single-label task with a different class distribution from PTB-XL.
    # Keep PTB-XL defaults untouched, and apply these only when dataset == cpsc.
    cpsc_use_stratified_split = True
    cpsc_use_weighted_sampler = False
    # CPSC recipe:
    # 'softmax_ce' -> single-label softmax/CE recipe for the CPSC task.
    # 'stable' -> reproduce the 2026-05-15 standard_val result
    #             (UIML + FDConvScale + EMA/SWA, no FreqAux).
    # 'ptb_exp0' -> try the PTB-XL exp0 recipe on CPSC
    #               (adds FreqAux to the stable recipe).
    # 'mrm_like' -> conservative recipe close to the base MRM training setup.
    # 'uiml_only' -> keep uncertainty-informed mutual learning, remove FDConvScale.
    # 'fdconv_only' -> keep FDConvScale, remove uncertainty-informed mutual learning.
    cpsc_recipe = 'softmax_ce'
    cpsc_overrides = {
        'use_frequency_aux': False,
        'use_multiband_frequency_aux': False,
        'use_classwise_frequency_aux': False,
        'use_tf_branch': False,
        'use_periodic_aux_branch': False,
        'use_category_fusion': False,
        'use_lead_relation_graph': False,
        'use_lcd_regularization': False,
        'use_fda_augmentation': False,
        'use_selective_mutual_learning': False,
        'use_auc_ranking_loss': False,
        'use_branch_complement_loss': False,
        'use_adaptive_temporal_window': False,
        'use_multirange_reweight': False,
        'mutual_feat_weight': 0.005,
        'mutual_logit_weight': 0.05,
        'selection_score_mode': 'fusion_auc_f1',
        'run_tag': 'v2',
        'run_note': 'uiml_freqaux_stable',
        'max_epoch': 100,
        'use_early_stopping': False,
        'early_stop_patience': 60,
        'lr_milestones': [50, 100, 120],
        'use_fdconv_scale': True,
        'use_ema': True,
        'use_swa': True,
        # Keep validation selection and final testing on the same evaluation
        # object. The best ensemble states are kept in memory, so this does not
        # require saving a large ensemble checkpoint during normal runs.
        'final_eval_mode': 'ensemble',
        'fusion_mode': 'standard',
        'branch2_fusion_bias': 1.08,
        'fusion_residual_weight': 0.18,
        'cpsc_use_stratified_split': True,
    }
    cpsc_recipe_overrides = {
        'softmax_ce': {
            'target_mode': 'single_label',
            'loss_name': 'ce',
            'label_smoothing': 0.02,
            'lr': 0.003,
            'weight_decay': 0.00001,
            'gradient_clip_norm': 5.0,
            'cpsc_use_weighted_sampler': False,
            'use_class_weighted_ce': True,
            'use_frequency_aux': False,
            'frequency_aux_gamma_init': 0.015,
            'use_fda_augmentation': False,
            'use_category_fusion': False,
            'category_fusion_loss_weight': 0.1,
            'category_fusion_alpha': 0.15,
            'use_periodic_aux_branch': False,
            'periodic_top_k': 3,
            'periodic_min_period': 8,
            'periodic_max_period': 128,
            'tf_loss_weight': 0.2,
            'tf_fusion_alpha': 0.25,
            'tf_fusion_disagreement_power': 1.0,
            'use_fdconv_scale': True,
            'mutual_feat_weight': 0.002,
            'mutual_logit_weight': 0.02,
            'uncertainty_rampup_epochs': 25,
            'use_ema': True,
            'ema_start': 5,
            'use_swa': False,
            'max_epoch': 80,
            'use_early_stopping': False,
            'early_stop_patience': 18,
            'lr_milestones': [30, 55],
            'final_eval_mode': 'ensemble',
            'selection_score_mode': 'fusion_auc',
            'fusion_mode': 'standard',
            'branch2_fusion_bias': 1.05,
            'fusion_residual_weight': 0.08,
            'fusion_logit_blend': 0.15,
            'run_tag': 'cpsc_softmax_ce_weighted_full',
            'run_note': 'cpsc_single_label_weighted_ce_fdconv_uiml_ema_fullrun',
        },
        'stable': {
            'use_frequency_aux': False,
            'frequency_aux_gamma_init': 0.015,
            'cpsc_use_stratified_split': True,
            'run_tag': 'v2',
            'run_note': 'uiml_freqaux_stable',
        },
        'stable_branch2': {
            'use_frequency_aux': False,
            'frequency_aux_gamma_init': 0.015,
            'cpsc_use_stratified_split': True,
            'selection_score_mode': 'branch2_auc',
            'run_tag': 'cpsc_stable_branch2',
            'run_note': 'cpsc_stable_select_branch2_auc',
        },
        'stable_single_auc': {
            'use_frequency_aux': False,
            'frequency_aux_gamma_init': 0.015,
            'cpsc_use_stratified_split': True,
            'selection_score_mode': 'single_auc',
            'run_tag': 'cpsc_stable_single_auc',
            'run_note': 'cpsc_stable_select_single_auc',
        },
        'stable_bce': {
            'loss_name': 'bce',
            'lr': 0.0005,
            'gradient_clip_norm': 5.0,
            'use_frequency_aux': False,
            'frequency_aux_gamma_init': 0.015,
            'cpsc_use_stratified_split': True,
            'selection_score_mode': 'fusion_auc_f1',
            'run_tag': 'cpsc_stable_bce',
            'run_note': 'cpsc_stable_uiml_fdconv_bce',
        },
        'stable_bce_single_auc': {
            'loss_name': 'bce',
            'lr': 0.0005,
            'gradient_clip_norm': 5.0,
            'use_frequency_aux': False,
            'frequency_aux_gamma_init': 0.015,
            'cpsc_use_stratified_split': True,
            'selection_score_mode': 'single_auc',
            'run_tag': 'cpsc_stable_bce_single_auc',
            'run_note': 'cpsc_stable_bce_select_single_auc',
        },
        'ptb_exp0': {
            'use_frequency_aux': True,
            'frequency_aux_gamma_init': 0.015,
            'use_multiband_frequency_aux': False,
            'use_classwise_frequency_aux': False,
            'use_tf_branch': False,
            'use_periodic_aux_branch': False,
            'use_category_fusion': False,
            'run_tag': 'cpsc_ptbexp0',
            'run_note': 'cpsc_try_ptb_exp0_recipe',
        },
        'mrm_like': {
            'mutual_feat_weight': 0.0,
            'mutual_logit_weight': 0.0,
            'use_fdconv_scale': False,
            'use_frequency_aux': False,
            'use_multiband_frequency_aux': False,
            'use_classwise_frequency_aux': False,
            'use_tf_branch': False,
            'use_periodic_aux_branch': False,
            'use_category_fusion': False,
            'use_ema': False,
            'use_swa': False,
            'final_eval_mode': 'checkpoint',
            'selection_score_mode': 'fusion_auc_f1',
            'run_tag': 'cpsc_mrm_like',
            'run_note': 'cpsc_conservative_mrm_like',
        },
        'uiml_only': {
            'mutual_feat_weight': 0.005,
            'mutual_logit_weight': 0.05,
            'use_fdconv_scale': False,
            'use_frequency_aux': False,
            'use_multiband_frequency_aux': False,
            'use_classwise_frequency_aux': False,
            'use_tf_branch': False,
            'use_periodic_aux_branch': False,
            'use_category_fusion': False,
            'use_ema': True,
            'use_swa': True,
            'final_eval_mode': 'checkpoint',
            'selection_score_mode': 'fusion_auc_f1',
            'run_tag': 'cpsc_uiml_only',
            'run_note': 'cpsc_uiml_without_fdconvscale',
        },
        'fdconv_only': {
            'mutual_feat_weight': 0.0,
            'mutual_logit_weight': 0.0,
            'use_fdconv_scale': True,
            'use_frequency_aux': False,
            'use_multiband_frequency_aux': False,
            'use_classwise_frequency_aux': False,
            'use_tf_branch': False,
            'use_periodic_aux_branch': False,
            'use_category_fusion': False,
            'use_ema': True,
            'use_swa': True,
            'final_eval_mode': 'checkpoint',
            'selection_score_mode': 'fusion_auc_f1',
            'run_tag': 'cpsc_fdconv_only',
            'run_note': 'cpsc_fdconvscale_without_uiml',
        },
    }

    # Comparison baselines are single-output models. They should not inherit
    # MRM-specific auxiliary losses/gates, otherwise the comparison is muddy.
    comparison_model_overrides = {
        'mutual_feat_weight': 0.0,
        'mutual_logit_weight': 0.0,
        'use_frequency_aux': False,
        'use_multiband_frequency_aux': False,
        'use_classwise_frequency_aux': False,
        'use_tf_branch': False,
        'use_periodic_aux_branch': False,
        'use_category_fusion': False,
        'use_lead_relation_graph': False,
        'use_lead_calibration': False,
        'use_lcd_regularization': False,
        'use_frequency_gate': False,
        'use_adaptive_temporal_window': False,
        'use_multirange_reweight': False,
        'use_fda_augmentation': False,
        'use_selective_mutual_learning': False,
        'use_auc_ranking_loss': False,
        'use_branch_complement_loss': False,
        'use_fdconv_scale': False,
        'use_fusion_f1_boost': False,
        'fusion_mode': 'standard',
        'use_ema': False,
        'use_swa': False,
        'run_note': 'comparison_baseline',
        'selection_score_mode': 'fusion_auc',
        'baseline_collect_train_metrics': False,
        'use_early_stopping': False,
        'max_epoch': 100,
        'num_epochs': 100,
        'lr_milestones': [50, 80],
        'final_eval_mode': 'checkpoint',
    }

    # The original MRM baseline is not a plain CNN baseline, but it also should
    # not inherit UFML-Net private modules such as FreqAux, FDConvScale or
    # EMA/SWA. Keep it as a clean MRM-style reference.
    mrm_baseline_overrides = {
        'mutual_feat_weight': 0.0,
        'mutual_logit_weight': 0.0,
        'use_frequency_aux': False,
        'use_multiband_frequency_aux': False,
        'use_classwise_frequency_aux': False,
        'use_tf_branch': False,
        'use_periodic_aux_branch': False,
        'use_category_fusion': False,
        'use_lcd_regularization': False,
        'use_fdconv_scale': False,
        'use_ema': False,
        'use_swa': False,
        'final_eval_mode': 'checkpoint',
        'run_note': 'mrm_baseline_clean',
    }

    # PTB-XL exp0 is pinned to the strongest stable 94.87-style reference run:
    # UIML + frequency auxiliary compensation + FDConvScale + EMA/SWA.
    # Experimental modules that reduced AUC are kept off here.
    experiment_overrides = {
        'exp0': {
            '_datasets': ['ptbxl'],
            'run_tag': 'freqaux_ref',
            'selection_score_mode': 'fusion_auc_f1',
            'weight_decay': 0.0,
            'lr_milestones': [50, 100, 120],
            'use_lead_relation_graph': False,
            'lead_relation_graph_dim': 16,
            'lead_relation_graph_gamma_init': 0.003,
            'use_lcd_regularization': False,
            'lcd_weight': 0.001,
            'lcd_gamma': 0.005,
            'lcd_warmup_epochs': 10,
            'lcd_projection_dim': 64,
            'lcd_crop_ratio_low': 0.85,
            'lcd_mask_ratio_up': 0.10,
            'use_frequency_gate': False,
            'use_adaptive_temporal_window': False,
            'atwa_gamma_init': 0.003,
            'use_multirange_reweight': False,
            'multirange_gamma_init': 0.002,
            'use_frequency_aux': True,
            'frequency_aux_gamma_init': 0.015,
            'use_multiband_frequency_aux': False,
            'multiband_frequency_splits': (0.18, 0.42, 1.0),
            'use_classwise_frequency_aux': False,
            'use_tf_branch': False,
            'use_periodic_aux_branch': False,
            'periodic_top_k': 3,
            'periodic_min_period': 8,
            'periodic_max_period': 128,
            'use_category_fusion': False,
            'use_branch_complement_loss': False,
            'branch_complement_weight': 0.0003,
            'branch_complement_rampup_epochs': 20,
            'branch2_fusion_bias': 1.08,
            'fusion_residual_weight': 0.18,
            'fusion_positive_margin': 0.55,
            'use_fusion_f1_boost': False,
            'fusion_f1_boost_weight': 0.03,
            'fusion_f1_boost_margin': 0.48,
            'fusion_f1_boost_power': 2.0,
            'use_val_threshold_optimization': False,
            'report_val_threshold_metrics': False,
            'threshold_optimization_mode': 'global',
            'threshold_grid': [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60],
            'threshold_min_f1_gain': 0.003,
            'run_note': 'uiml_freqaux_fdconv_ema_swa_ref',
            'use_fda_augmentation': False,
            'use_selective_mutual_learning': False,
            'use_auc_ranking_loss': False,
        },
        # PTB-XL exp2 needs light structural help for form labels, but the
        # frequency auxiliary path is still kept off because it can disturb
        # morphology ranking.
        'exp2': {
            '_datasets': ['ptbxl'],
            'run_tag': 'exp2_fdconv_uiml_fda_shape',
            'selection_score_mode': 'fusion_auc',
            'weight_decay': 0.0,
            'use_frequency_aux': False,
            'use_multiband_frequency_aux': False,
            'use_classwise_frequency_aux': False,
            'use_tf_branch': False,
            'use_periodic_aux_branch': False,
            'use_category_fusion': False,
            'use_shape_enhance': True,
            'shape_gamma_init': 0.005,
            'use_lead_relation_graph': False,
            'use_lcd_regularization': False,
            'use_lcdr_pretrained': False,
            'use_frequency_gate': False,
            'use_adaptive_temporal_window': False,
            'use_multirange_reweight': False,
            'use_fda_augmentation': True,
            'fda_prob': 0.25,
            'fda_strength': 0.04,
            'use_selective_mutual_learning': False,
            'use_auc_ranking_loss': False,
            'use_branch_complement_loss': False,
            'use_fusion_f1_boost': False,
            'use_fdconv_scale': True,
            'mutual_feat_weight': 0.002,
            'mutual_logit_weight': 0.02,
            'fusion_mode': 'standard',
            'branch2_fusion_bias': 1.0,
            'fusion_residual_weight': 0.05,
            'fusion_logit_blend': 0.1,
            'use_ema': True,
            'ema_start': 5,
            'use_swa': False,
            'final_eval_mode': 'ensemble',
            'max_epoch': 60,
            'early_stop_patience': 15,
            'lr_milestones': [30, 45],
            'run_note': 'exp2_light_fdconv_uiml_fda_shape_no_freqaux',
        },
        # Rhythm labels benefit from temporal branch2, and may gain extra ranking
        # signal from a light periodic auxiliary branch. Keep frequency residuals
        # off here so the periodic cue is tested cleanly.
        'exp3': {
            '_datasets': ['ptbxl'],
            'run_tag': 'exp3_periodic_light',
            'selection_score_mode': 'fusion_auc',
            'use_frequency_aux': False,
            'frequency_aux_gamma_init': 0.008,
            'use_multiband_frequency_aux': False,
            'use_classwise_frequency_aux': False,
            'use_tf_branch': False,
            'use_periodic_aux_branch': True,
            'periodic_top_k': 3,
            'periodic_min_period': 8,
            'periodic_max_period': 128,
            'tf_loss_weight': 0.08,
            'tf_fusion_alpha': 0.12,
            'tf_fusion_disagreement_power': 0.75,
            'use_category_fusion': False,
            'category_fusion_loss_weight': 0.1,
            'category_fusion_alpha': 0.15,
            'use_lead_relation_graph': False,
            'use_lcd_regularization': False,
            'use_lcdr_pretrained': False,
            'use_adaptive_temporal_window': False,
            'use_multirange_reweight': False,
            'use_fda_augmentation': False,
            'use_selective_mutual_learning': False,
            'use_auc_ranking_loss': False,
            'use_branch_complement_loss': False,
            'use_fdconv_scale': True,
            'mutual_feat_weight': 0.002,
            'mutual_logit_weight': 0.02,
            'branch2_fusion_bias': 1.05,
            'fusion_residual_weight': 0.05,
            'fusion_logit_blend': 0.1,
            'use_fusion_f1_boost': False,
            'fusion_mode': 'standard',
            'max_epoch': 60,
            'early_stop_patience': 15,
            'lr_milestones': [30, 45],
            'use_ema': True,
            'ema_start': 5,
            'use_swa': False,
            'run_note': 'exp3_periodic_aux_light_no_freqaux',
        }
    }

config = Config()
