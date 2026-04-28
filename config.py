# -*- coding: utf-8 -*-
'''
@time: 2021/4/16 18:45

@ author:
'''

class Config:

    seed = 7

    # path
    datafolder = '/root/autodl-tmp/data/ptbxl'
    # datafolder = '../data/CPSC/'
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
    MyNet6View, resnet1d_wang, xresnet1d101, inceptiontime, fcn_wang, lstm, lstm_bidir, vit, mobilenetv3_small
    '''
    model_name = 'MyNet6View'

    model_name2 = 'MyNet'

    batch_size = 64

    max_epoch = 100

    lr = 0.001
    lr_milestones = [70, 90]
    lr_gamma = 0.1
    sampling_frequency = 100
    dataset_name = 'auto'

    device_num = 1

    # eg: MyNet6View_all_checkpoint_best_tpr.pth
    checkpoints = 'MyNet6View_exp0_checkpoint_best_auc.pth'

    # knowledge distillation param
    alpha = 0.5
    temperature = 2

    # uncertainty-informed mutual learning
    mutual_temperature = 2.0
    feature_temperature = 2.0
    mutual_feat_weight = 0.005
    mutual_logit_weight = 0.05
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
    ema_decay = 0.999
    ema_start = 10
    swa_start = 45

    # Paper-inspired lightweight enhancements.
    # These modules are residual gates on top of the restored strong baseline.
    use_shape_enhance = False
    shape_gamma_init = 0.01
    use_lead_calibration = False
    lead_gamma_init = 0.005
    use_lead_relation_graph = False
    lead_relation_graph_dim = 16
    lead_relation_graph_gamma_init = 0.003
    use_frequency_gate = False
    frequency_gamma_init = 0.02
    use_frequency_aux = True
    frequency_aux_gamma_init = 0.015
    frequency_aux_uncertainty_power = 1.5
    frequency_aux_disagreement_power = 1.0
    use_tf_branch = False
    tf_n_fft = 64
    tf_hop_length = 16
    tf_win_length = 64
    tf_loss_weight = 0.2
    tf_fusion_alpha = 0.25
    tf_fusion_disagreement_power = 1.0
    use_category_fusion = True
    category_fusion_loss_weight = 0.1
    category_fusion_alpha = 0.15
    use_fda_augmentation = False
    fda_prob = 0.5
    fda_strength = 0.08
    fda_importance_power = 1.5

config = Config()
