# -*- coding: utf-8 -*-

from copy import copy
from torch import nn

from .ufml_net import UFMLNet


class MRMBaseline(nn.Module):
    """Conservative wrapper for the original MRM-style backbone.

    The implementation reuses ``Dui1`` through ``UFMLNet`` but disables the
    paper-specific UFML additions such as frequency auxiliary compensation,
    FDConvScale, LCD regularization, and other experimental gates.
    """

    is_multi_branch_model = True

    def __init__(self, num_classes=5, config=None):
        super().__init__()
        baseline_config = copy(config)
        baseline_config.use_shape_enhance = False
        baseline_config.use_fdconv_scale = False
        baseline_config.use_lead_calibration = False
        baseline_config.use_lead_relation_graph = False
        baseline_config.use_frequency_gate = False
        baseline_config.use_adaptive_temporal_window = False
        baseline_config.use_multirange_reweight = False
        baseline_config.use_frequency_aux = False
        baseline_config.use_multiband_frequency_aux = False
        baseline_config.use_lcd_regularization = False
        baseline_config.use_tf_branch = False
        baseline_config.use_periodic_aux_branch = False
        baseline_config.use_category_fusion = False
        self.model = UFMLNet(num_classes=num_classes, config=baseline_config)

    def forward(self, x):
        return self.model(x)

    def compute_lcd_loss(self, x):
        return self.model.compute_lcd_loss(x)
