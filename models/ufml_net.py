# -*- coding: utf-8 -*-

from torch import nn

from .duibi import Dui1
from .common import ensure_channels_first


class UFMLNet(nn.Module):
    """UFML-Net wrapper.

    The current best model implementation is preserved in ``models/duibi.py`` as
    ``Dui1``. This wrapper gives the final paper model a stable name and keeps
    the old file intact for compatibility.
    """

    is_multi_branch_model = True

    def __init__(self, num_classes=5, config=None, input_channels=12):
        super().__init__()
        self.input_channels = input_channels
        self.backbone = Dui1(
            num_classes=num_classes,
            input_channels=input_channels,
            use_shape_enhance=getattr(config, 'use_shape_enhance', False),
            shape_gamma_init=getattr(config, 'shape_gamma_init', 0.01),
            use_fdconv_scale=getattr(config, 'use_fdconv_scale', True),
            use_lead_calibration=getattr(config, 'use_lead_calibration', False),
            lead_gamma_init=getattr(config, 'lead_gamma_init', 0.005),
            use_lead_relation_graph=getattr(config, 'use_lead_relation_graph', False),
            lead_relation_graph_dim=getattr(config, 'lead_relation_graph_dim', 32),
            lead_relation_graph_gamma_init=getattr(config, 'lead_relation_graph_gamma_init', 0.0015),
            use_frequency_gate=getattr(config, 'use_frequency_gate', False),
            frequency_gamma_init=getattr(config, 'frequency_gamma_init', 0.02),
            use_adaptive_temporal_window=getattr(config, 'use_adaptive_temporal_window', False),
            atwa_gamma_init=getattr(config, 'atwa_gamma_init', 0.003),
            use_multirange_reweight=getattr(config, 'use_multirange_reweight', False),
            multirange_gamma_init=getattr(config, 'multirange_gamma_init', 0.002),
            use_frequency_aux=getattr(config, 'use_frequency_aux', True),
            frequency_aux_gamma_init=getattr(config, 'frequency_aux_gamma_init', 0.015),
            use_multiband_frequency_aux=getattr(config, 'use_multiband_frequency_aux', False),
            multiband_frequency_splits=getattr(config, 'multiband_frequency_splits', (0.18, 0.42, 1.0)),
            frequency_aux_uncertainty_power=getattr(config, 'frequency_aux_uncertainty_power', 1.5),
            frequency_aux_disagreement_power=getattr(config, 'frequency_aux_disagreement_power', 1.0),
            use_lcd_regularization=getattr(config, 'use_lcd_regularization', False),
            lcd_projection_dim=getattr(config, 'lcd_projection_dim', 64),
            lcd_gamma=getattr(config, 'lcd_gamma', 0.005),
            lcd_crop_ratio_low=getattr(config, 'lcd_crop_ratio_low', 0.85),
            lcd_mask_ratio_up=getattr(config, 'lcd_mask_ratio_up', 0.10),
            use_tf_branch=getattr(config, 'use_tf_branch', False),
            tf_n_fft=getattr(config, 'tf_n_fft', 64),
            tf_hop_length=getattr(config, 'tf_hop_length', 16),
            tf_win_length=getattr(config, 'tf_win_length', 64),
            use_periodic_aux_branch=getattr(config, 'use_periodic_aux_branch', False),
            periodic_top_k=getattr(config, 'periodic_top_k', 3),
            periodic_min_period=getattr(config, 'periodic_min_period', 8),
            periodic_max_period=getattr(config, 'periodic_max_period', 128),
            use_category_fusion=getattr(config, 'use_category_fusion', False),
        )

    def forward(self, x):
        x = ensure_channels_first(x, self.input_channels).permute(0, 2, 1)
        return self.backbone(x)

    def compute_lcd_loss(self, x):
        x = ensure_channels_first(x, self.input_channels).permute(0, 2, 1)
        return self.backbone.compute_lcd_loss(x)
