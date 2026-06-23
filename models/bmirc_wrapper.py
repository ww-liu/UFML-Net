# -*- coding: utf-8 -*-

from torch import nn


class BMIRCWrapper(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        raise NotImplementedError(
            'BMIRC requires self-supervised pretraining and fine-tuning. '
            'Run official BMIRC as an external baseline first.'
        )
