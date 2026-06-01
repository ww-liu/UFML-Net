# -*- coding: utf-8 -*-

from torch import nn


class LCDWrapper(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        raise NotImplementedError(
            'LCD requires self-supervised pretraining/fine-tuning. '
            'Run official ECG_SSL_LCD as an external baseline first.'
        )
