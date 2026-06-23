# -*- coding: utf-8 -*-

from torch import nn


class ECGGTMDWrapper(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        raise NotImplementedError(
            'ECG-GTMD requires graph construction and multi-domain preprocessing. '
            'Run it as an external baseline first.'
        )
