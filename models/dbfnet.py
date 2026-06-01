# -*- coding: utf-8 -*-

import torch
from torch import nn

from .common import ConvBNReLU, ensure_channels_first


class _DBFBranch(nn.Module):
    def __init__(self, in_channels, channels, kernels):
        super().__init__()
        layers = []
        current = in_channels
        for out_channels, kernel in zip(channels, kernels):
            layers.extend([ConvBNReLU(current, out_channels, kernel), nn.MaxPool1d(2)])
            current = out_channels
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        feat_map = self.encoder(x)
        return self.pool(feat_map).squeeze(-1)


class DBFNet(nn.Module):
    """External DBFNet placeholder.

    The previous lightweight in-project approximation has been disabled because
    it was not an official reproduction of the 2025 DBFNet architecture. Keeping
    it runnable would make the comparison ambiguous.
    """

    is_multi_branch_model = True

    def __init__(self, num_classes=5, config=None, in_channels=12):
        super().__init__()
        raise NotImplementedError(
            'DBFNet should be run from the official paper/code or integrated '
            'with its original dual-branch mutual-learning recipe. Do not use '
            'UFML-Net modules as a fake DBFNet baseline.'
        )
        self.in_channels = in_channels
        self.branch_short = _DBFBranch(in_channels, (64, 128, 128), (7, 5, 3))
        self.branch_long = _DBFBranch(in_channels, (64, 128, 128), (31, 15, 7))
        self.fc_short = nn.Linear(128, num_classes)
        self.fc_long = nn.Linear(128, num_classes)
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = ensure_channels_first(x, self.in_channels)
        feat_short = self.branch_short(x)
        feat_long = self.branch_long(x)
        logits_short = self.fc_short(feat_short)
        logits_long = self.fc_long(feat_long)
        # The project training loop applies its own uncertainty-guided fusion to
        # branch logits, so the learned fusion head is intentionally not returned
        # here. It remains available for future DBFNet-specific training.
        _ = self.fusion(torch.cat([feat_short, feat_long], dim=1))
        return feat_short, logits_short, feat_long, logits_long
