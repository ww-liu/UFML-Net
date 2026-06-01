# -*- coding: utf-8 -*-

from torch import nn

from .common import ConvBNReLU, ensure_channels_first


class CNN1D(nn.Module):
    is_single_branch_model = True

    def __init__(self, in_channels=12, num_classes=5):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, 64, 7),
            nn.MaxPool1d(2),
            ConvBNReLU(64, 128, 5),
            nn.MaxPool1d(2),
            ConvBNReLU(128, 128, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = ensure_channels_first(x, self.in_channels)
        feat_map = self.encoder(x)
        feat = self.pool(feat_map).squeeze(-1)
        return self.classifier(feat)
