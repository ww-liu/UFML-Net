# -*- coding: utf-8 -*-

import torch.nn.functional as F
from torch import nn

from .common import ConvBNReLU, ensure_channels_first


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + self.shortcut(x), inplace=True)


class ResNet1D(nn.Module):
    is_single_branch_model = True

    def __init__(self, in_channels=12, num_classes=5, channels=(64, 128, 128), blocks_per_stage=2):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * len(channels)
        layers = [ConvBNReLU(in_channels, channels[0], 7)]
        current_channels = channels[0]
        for stage_idx, out_channels in enumerate(channels):
            stride = 1 if stage_idx == 0 else 2
            layers.append(ResidualBlock1D(current_channels, out_channels, kernel_size=7, stride=stride))
            current_channels = out_channels
            for _ in range(blocks_per_stage[stage_idx] - 1):
                layers.append(ResidualBlock1D(current_channels, out_channels, kernel_size=7))
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        x = ensure_channels_first(x, self.in_channels)
        feat_map = self.encoder(x)
        feat = self.pool(feat_map).squeeze(-1)
        return self.classifier(feat)
