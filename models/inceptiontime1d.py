# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn

from .common import ensure_channels_first


class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32):
        super().__init__()
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            branch_in = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            branch_in = in_channels
        branch_channels = out_channels // 4
        self.conv_k9 = nn.Conv1d(branch_in, branch_channels, kernel_size=9, padding=4, bias=False)
        self.conv_k19 = nn.Conv1d(branch_in, branch_channels, kernel_size=19, padding=9, bias=False)
        self.conv_k39 = nn.Conv1d(branch_in, branch_channels, kernel_size=39, padding=19, bias=False)
        self.pool_proj = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(branch_channels * 4)

    def forward(self, x):
        z = self.bottleneck(x)
        out = torch.cat([self.conv_k9(z), self.conv_k19(z), self.conv_k39(z), self.pool_proj(x)], dim=1)
        return F.relu(self.bn(out), inplace=True)


class InceptionTime1D(nn.Module):
    is_single_branch_model = True

    def __init__(self, in_channels=12, num_classes=5, depth=6, hidden_channels=128):
        super().__init__()
        self.in_channels = in_channels
        modules = []
        current_channels = in_channels
        for _ in range(depth):
            modules.append(InceptionModule1D(current_channels, hidden_channels))
            current_channels = hidden_channels
        self.encoder = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = ensure_channels_first(x, self.in_channels)
        feat_map = self.encoder(x)
        feat = self.pool(feat_map).squeeze(-1)
        return self.classifier(feat)
