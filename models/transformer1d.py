# -*- coding: utf-8 -*-

from torch import nn

from .common import ConvBNReLU, ensure_channels_first


class CNNTransformer1D(nn.Module):
    is_single_branch_model = True

    def __init__(self, in_channels=12, num_classes=5, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, hidden_dim, 7, stride=2),
            ConvBNReLU(hidden_dim, hidden_dim, 5, stride=2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = ensure_channels_first(x, self.in_channels)
        feat_map = self.stem(x)
        sequence = feat_map.permute(0, 2, 1)
        encoded = self.encoder(sequence)
        feat = self.pool(encoded.permute(0, 2, 1)).squeeze(-1)
        return self.classifier(feat)
