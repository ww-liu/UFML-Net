# -*- coding: utf-8 -*-

from torch import nn

from .common import ConvBNReLU, ensure_channels_first


class BiLSTM(nn.Module):
    is_single_branch_model = True

    def __init__(self, input_size=12, num_classes=5, hidden_size=128, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = ensure_channels_first(x, self.input_size).permute(0, 2, 1)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


class CNNBiLSTM(nn.Module):
    is_single_branch_model = True

    def __init__(self, in_channels=12, num_classes=5, cnn_channels=128, hidden_size=128, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.cnn = nn.Sequential(
            ConvBNReLU(in_channels, 64, 7),
            nn.MaxPool1d(2),
            ConvBNReLU(64, cnn_channels, 5),
            nn.MaxPool1d(2),
            ConvBNReLU(cnn_channels, cnn_channels, 3),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = ensure_channels_first(x, self.in_channels)
        feat_map = self.cnn(x)
        sequence = feat_map.permute(0, 2, 1)
        output, _ = self.lstm(sequence)
        return self.classifier(output[:, -1, :])
