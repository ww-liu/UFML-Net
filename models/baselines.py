# -*- coding: utf-8 -*-

"""Backward-compatible clean baseline builders."""


def build_baseline_model(model_name, input_channels, num_classes):
    name = str(model_name).lower()
    if name == 'fcn_wang':
        from .cnn1d import CNN1D
        return CNN1D(in_channels=input_channels, num_classes=num_classes)
    if name == 'resnet1d_wang':
        from .resnet1d import ResNet1D
        return ResNet1D(in_channels=input_channels, num_classes=num_classes)
    if name == 'xresnet1d101':
        from .resnet1d import ResNet1D
        return ResNet1D(
            in_channels=input_channels,
            num_classes=num_classes,
            channels=(64, 128, 256, 512),
            blocks_per_stage=(3, 4, 23, 3),
        )
    if name == 'inceptiontime':
        from .inceptiontime1d import InceptionTime1D
        return InceptionTime1D(in_channels=input_channels, num_classes=num_classes)
    if name in ('lstm', 'lstm_bidir'):
        from .bilstm import BiLSTM
        return BiLSTM(input_size=input_channels, num_classes=num_classes)
    if name == 'cnn_lstm':
        from .bilstm import CNNBiLSTM
        return CNNBiLSTM(in_channels=input_channels, num_classes=num_classes)
    raise ValueError(f'Unsupported comparison model: {model_name}')
