# -*- coding: utf-8 -*-


from .common import is_baseline_model, is_external_model, is_mrm_baseline_model, is_ufml_model


def build_model(model_name, num_classes, config):
    model_name = str(model_name).lower()

    if model_name == 'cnn1d':
        from .cnn1d import CNN1D
        return CNN1D(in_channels=12, num_classes=num_classes)

    if model_name == 'resnet1d':
        from .resnet1d import ResNet1D
        return ResNet1D(in_channels=12, num_classes=num_classes)

    if model_name == 'inceptiontime1d':
        from .inceptiontime1d import InceptionTime1D
        return InceptionTime1D(in_channels=12, num_classes=num_classes)

    if model_name == 'bilstm':
        from .bilstm import BiLSTM
        return BiLSTM(input_size=12, num_classes=num_classes)

    if model_name == 'cnn_bilstm':
        from .bilstm import CNNBiLSTM
        return CNNBiLSTM(in_channels=12, num_classes=num_classes)

    if model_name == 'cnn_transformer':
        from .transformer1d import CNNTransformer1D
        return CNNTransformer1D(in_channels=12, num_classes=num_classes)

    if model_name == 'mrm_baseline':
        from .mrm_baseline import MRMBaseline
        return MRMBaseline(num_classes=num_classes, config=config)

    if model_name in ('ufml_net', 'mynet6view'):
        from .ufml_net import UFMLNet
        return UFMLNet(num_classes=num_classes, config=config)

    if model_name.startswith('ufml_'):
        from .mrm_ablation import build_ufml_ablation
        return build_ufml_ablation(model_name, num_classes, config)

    if model_name == 'dbfnet':
        from .dbfnet import DBFNet
        return DBFNet(num_classes=num_classes, config=config)

    if model_name == 'ecg_gtmd':
        from .ecg_gtmd_wrapper import ECGGTMDWrapper
        return ECGGTMDWrapper(num_classes=num_classes, config=config)

    if model_name == 'bmirc':
        from .bmirc_wrapper import BMIRCWrapper
        return BMIRCWrapper(num_classes=num_classes, config=config)

    if model_name == 'lcd':
        from .lcd_wrapper import LCDWrapper
        return LCDWrapper(num_classes=num_classes, config=config)

    # Backward-compatible names used by previous experiments.
    if model_name == 'fcn_wang':
        from .cnn1d import CNN1D
        return CNN1D(in_channels=12, num_classes=num_classes)
    if model_name in ('resnet1d_wang', 'xresnet1d101'):
        from .resnet1d import ResNet1D
        if model_name == 'xresnet1d101':
            return ResNet1D(in_channels=12, num_classes=num_classes, channels=(64, 128, 256, 512), blocks_per_stage=(3, 4, 23, 3))
        return ResNet1D(in_channels=12, num_classes=num_classes)
    if model_name == 'inceptiontime':
        from .inceptiontime1d import InceptionTime1D
        return InceptionTime1D(in_channels=12, num_classes=num_classes)
    if model_name in ('lstm', 'lstm_bidir'):
        from .bilstm import BiLSTM
        return BiLSTM(input_size=12, num_classes=num_classes)
    if model_name == 'cnn_lstm':
        from .bilstm import CNNBiLSTM
        return CNNBiLSTM(in_channels=12, num_classes=num_classes)

    raise ValueError(f'Unknown model_name: {model_name}')


def is_multi_output_model(model_name):
    name = str(model_name).lower()
    return is_mrm_baseline_model(name) or is_ufml_model(name)
