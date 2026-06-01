# -*- coding: utf-8 -*-

import torch
from torch import nn


def ensure_channels_first(x, in_channels=12):
    """Accept ECG tensors as [B, 12, L] or [B, L, 12], return [B, 12, L]."""
    if x.dim() != 3:
        raise ValueError(f'Expected a 3D ECG tensor, got shape {tuple(x.shape)}')
    if x.size(1) == in_channels:
        return x
    if x.size(2) == in_channels:
        return x.permute(0, 2, 1)
    raise ValueError(f'Cannot infer ECG channel dimension from shape {tuple(x.shape)}')


def extract_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        tensors = [o for o in outputs if torch.is_tensor(o)]
        if len(tensors) == 0:
            raise ValueError('Model output tuple/list contains no Tensor.')
        return tensors[-1]
    raise TypeError(f'Unsupported model output type: {type(outputs)}')


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


BASELINE_MODELS = {
    'cnn1d',
    'resnet1d',
    'inceptiontime1d',
    'bilstm',
    'cnn_bilstm',
    'cnn_transformer',
    # Backward-compatible aliases used by earlier comparison runs.
    'fcn_wang',
    'resnet1d_wang',
    'xresnet1d101',
    'inceptiontime',
    'lstm',
    'lstm_bidir',
    'cnn_lstm',
}

UFML_MODELS = {
    'ufml_net',
    'ufml_no_uncertainty',
    'ufml_no_frequency',
    'ufml_no_fdconv',
    'ufml_no_dswin',
    'ufml_no_mutual',
    'ufml_no_ema_swa',
    'ufml_fdconv_only',
    'ufml_dswin_only',
    'ufml_frequency_only',
    'ufml_fdconv_dswin',
}

EXTERNAL_MODELS = {
    'dbfnet',
    'ecg_gtmd',
    'bmirc',
    'lcd',
}


def normalize_model_name(model_name):
    return str(model_name or '').lower()


def is_baseline_model(model_name):
    return normalize_model_name(model_name) in BASELINE_MODELS


def is_ufml_model(model_name):
    return normalize_model_name(model_name) in UFML_MODELS


def is_external_model(model_name):
    return normalize_model_name(model_name) in EXTERNAL_MODELS


def is_mrm_baseline_model(model_name):
    return normalize_model_name(model_name) == 'mrm_baseline'


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
