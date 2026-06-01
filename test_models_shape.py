# -*- coding: utf-8 -*-

import torch

from config import config
from models.common import count_parameters, extract_logits, is_baseline_model
from models.registry import build_model


def print_output_shape(model_name, model, outputs):
    num_params = count_parameters(model)
    if torch.is_tensor(outputs):
        print(f'{model_name}: output_type=Tensor, output_shape={tuple(outputs.shape)}, num_params={num_params}')
        return
    if isinstance(outputs, (tuple, list)):
        if is_baseline_model(model_name):
            raise AssertionError(f'{model_name} is a baseline and must return a Tensor, not {type(outputs)}')
        shapes = [
            tuple(output.shape)
            for output in outputs
            if torch.is_tensor(output)
        ]
        print(f'{model_name}: output_type={type(outputs).__name__}, output_shapes={shapes}, logits={tuple(extract_logits(outputs).shape)}, num_params={num_params}')
        return
    print(f'{model_name}: unsupported output type {type(outputs)}')


def main():
    model_names = [
        'cnn1d',
        'resnet1d',
        'inceptiontime1d',
        'bilstm',
        'cnn_bilstm',
        'cnn_transformer',
        'ufml_net',
        'mrm_baseline',
        'dbfnet',
        'ecg_gtmd',
        'bmirc',
        'lcd',
    ]
    x = torch.randn(2, 12, 1000)
    num_classes = 5
    for model_name in model_names:
        try:
            model = build_model(model_name, num_classes, config)
            model.eval()
            with torch.no_grad():
                outputs = model(x)
            if is_baseline_model(model_name):
                assert torch.is_tensor(outputs), f'{model_name} must return a Tensor.'
                assert tuple(outputs.shape) == (2, num_classes), f'{model_name} output shape mismatch: {tuple(outputs.shape)}'
            else:
                assert tuple(extract_logits(outputs).shape) == (2, num_classes), f'{model_name} logits shape mismatch.'
            print_output_shape(model_name, model, outputs)
        except NotImplementedError as exc:
            print(f'SKIP: {model_name} is not implemented yet. {exc}')


if __name__ == '__main__':
    main()
