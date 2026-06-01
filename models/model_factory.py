# -*- coding: utf-8 -*-

"""Backward-compatible model factory.

New code should import ``build_model`` from ``models.registry`` directly. This
module is kept so older scripts that import ``models.model_factory`` continue to
work.
"""

from .registry import build_model as _build_model


def build_model(config, num_classes, input_channels=12):
    return _build_model(getattr(config, 'model_name', 'ufml_net'), num_classes, config)


def is_supported_model(model_name):
    try:
        _ = str(model_name).lower()
        return True
    except Exception:
        return False
