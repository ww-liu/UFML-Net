# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import config
from data_load_georgia import load_data as load_georgia_data

try:
    from data_process import (
        load_dataset,
        compute_label_aggregations,
        select_data,
        preprocess_signals,
        data_slice,
        hf_dataset,
    )
    _DATA_PROCESS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _DATA_PROCESS_IMPORT_ERROR = exc
    load_dataset = None
    compute_label_aggregations = None
    select_data = None
    preprocess_signals = None
    data_slice = None
    hf_dataset = None

try:
    from data_load import load_data as load_cpsc_data
    _CPSC_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _CPSC_IMPORT_ERROR = exc
    load_cpsc_data = None


def _require_data_process():
    if _DATA_PROCESS_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            'PTB-XL/HF dataset loading requires data_process.py dependencies. '
            f'Original error: {_DATA_PROCESS_IMPORT_ERROR}'
        ) from _DATA_PROCESS_IMPORT_ERROR


def _require_cpsc_loader():
    if _CPSC_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            'CPSC dataset loading requires data_load.py dependencies. '
            f'Original error: {_CPSC_IMPORT_ERROR}'
        ) from _CPSC_IMPORT_ERROR


def detect_dataset_type(datafolder, dataset=None):
    requested_dataset = str(
        dataset or
        getattr(config, 'dataset', '') or
        getattr(config, 'dataset_name', '')
    ).lower()
    if requested_dataset in ('georgia', 'ga'):
        return 'georgia'
    if requested_dataset in ('ptbxl', 'ptb-xl'):
        return 'ptbxl'
    if requested_dataset == 'cpsc':
        return 'cpsc'

    path = os.path.abspath(datafolder).lower()
    if (
        os.path.basename(path) == 'georgia' or
        os.path.exists(os.path.join(datafolder, 'training', 'georgia')) or
        os.path.exists(os.path.join(datafolder, 'georgia')) or
        os.path.exists(os.path.join(datafolder, 'g1'))
    ):
        return 'georgia'
    if os.path.exists(os.path.join(datafolder, 'raw', 'ptbxl_database.csv')) or 'ptb' in path:
        return 'ptbxl'
    if (
        os.path.exists(os.path.join(datafolder, 'cpsc_database.csv')) or
        os.path.exists(os.path.join(datafolder, 'data2.npy')) or
        (
            os.path.exists(os.path.join(datafolder, 'data2.npy')) and
            os.path.exists(os.path.join(datafolder, 'labels2.npy'))
        ) or
        os.path.exists(os.path.join(datafolder, 'TrainingSet1')) or
        'cpsc' in path or
        os.path.basename(path) == 'tmp_data'
    ):
        return 'cpsc'
    return 'hf'


class ECGDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, is_train: bool = False):
        super(ECGDataset, self).__init__()
        self.data = signals
        self.label = labels
        self.num_classes = self.label.shape[1]
        self.cls_num_list = np.sum(self.label, axis=0)
        self.is_train = is_train

    @staticmethod
    def _frequency_dynamic_augmentation(x: torch.Tensor) -> torch.Tensor:
        # x: [time, leads]
        x_lead_first = x.transpose(0, 1)
        spectrum = torch.fft.rfft(x_lead_first, dim=-1)
        amplitude = spectrum.abs()
        phase = torch.angle(spectrum)

        max_amplitude = amplitude.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        importance = (amplitude / max_amplitude).clamp(0.0, 1.0)
        strength = getattr(config, 'fda_strength', 0.08)
        power = getattr(config, 'fda_importance_power', 1.5)
        noise_scale = strength * torch.pow(1.0 - importance, power)
        perturb = 1.0 + torch.randn_like(amplitude) * noise_scale
        augmented_amplitude = (amplitude * perturb).clamp_min(0.0)

        augmented_spectrum = torch.polar(augmented_amplitude, phase)
        augmented = torch.fft.irfft(
            augmented_spectrum,
            n=x_lead_first.size(-1),
            dim=-1
        )
        return augmented.transpose(0, 1).to(dtype=x.dtype)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        x = torch.tensor(x, dtype=torch.float32)
        if self.is_train and getattr(config, 'use_fda_augmentation', False):
            if torch.rand(1).item() < getattr(config, 'fda_prob', 0.5):
                x = self._frequency_dynamic_augmentation(x)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.data)


class DownLoadECGData:
    def __init__(self, experiment_name, task, datafolder, sampling_frequency=100, min_samples=0,
                 train_fold=8, val_fold=9, test_fold=10):
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.experiment_name = experiment_name
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency

    def preprocess_data(self):
        _require_data_process()
        data, raw_labels = load_dataset(self.datafolder, self.sampling_frequency)
        labels = compute_label_aggregations(raw_labels, self.datafolder, self.task)
        data, labels, y, _ = select_data(data, labels, self.task, self.min_samples)

        if detect_dataset_type(self.datafolder) == 'cpsc':
            data = data_slice(data)

        x_test = data[labels.strat_fold == self.test_fold]
        y_test = y[labels.strat_fold == self.test_fold]
        x_val = data[labels.strat_fold == self.val_fold]
        y_val = y[labels.strat_fold == self.val_fold]
        x_train = data[labels.strat_fold <= self.train_fold]
        y_train = y[labels.strat_fold <= self.train_fold]

        x_train, x_val, x_test = preprocess_signals(x_train, x_val, x_test)
        return x_train, y_train, x_val, y_val, x_test, y_test


def _build_loaders_from_arrays(x_train, y_train, x_val, y_val, x_test, y_test):
    ds_train = ECGDataset(x_train, y_train, is_train=True)
    ds_val = ECGDataset(x_val, y_val, is_train=False)
    ds_test = ECGDataset(x_test, y_test, is_train=False)

    train_dataloader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader, ds_train.num_classes


def load_datasets(datafolder=None, experiment=None):
    if datafolder is None:
        raise ValueError('datafolder must be provided')

    dataset_type = detect_dataset_type(datafolder)

    if dataset_type == 'ptbxl':
        _require_data_process()
        experiments = {
            'exp0': ('exp0', 'all'),
            'exp1': ('exp1', 'diagnostic'),
            'exp1.1': ('exp1.1', 'subdiagnostic'),
            'exp1.1.1': ('exp1.1.1', 'superdiagnostic'),
            'exp2': ('exp2', 'form'),
            'exp3': ('exp3', 'rhythm'),
        }
        if experiment not in experiments:
            raise ValueError(f'Unsupported PTB-XL experiment: {experiment}')
        name, task = experiments[experiment]
        ded = DownLoadECGData(
            name,
            task,
            datafolder,
            sampling_frequency=getattr(config, 'sampling_frequency', 100)
        )
        x_train, y_train, x_val, y_val, x_test, y_test = ded.preprocess_data()
        return _build_loaders_from_arrays(x_train, y_train, x_val, y_val, x_test, y_test)

    if dataset_type == 'cpsc':
        _require_cpsc_loader()
        train_loader, val_loader, test_loader = load_cpsc_data(config.batch_size, datafolder=datafolder)
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, test_loader, num_classes

    if dataset_type == 'georgia':
        x_train, y_train, x_val, y_val, x_test, y_test = load_georgia_data(
            datafolder=datafolder,
            min_label_count=getattr(config, 'georgia_min_label_count', 100),
            seed=getattr(config, 'seed', 1),
        )
        return _build_loaders_from_arrays(x_train, y_train, x_val, y_val, x_test, y_test)

    _require_data_process()
    x_train, y_train, x_val, y_val, x_test, y_test = hf_dataset(datafolder)
    return _build_loaders_from_arrays(x_train, y_train, x_val, y_val, x_test, y_test)
