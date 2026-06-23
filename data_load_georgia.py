# -*- coding: utf-8 -*-

import csv
import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
from numpy.lib.format import open_memmap
from scipy.signal import resample, resample_poly
from sklearn.model_selection import train_test_split

from config import config


LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
TARGET_FS = 100
TARGET_LENGTH = 1000
CACHE_VERSION = 1


def resolve_georgia_root(datafolder):
    if datafolder is None:
        raise ValueError('Georgia datafolder must be provided.')

    root = Path(datafolder).expanduser()
    candidates = [
        root,
        root / 'georgia',
        root / 'training' / 'georgia',
    ]
    for candidate in candidates:
        if _looks_like_georgia_root(candidate):
            return candidate

    raise FileNotFoundError(
        'Could not locate the Georgia subset. Pass either the Challenge 2020 '
        'root, the training root, or the training/georgia directory.'
    )


def _looks_like_georgia_root(path):
    if not path.is_dir():
        return False
    if any((path / f'g{i}').is_dir() for i in range(1, 12)):
        return True
    return any(path.glob('*.hea')) and any(path.glob('*.mat'))


def get_cache_dir(min_label_count=None):
    min_label_count = int(min_label_count or getattr(config, 'georgia_min_label_count', 100))
    root = Path(getattr(config, 'georgia_processed_root', 'processed/georgia'))
    return root / f'min{min_label_count}'


def parse_hea(path):
    lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    if not lines:
        raise ValueError(f'Empty header file: {path}')
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f'Invalid header first line in {path}: {lines[0]}')

    record_id = first[0]
    n_leads = int(first[1])
    sampling_rate = float(first[2])
    signal_length = int(first[3])
    lead_names = []
    gains = []
    baselines = []
    comments = {}

    for line in lines[1:]:
        if not line.strip():
            continue
        if line.startswith('#'):
            body = line[1:].strip()
            if ':' in body:
                key, value = body.split(':', 1)
                comments[key.strip()] = value.strip()
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        lead_names.append(parts[-1])
        gain, baseline = _parse_gain_baseline(parts[2])
        gains.append(gain)
        baselines.append(baseline)

    dx_codes = [
        code.strip()
        for code in comments.get('Dx', '').split(',')
        if code.strip() and code.strip().lower() != 'nan'
    ]
    dx = list(dict.fromkeys(dx_codes))
    return {
        'record_id': record_id,
        'hea_path': str(path),
        'mat_path': str(Path(path).with_suffix('.mat')),
        'n_leads': n_leads,
        'sampling_rate': sampling_rate,
        'signal_length': signal_length,
        'lead_names': lead_names,
        'gains': gains,
        'baselines': baselines,
        'age': comments.get('Age', ''),
        'sex': comments.get('Sex', ''),
        'dx': dx,
        'dx_raw': comments.get('Dx', ''),
    }


def _parse_gain_baseline(field):
    match = re.match(r'(?P<gain>[-+0-9.]+)(?:\((?P<baseline>[-+0-9.]+)\))?', field)
    if not match:
        return 1.0, 0.0
    gain = float(match.group('gain') or 1.0)
    baseline = float(match.group('baseline') or 0.0)
    if gain == 0.0:
        gain = 1.0
    return gain, baseline


def scan_records(georgia_root):
    root = resolve_georgia_root(georgia_root)
    records = []
    for header_path in sorted(root.rglob('*.hea')):
        record = parse_hea(header_path)
        if not Path(record['mat_path']).is_file():
            raise FileNotFoundError(f'Missing MAT file for {record["record_id"]}: {record["mat_path"]}')
        records.append(record)
    if not records:
        raise FileNotFoundError(f'No .hea files found under {root}')
    return records


def build_label_map(records, min_label_count):
    counts = Counter(code for record in records for code in set(record['dx']))
    kept_codes = [
        code
        for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_label_count
    ]
    return {
        'min_label_count': int(min_label_count),
        'cache_version': CACHE_VERSION,
        'lead_order': LEAD_ORDER,
        'sampling_frequency': TARGET_FS,
        'signal_length': TARGET_LENGTH,
        'num_classes': len(kept_codes),
        'labels': [
            {'index': index, 'code': code, 'count': int(counts[code])}
            for index, code in enumerate(kept_codes)
        ],
    }


def process_signal(record):
    signal = sio.loadmat(record['mat_path'])['val'].astype(np.float32)
    if signal.ndim != 2:
        raise ValueError(f'Expected 2D signal in {record["mat_path"]}, got shape {signal.shape}')

    lead_names = record['lead_names']
    if signal.shape[0] != len(lead_names):
        raise ValueError(
            f'Lead count mismatch for {record["record_id"]}: signal shape {signal.shape}, '
            f'header leads {len(lead_names)}'
        )
    missing = [lead for lead in LEAD_ORDER if lead not in lead_names]
    if missing:
        raise ValueError(f'Record {record["record_id"]} is missing leads: {missing}')
    order = [lead_names.index(lead) for lead in LEAD_ORDER]
    signal = signal[order]

    gains = np.asarray(record.get('gains') or [1.0] * len(lead_names), dtype=np.float32)[order]
    baselines = np.asarray(record.get('baselines') or [0.0] * len(lead_names), dtype=np.float32)[order]
    signal = (signal - baselines[:, None]) / gains[:, None]

    source_fs = float(record['sampling_rate'])
    raw_target_len = int(round(source_fs * 10.0))
    if signal.shape[1] > raw_target_len:
        start = (signal.shape[1] - raw_target_len) // 2
        signal = signal[:, start:start + raw_target_len]

    if int(round(source_fs)) == 500:
        signal = resample_poly(signal, up=1, down=5, axis=1).astype(np.float32)
    elif int(round(source_fs)) == TARGET_FS:
        signal = signal.astype(np.float32)
    else:
        target = int(round(signal.shape[1] * TARGET_FS / source_fs))
        signal = resample(signal, target, axis=1).astype(np.float32)

    if signal.shape[1] > TARGET_LENGTH:
        start = (signal.shape[1] - TARGET_LENGTH) // 2
        signal = signal[:, start:start + TARGET_LENGTH]
    elif signal.shape[1] < TARGET_LENGTH:
        pad = TARGET_LENGTH - signal.shape[1]
        left = pad // 2
        right = pad - left
        signal = np.pad(signal, ((0, 0), (left, right)), mode='constant')

    if signal.shape != (len(LEAD_ORDER), TARGET_LENGTH):
        raise ValueError(f'Unexpected processed shape for {record["record_id"]}: {signal.shape}')
    return signal.T.astype(np.float32)


def primary_label_for(record, label_counts, label_to_index):
    kept = [code for code in record['dx'] if code in label_to_index]
    if not kept:
        return ''
    return min(kept, key=lambda code: (label_counts[code], code))


def make_splits(metadata, label_matrix, seed, split_path):
    indices = np.arange(len(metadata))
    primary = np.asarray([row['primary_label'] for row in metadata])
    fallback = False
    method = 'two_stage_primary_label_stratified'

    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=seed,
            shuffle=True,
            stratify=primary,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.125,
            random_state=seed,
            shuffle=True,
            stratify=primary[train_val_idx],
        )
    except ValueError as exc:
        fallback = True
        method = 'random_fallback'
        print(f'Warning: Georgia stratified split fallback to random split: {exc}')
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(indices)
        n = len(shuffled)
        n_train = int(n * 0.7)
        n_val = int(n * 0.1)
        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train:n_train + n_val]
        test_idx = shuffled[n_train + n_val:]

    split_indices = {
        'train': sorted(int(i) for i in train_idx),
        'val': sorted(int(i) for i in val_idx),
        'test': sorted(int(i) for i in test_idx),
    }
    split_payload = {
        'seed': int(seed),
        'ratios': {'train': 0.7, 'val': 0.1, 'test': 0.2},
        'method': method,
        'fallback': fallback,
        'splits': {},
    }
    for split_name, split_idx in split_indices.items():
        y = label_matrix[split_idx]
        label_counts = y.sum(axis=0).astype(int).tolist()
        positive_counts = [count for count in label_counts if count > 0]
        split_payload['splits'][split_name] = {
            'indices': split_idx,
            'record_ids': [metadata[i]['record_id'] for i in split_idx],
            'num_samples': len(split_idx),
            'label_positive_counts': label_counts,
            'min_positive_count': int(min(positive_counts)) if positive_counts else 0,
        }

    with open(split_path, 'w', encoding='utf-8') as split_file:
        json.dump(split_payload, split_file, indent=2, ensure_ascii=False)
    return split_payload


def print_split_summary(split_payload, label_map):
    labels = [item['code'] for item in label_map['labels']]
    print('Georgia split summary:')
    for split_name in ('train', 'val', 'test'):
        split = split_payload['splits'][split_name]
        counts = {
            labels[index]: int(count)
            for index, count in enumerate(split['label_positive_counts'])
        }
        print(
            f'  {split_name}: n={split["num_samples"]}, '
            f'min_positive_count={split["min_positive_count"]}, '
            f'label_counts={counts}'
        )


def _write_metadata_csv(path, metadata):
    fieldnames = [
        'record_id', 'hea_path', 'mat_path', 'sampling_rate', 'signal_length',
        'age', 'sex', 'dx_raw', 'kept_dx', 'primary_label',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def _load_json(path):
    with open(path, encoding='utf-8') as json_file:
        return json.load(json_file)


def prepare_cache(datafolder, min_label_count=None, seed=None, force_rebuild=False):
    min_label_count = int(min_label_count or getattr(config, 'georgia_min_label_count', 100))
    seed = int(seed if seed is not None else getattr(config, 'seed', 1))
    cache_dir = get_cache_dir(min_label_count)
    signals_path = cache_dir / 'signals.npy'
    labels_path = cache_dir / 'labels.npy'
    metadata_path = cache_dir / 'metadata.csv'
    label_map_path = cache_dir / 'label_map.json'
    split_path = cache_dir / f'splits_seed{seed}_min{min_label_count}.json'

    cache_ready = (
        signals_path.is_file() and
        labels_path.is_file() and
        metadata_path.is_file() and
        label_map_path.is_file() and
        split_path.is_file()
    )
    if cache_ready and not force_rebuild:
        cached_label_map = _load_json(label_map_path)
        if cached_label_map.get('cache_version') == CACHE_VERSION:
            print(f'Georgia processed cache found: {cache_dir}')
            return {
                'cache_dir': str(cache_dir),
                'signals_path': str(signals_path),
                'labels_path': str(labels_path),
                'metadata_path': str(metadata_path),
                'label_map_path': str(label_map_path),
                'split_path': str(split_path),
            }
        print(f'Georgia cache version changed; rebuilding: {cache_dir}')

    os.makedirs(cache_dir, exist_ok=True)
    records = scan_records(datafolder)
    label_counts = Counter(code for record in records for code in set(record['dx']))
    label_map = build_label_map(records, min_label_count)
    kept_codes = [entry['code'] for entry in label_map['labels']]
    label_to_index = {code: index for index, code in enumerate(kept_codes)}
    filtered_records = []
    for record in records:
        kept = list(dict.fromkeys(code for code in record['dx'] if code in label_to_index))
        if not kept:
            continue
        record = dict(record)
        record['kept_dx_list'] = kept
        record['primary_label'] = primary_label_for(record, label_counts, label_to_index)
        filtered_records.append(record)

    if len(kept_codes) == 0:
        raise ValueError(f'No Georgia labels meet min_label_count={min_label_count}.')
    if not filtered_records:
        raise ValueError('No Georgia records remain after label filtering.')

    signals_tmp = cache_dir / 'signals.tmp.npy'
    labels_tmp = cache_dir / 'labels.tmp.npy'
    signals = open_memmap(
        signals_tmp,
        mode='w+',
        dtype=np.float32,
        shape=(len(filtered_records), TARGET_LENGTH, len(LEAD_ORDER)),
    )
    labels = np.zeros((len(filtered_records), len(kept_codes)), dtype=np.float32)
    metadata = []

    print(
        f'Building Georgia processed cache at {cache_dir}: '
        f'{len(filtered_records)} records, {len(kept_codes)} labels.'
    )
    for index, record in enumerate(filtered_records):
        signals[index] = process_signal(record)
        for code in record['kept_dx_list']:
            labels[index, label_to_index[code]] = 1.0
        metadata.append({
            'record_id': record['record_id'],
            'hea_path': record['hea_path'],
            'mat_path': record['mat_path'],
            'sampling_rate': record['sampling_rate'],
            'signal_length': record['signal_length'],
            'age': record['age'],
            'sex': record['sex'],
            'dx_raw': record['dx_raw'],
            'kept_dx': ','.join(record['kept_dx_list']),
            'primary_label': record['primary_label'],
        })
        if (index + 1) % 1000 == 0 or index + 1 == len(filtered_records):
            print(f'  processed {index + 1}/{len(filtered_records)} Georgia records')

    signals.flush()
    del signals
    np.save(labels_tmp, labels)
    os.replace(signals_tmp, signals_path)
    os.replace(labels_tmp, labels_path)
    _write_metadata_csv(metadata_path, metadata)
    with open(label_map_path, 'w', encoding='utf-8') as label_file:
        json.dump(label_map, label_file, indent=2, ensure_ascii=False)

    split_payload = make_splits(metadata, labels, seed, split_path)
    print_split_summary(split_payload, label_map)
    return {
        'cache_dir': str(cache_dir),
        'signals_path': str(signals_path),
        'labels_path': str(labels_path),
        'metadata_path': str(metadata_path),
        'label_map_path': str(label_map_path),
        'split_path': str(split_path),
    }


def _standardize_by_train(x_train, x_val, x_test):
    mean = float(np.mean(x_train))
    std = float(np.std(x_train))
    if not np.isfinite(std) or std < 1e-6:
        warnings.warn('Georgia train split standard deviation is near zero; skipping scaling.')
        return x_train, x_val, x_test
    print(f'Georgia standardization: train_mean={mean:.6f}, train_std={std:.6f}')
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
    )


def load_data(datafolder=None, min_label_count=None, seed=None, standardize=True):
    paths = prepare_cache(datafolder, min_label_count=min_label_count, seed=seed)
    label_map = _load_json(paths['label_map_path'])
    split_payload = _load_json(paths['split_path'])
    signals = np.load(paths['signals_path'], mmap_mode='r')
    labels = np.load(paths['labels_path'])

    train_idx = np.asarray(split_payload['splits']['train']['indices'], dtype=np.int64)
    val_idx = np.asarray(split_payload['splits']['val']['indices'], dtype=np.int64)
    test_idx = np.asarray(split_payload['splits']['test']['indices'], dtype=np.int64)

    x_train = np.asarray(signals[train_idx], dtype=np.float32)
    x_val = np.asarray(signals[val_idx], dtype=np.float32)
    x_test = np.asarray(signals[test_idx], dtype=np.float32)
    if standardize:
        x_train, x_val, x_test = _standardize_by_train(x_train, x_val, x_test)

    y_train = labels[train_idx].astype(np.float32)
    y_val = labels[val_idx].astype(np.float32)
    y_test = labels[test_idx].astype(np.float32)

    print_split_summary(split_payload, label_map)
    print(
        'Georgia loader ready:',
        f'x_train={x_train.shape}, y_train={y_train.shape},',
        f'x_val={x_val.shape}, y_val={y_val.shape},',
        f'x_test={x_test.shape}, y_test={y_test.shape},',
        f'label_map={paths["label_map_path"]}',
    )
    config.georgia_cache_dir = paths['cache_dir']
    config.georgia_label_map_path = paths['label_map_path']
    config.georgia_split_path = paths['split_path']
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    load_data(getattr(config, 'datafolder', None))
