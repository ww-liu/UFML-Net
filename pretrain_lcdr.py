# -*- coding: utf-8 -*-
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from config import config
from dataset import load_datasets
from main_mul import (
    apply_dataset_overrides,
    apply_experiment_overrides,
    apply_model_overrides,
    get_artifact_dataset_tag,
    get_result_dataset_tag,
    get_tagged_experiment_name,
)
from models.model_factory import build_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def random_crop_resize(signal, crop_ratio_low):
    # signal: [B, T]
    batch_size, length = signal.shape
    crop_ratio_low = float(crop_ratio_low)
    if crop_ratio_low >= 1.0:
        return signal

    views = []
    min_len = max(8, int(length * crop_ratio_low))
    for idx in range(batch_size):
        crop_len = random.randint(min_len, length)
        start = random.randint(0, length - crop_len) if crop_len < length else 0
        crop = signal[idx:idx + 1, start:start + crop_len].unsqueeze(1)
        resized = F.interpolate(crop, size=length, mode='linear', align_corners=False)
        views.append(resized.squeeze(1))
    return torch.cat(views, dim=0)


def random_timeout(signal, mask_ratio):
    # signal: [B, T]
    mask_ratio = float(mask_ratio)
    if mask_ratio <= 0.0:
        return signal

    batch_size, length = signal.shape
    max_mask_len = max(1, int(length * mask_ratio))
    output = signal.clone()
    for idx in range(batch_size):
        mask_len = random.randint(1, max_mask_len)
        start = random.randint(0, max(0, length - mask_len))
        output[idx, start:start + mask_len] = 0.0
    return output


def augment_lead_signal(signal, crop_ratio_low, mask_ratio):
    signal = random_crop_resize(signal, crop_ratio_low)
    signal = random_timeout(signal, mask_ratio)
    return signal


def make_lead_view(inputs, lead_index, crop_ratio_low, mask_ratio, context_scale):
    # inputs: [B, T, 12]
    batch_size = inputs.size(0)
    view = inputs * float(context_scale)
    batch_ids = torch.arange(batch_size, device=inputs.device)
    lead_signal = inputs[batch_ids, :, lead_index]
    lead_signal = augment_lead_signal(lead_signal, crop_ratio_low, mask_ratio)
    view[batch_ids, :, lead_index] = lead_signal
    return view


def normalized_regression_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2.0 - 2.0 * (x * y).sum(dim=1).mean()


def cross_correlation_decorrelation_loss(x, y):
    x = (x - x.mean(dim=0)) / (x.std(dim=0, unbiased=False) + 1e-6)
    y = (y - y.mean(dim=0)) / (y.std(dim=0, unbiased=False) + 1e-6)
    corr = torch.matmul(x.T, y) / max(1, x.size(0))
    diag = torch.diagonal(corr)
    on_diag = torch.mean((diag - 1.0) ** 2)
    off_diag = corr - torch.diag(diag)
    off_diag = torch.mean(off_diag ** 2)
    return on_diag + off_diag


class LCDRProjectionHead(nn.Module):
    def __init__(self, hidden_dim, projection_dim):
        super(LCDRProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(x.float())


def extract_main_feature(model, inputs):
    feat1, _, feat2, _, _, _ = model(inputs)
    return torch.cat([feat1.float(), feat2.float()], dim=1)


def save_pretrain_log(log_path, row):
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def pretrain_lcdr():
    setup_seed(config.seed)
    result_dataset_tag = get_result_dataset_tag(config.datafolder)
    artifact_dataset_tag = get_artifact_dataset_tag(config.datafolder)
    applied_experiment_overrides = apply_experiment_overrides(config, result_dataset_tag)
    applied_dataset_overrides = apply_dataset_overrides(config, result_dataset_tag)
    applied_model_overrides = apply_model_overrides(config)

    if config.model_name != 'MyNet6View':
        raise ValueError('LCDR pretraining is only implemented for MyNet6View.')

    print('torch.cuda.is_available:', torch.cuda.is_available())
    print('dataset:', result_dataset_tag)
    if applied_experiment_overrides:
        print('applied_experiment_overrides:', applied_experiment_overrides)
    if applied_dataset_overrides:
        print('applied_dataset_overrides:', applied_dataset_overrides)
    if applied_model_overrides:
        print('applied_model_overrides:', applied_model_overrides)

    train_dataloader, _, _, num_classes = load_datasets(
        experiment=config.experiment,
        datafolder=config.datafolder
    )

    model = build_model(config, num_classes=num_classes, input_channels=12).to(device)
    projector = LCDRProjectionHead(
        hidden_dim=getattr(config, 'lcdr_pretrain_hidden_dim', 256),
        projection_dim=getattr(config, 'lcdr_pretrain_projection_dim', 128),
    ).to(device)

    sample_inputs, _ = next(iter(train_dataloader))
    sample_inputs = sample_inputs[:min(2, sample_inputs.size(0))].to(device)
    model.eval()
    projector.train()
    with torch.no_grad():
        _ = projector(extract_main_feature(model, sample_inputs))

    optimizer = optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=getattr(config, 'lcdr_pretrain_lr', 0.001),
        weight_decay=getattr(config, 'lcdr_pretrain_weight_decay', 0.0001),
    )

    ensure_dir('checkpoints')
    ensure_dir('results')
    checkpoint_path = getattr(config, 'lcdr_pretrained_path', '')
    if not checkpoint_path:
        checkpoint_path = os.path.join(
            'checkpoints',
            f'{config.model_name}_{artifact_dataset_tag}_{get_tagged_experiment_name(config.experiment)}_lcdr_pretrain.pth'
        )
    log_path = os.path.join(
        'results',
        f'lcdr_pretrain_{artifact_dataset_tag}_{get_tagged_experiment_name(config.experiment)}.csv'
    )

    crop_ratio_low = getattr(config, 'lcd_crop_ratio_low', 0.85)
    mask_ratio = getattr(config, 'lcd_mask_ratio_up', 0.10)
    context_scale = getattr(config, 'lcdr_pretrain_context_scale', 0.05)
    inter_weight = getattr(config, 'lcdr_pretrain_inter_weight', 0.5)
    decor_weight = getattr(config, 'lcdr_pretrain_decor_weight', 0.02)
    epochs = getattr(config, 'lcdr_pretrain_epochs', 20)
    best_loss = float('inf')

    print(f'LCDR pretraining for {epochs} epochs')
    print(f'checkpoint: {checkpoint_path}')

    for epoch in range(1, epochs + 1):
        model.train()
        projector.train()
        epoch_loss = 0.0
        epoch_intra = 0.0
        epoch_inter = 0.0
        epoch_decor = 0.0
        steps = 0
        since = time.time()

        for inputs, _ in train_dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            num_leads = inputs.size(2)
            lead_a = torch.randint(0, num_leads, (batch_size,), device=device)
            lead_b = torch.randint(0, num_leads, (batch_size,), device=device)
            same_lead = lead_a == lead_b
            lead_b[same_lead] = (lead_b[same_lead] + 1) % num_leads

            view_a1 = make_lead_view(inputs, lead_a, crop_ratio_low, mask_ratio, context_scale)
            view_a2 = make_lead_view(inputs, lead_a, crop_ratio_low, mask_ratio, context_scale)
            view_b = make_lead_view(inputs, lead_b, crop_ratio_low, mask_ratio, context_scale)

            z_a1 = projector(extract_main_feature(model, view_a1))
            z_a2 = projector(extract_main_feature(model, view_a2))
            z_b = projector(extract_main_feature(model, view_b))

            intra_loss = normalized_regression_loss(z_a1, z_a2)
            inter_loss = 0.5 * (
                normalized_regression_loss(z_a1, z_b) +
                normalized_regression_loss(z_a2, z_b)
            )
            decor_loss = cross_correlation_decorrelation_loss(z_a1, z_a2)
            loss = intra_loss + inter_weight * inter_loss + decor_weight * decor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_intra += intra_loss.item()
            epoch_inter += inter_loss.item()
            epoch_decor += decor_loss.item()
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        avg_intra = epoch_intra / max(1, steps)
        avg_inter = epoch_inter / max(1, steps)
        avg_decor = epoch_decor / max(1, steps)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'dataset': artifact_dataset_tag,
                'experiment': config.experiment,
                'run_tag': getattr(config, 'run_tag', ''),
            }, checkpoint_path)
            saved = 'yes'
        else:
            saved = 'no'

        row = {
            'dataset': artifact_dataset_tag,
            'experiment': config.experiment,
            'model_name': config.model_name,
            'seed': config.seed,
            'epoch': epoch,
            'loss': avg_loss,
            'intra_loss': avg_intra,
            'inter_loss': avg_inter,
            'decor_loss': avg_decor,
            'best_loss': best_loss,
            'saved': saved,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        save_pretrain_log(log_path, row)
        print(
            f'#pretrain epoch: {epoch} loss: {avg_loss:.5f} '
            f'intra: {avg_intra:.5f} inter: {avg_inter:.5f} '
            f'decor: {avg_decor:.5f} saved: {saved} '
            f'time: {time.time() - since:.1f}s'
        )

    print(f'LCDR pretraining finished. Best checkpoint: {checkpoint_path}')
    print(f'Pretraining log saved to: {log_path}')


if __name__ == '__main__':
    pretrain_lcdr()
