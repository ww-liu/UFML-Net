# -*- coding: utf-8 -*-
"""
@time: 2021/4/15 15:40

@ author:
"""
import warnings
import torch.nn.functional as F
import torch
import time
import os
from segmentation_models_pytorch.losses import FocalLoss
from torch.optim.lr_scheduler import MultiStepLR
import utils
from torch import nn, optim
from dataset import load_datasets, detect_dataset_type
from config import config
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillKL(nn.Module):
    """KL divergence for distillation."""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        return F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)


class ModelEma(nn.Module):
    # Exponential moving average for late-stage evaluation stability.
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.module = self._clone_model(model)
        self.decay = decay

    @staticmethod
    def _clone_model(model):
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model

    def update(self, model):
        with torch.no_grad():
            model_state = model.state_dict()
            for name, ema_v in self.module.state_dict().items():
                model_v = model_state[name].detach()
                if not torch.is_floating_point(ema_v):
                    ema_v.copy_(model_v)
                else:
                    ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)


class ModelAverage(nn.Module):
    # Simple checkpoint averaging used as lightweight SWA.
    def __init__(self, model):
        super().__init__()
        self.module = ModelEma._clone_model(model)
        self.num_models = 0

    def update(self, model):
        with torch.no_grad():
            model_state = model.state_dict()
            self.num_models += 1
            for name, avg_v in self.module.state_dict().items():
                model_v = model_state[name].detach()
                if not torch.is_floating_point(avg_v):
                    avg_v.copy_(model_v)
                else:
                    avg_v.mul_((self.num_models - 1.0) / self.num_models).add_(model_v / self.num_models)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_checkpoint(filepath, model, optimizer):
    if os.path.isfile(filepath):
        print("Loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model
    print("No checkpoint found at '{}'".format(filepath))
    return None


def get_result_dataset_tag(datafolder):
    dataset_type = detect_dataset_type(datafolder)
    if dataset_type == 'ptbxl':
        return 'ptbxl'
    if dataset_type == 'cpsc':
        return 'cpsc'
    return os.path.basename(os.path.abspath(datafolder)) or 'hf'


def get_artifact_dataset_tag(datafolder):
    dataset_type = detect_dataset_type(datafolder)
    if dataset_type == 'ptbxl':
        return 'ptbxl'
    if dataset_type == 'cpsc':
        base = os.path.basename(os.path.abspath(datafolder))
        return base or 'cpsc'
    return os.path.basename(os.path.abspath(datafolder)) or 'hf'


def save_checkpoint(best_auc, model, optimizer, epoch, dataset_tag='dataset'):
    print('Model Saving...')
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    }, os.path.join('checkpoints', f'{config.model_name}_{dataset_tag}_{config.experiment}_checkpoint_best.pth'))


def get_uncertainty_weight(logits, min_weight=0.1, reduce=True):
    probs = torch.sigmoid(logits)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6)) +
                (1.0 - probs) * torch.log((1.0 - probs).clamp_min(1e-6)))
    entropy = entropy / np.log(2.0)
    if reduce:
        entropy = entropy.mean(dim=1)
    certainty = (1.0 - entropy).clamp(0.0, 1.0)
    return certainty * (1.0 - min_weight) + min_weight


def weighted_feature_kd(student_feat, teacher_feat, teacher_weight, temperature):
    student_log_prob = F.log_softmax(student_feat / temperature, dim=1)
    teacher_prob = F.softmax(teacher_feat.detach() / temperature, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=1)
    return (loss * teacher_weight.detach()).mean() * (temperature ** 2)


def weighted_logit_kd(student_logits, teacher_logits, teacher_weight, temperature):
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=1)
    return (loss * teacher_weight.detach()).mean() * (temperature ** 2)


def get_rampup_weight(epoch, total_epochs):
    if total_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(total_epochs))


def fuse_branch_predictions(out1, out2, tf_logits=None, category_fusion_logits=None):
    # Per-class uncertainty weighting lets each label prefer a different branch.
    certainty1 = get_uncertainty_weight(out1, min_weight=config.uncertainty_min_weight, reduce=False)
    certainty2 = get_uncertainty_weight(out2, min_weight=config.uncertainty_min_weight, reduce=False)
    weight1 = certainty1.pow(
        config.fusion_conf_power
    ) * config.branch1_fusion_bias
    weight2 = certainty2.pow(
        config.fusion_conf_power
    ) * config.branch2_fusion_bias
    prob1 = torch.sigmoid(out1)
    prob2 = torch.sigmoid(out2)
    norm = weight1 + weight2 + 1e-6
    fused_prob = (prob1 * weight1 + prob2 * weight2) / norm
    fused_logit = (out1 * weight1 + out2 * weight2) / norm
    fused_logit_prob = torch.sigmoid(config.fusion_logit_scale * fused_logit)
    fused_prob = (
        (1.0 - config.fusion_logit_blend) * fused_prob +
        config.fusion_logit_blend * fused_logit_prob
    )

    stronger_prob = torch.where(weight2 >= weight1, prob2, prob1)
    disagreement = torch.abs(prob2 - prob1)
    positive_conf = torch.clamp(
        (torch.maximum(prob1, prob2) - config.fusion_positive_margin) /
        max(1e-6, 1.0 - config.fusion_positive_margin),
        min=0.0,
        max=1.0
    )
    residual_gate = (disagreement * positive_conf).pow(config.fusion_gate_power)
    fused_prob = fused_prob + config.fusion_residual_weight * residual_gate * (stronger_prob - fused_prob)
    fused_prob = fused_prob.clamp(1e-6, 1.0 - 1e-6)

    if tf_logits is not None:
        tf_prob = torch.sigmoid(tf_logits)
        tf_certainty = get_uncertainty_weight(tf_logits, min_weight=config.uncertainty_min_weight, reduce=False)
        tf_disagreement = torch.abs(prob2 - prob1).detach().pow(getattr(config, 'tf_fusion_disagreement_power', 1.0))
        tf_gate = tf_certainty * tf_disagreement
        fused_prob = fused_prob + getattr(config, 'tf_fusion_alpha', 0.25) * tf_gate * (tf_prob - fused_prob)
        fused_prob = fused_prob.clamp(1e-6, 1.0 - 1e-6)

    if category_fusion_logits is not None:
        category_prob = torch.sigmoid(category_fusion_logits)
        fused_prob = (
            (1.0 - getattr(config, 'category_fusion_alpha', 0.25)) * fused_prob +
            getattr(config, 'category_fusion_alpha', 0.25) * category_prob
        )
        fused_prob = fused_prob.clamp(1e-6, 1.0 - 1e-6)

    return prob1, prob2, fused_prob


def compute_metrics(targets, predictions):
    auc = roc_auc_score(targets, predictions)
    tpr = utils.compute_TPR(targets, predictions)
    acc = utils.compute_ACC(targets, predictions)
    f1 = utils.compute_F1(targets, predictions)
    return auc, tpr, acc, f1


def collect_outputs(prob1, prob2, probf, target, outputs1, outputs2, outputsf, targets):
    for i in range(len(target)):
        outputs1.append(prob1[i].cpu().detach().numpy())
        outputs2.append(prob2[i].cpu().detach().numpy())
        outputsf.append(probf[i].cpu().detach().numpy())
        targets.append(target[i].cpu().detach().numpy())


def _as_model_list(model_or_models):
    if isinstance(model_or_models, (list, tuple)):
        return list(model_or_models)
    return [model_or_models]


def unpack_model_outputs(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 6:
            feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits = outputs
            return feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits
        if len(outputs) == 5:
            feat_ins, out_ins, feat_all, out, tf_logits = outputs
            return feat_ins, out_ins, feat_all, out, tf_logits, None
        if len(outputs) == 4:
            feat_ins, out_ins, feat_all, out = outputs
            return feat_ins, out_ins, feat_all, out, None, None
    raise ValueError('Unexpected model outputs format')


def save_results_to_csv(run_id, experiment, model_name, seed, best_epoch,
                        bauc1, btpr1, bacc1, bf11,
                        bauc2, btpr2, bacc2, bf12,
                        baucf, btprf, baccf, bf1f,
                        dataset_tag='dataset'):
    results_dir = 'results'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    csv_path = os.path.join(results_dir, f'results_{dataset_tag}.csv')

    result_data = {
        'run_id': [run_id],
        'dataset': [dataset_tag],
        'experiment': [experiment],
        'model_name': [model_name],
        'seed': [seed],
        'best_epoch': [best_epoch],
        'branch1_auc': [bauc1],
        'branch1_tpr': [btpr1],
        'branch1_acc': [bacc1],
        'branch1_f1': [bf11],
        'branch2_auc': [bauc2],
        'branch2_tpr': [btpr2],
        'branch2_acc': [bacc2],
        'branch2_f1': [bf12],
        'fusion_auc': [baucf],
        'fusion_tpr': [btprf],
        'fusion_acc': [baccf],
        'fusion_f1': [bf1f],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }

    df = pd.DataFrame(result_data)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f'结果已追加到: {csv_path}')
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f'结果已保存到: {csv_path}')


def save_epoch_result_to_csv(run_id, experiment, model_name, seed, epoch,
                             train_auc1, train_tpr1, train_acc1, train_f11,
                             train_auc2, train_tpr2, train_acc2, train_f12,
                             train_aucf, train_tprf, train_accf, train_f1f,
                             val_auc1, val_tpr1, val_acc1, val_f11,
                             val_auc2, val_tpr2, val_acc2, val_f12,
                             val_aucf, val_tprf, val_accf, val_f1f,
                             test_auc1, test_tpr1, test_acc1, test_f11,
                             test_auc2, test_tpr2, test_acc2, test_f12,
                             test_aucf, test_tprf, test_accf, test_f1f,
                             train_loss, val_loss, test_loss,
                             dataset_tag='dataset',
                             csv_file_prefix='training_log'):
    results_dir = 'results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    result_data = {
        'run_id': [run_id],
        'dataset': [dataset_tag],
        'experiment': [experiment],
        'model_name': [model_name],
        'seed': [seed],
        'epoch': [epoch],
        'train_branch1_auc': [train_auc1],
        'train_branch1_tpr': [train_tpr1],
        'train_branch1_acc': [train_acc1],
        'train_branch1_f1': [train_f11],
        'train_branch2_auc': [train_auc2],
        'train_branch2_tpr': [train_tpr2],
        'train_branch2_acc': [train_acc2],
        'train_branch2_f1': [train_f12],
        'train_fusion_auc': [train_aucf],
        'train_fusion_tpr': [train_tprf],
        'train_fusion_acc': [train_accf],
        'train_fusion_f1': [train_f1f],
        'val_branch1_auc': [val_auc1],
        'val_branch1_tpr': [val_tpr1],
        'val_branch1_acc': [val_acc1],
        'val_branch1_f1': [val_f11],
        'val_branch2_auc': [val_auc2],
        'val_branch2_tpr': [val_tpr2],
        'val_branch2_acc': [val_acc2],
        'val_branch2_f1': [val_f12],
        'val_fusion_auc': [val_aucf],
        'val_fusion_tpr': [val_tprf],
        'val_fusion_acc': [val_accf],
        'val_fusion_f1': [val_f1f],
        'test_branch1_auc': [test_auc1],
        'test_branch1_tpr': [test_tpr1],
        'test_branch1_acc': [test_acc1],
        'test_branch1_f1': [test_f11],
        'test_branch2_auc': [test_auc2],
        'test_branch2_tpr': [test_tpr2],
        'test_branch2_acc': [test_acc2],
        'test_branch2_f1': [test_f12],
        'test_fusion_auc': [test_aucf],
        'test_fusion_tpr': [test_tprf],
        'test_fusion_acc': [test_accf],
        'test_fusion_f1': [test_f1f],
        'train_loss': [train_loss],
        'val_loss': [val_loss],
        'test_loss': [test_loss],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }

    df = pd.DataFrame(result_data)
    csv_filename = f"{csv_file_prefix}_{dataset_tag}_{experiment}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    # If an older log file uses a different schema, write to a versioned file
    # instead of silently appending misaligned rows.
    if os.path.exists(csv_path):
        try:
            existing_header = pd.read_csv(csv_path, nrows=0)
            if list(existing_header.columns) != list(df.columns):
                csv_filename = f"{csv_file_prefix}_v2_{dataset_tag}_{experiment}.csv"
                csv_path = os.path.join(results_dir, csv_filename)
        except Exception:
            csv_filename = f"{csv_file_prefix}_v2_{dataset_tag}_{experiment}.csv"
            csv_path = os.path.join(results_dir, csv_filename)

    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)
    if write_header:
        print(f'Detailed log file created for experiment "{experiment}": {csv_path}')


def save_final_test_to_csv(run_id, experiment, model_name, seed, best_epoch,
                           bauc1, btpr1, bacc1, bf11,
                           bauc2, btpr2, bacc2, bf12,
                           baucf, btprf, baccf, bf1f,
                           dataset_tag='dataset'):
    results_dir = 'results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    csv_path = os.path.join(results_dir, f'final_test_{dataset_tag}_{experiment}.csv')
    result_data = {
        'run_id': [run_id],
        'dataset': [dataset_tag],
        'experiment': [experiment],
        'model_name': [model_name],
        'seed': [seed],
        'best_epoch': [best_epoch],
        'branch1_auc': [bauc1],
        'branch1_tpr': [btpr1],
        'branch1_acc': [bacc1],
        'branch1_f1': [bf11],
        'branch2_auc': [bauc2],
        'branch2_tpr': [btpr2],
        'branch2_acc': [bacc2],
        'branch2_f1': [bf12],
        'fusion_auc': [baucf],
        'fusion_tpr': [btprf],
        'fusion_acc': [baccf],
        'fusion_f1': [bf1f],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }

    df = pd.DataFrame(result_data)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)


def train_epoch(model, ema_model, optimizer, criterion, criterion_cross, scheduler_teacher_model, train_dataloader, epoch):
    model.train()
    loss_meter, it_count = 0, 0
    mutual_loss_meter = 0
    tf_loss_meter = 0
    category_fusion_loss_meter = 0
    outputs1 = []
    outputs2 = []
    outputsf = []
    targets = []
    rampup = get_rampup_weight(epoch, config.uncertainty_rampup_epochs)

    for inputs1, target in train_dataloader:
        inputs1 = inputs1.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits = unpack_model_outputs(model(inputs1))
        feat_ins = feat_ins.to(torch.float)
        out_ins = out_ins.to(torch.float)
        feat_all = feat_all.to(torch.float)
        out = out.to(torch.float)
        if tf_logits is not None:
            tf_logits = tf_logits.to(torch.float)
        if category_fusion_logits is not None:
            category_fusion_logits = category_fusion_logits.to(torch.float)

        loss1 = criterion(out_ins, target)
        loss2 = criterion(out, target)
        tf_loss = out_ins.new_tensor(0.0)
        if tf_logits is not None:
            tf_loss = getattr(config, 'tf_loss_weight', 0.2) * criterion(tf_logits, target)
        category_fusion_loss = out_ins.new_tensor(0.0)
        if category_fusion_logits is not None:
            category_fusion_loss = getattr(config, 'category_fusion_loss_weight', 0.2) * criterion(category_fusion_logits, target)

        certainty1 = get_uncertainty_weight(out_ins, min_weight=config.uncertainty_min_weight)
        certainty2 = get_uncertainty_weight(out, min_weight=config.uncertainty_min_weight)

        mutual_feat_loss = weighted_feature_kd(
            feat_ins, feat_all, certainty2, config.feature_temperature
        ) + weighted_feature_kd(
            feat_all, feat_ins, certainty1, config.feature_temperature
        )
        mutual_logit_loss = weighted_logit_kd(
            out_ins, out, certainty2, config.mutual_temperature
        ) + weighted_logit_kd(
            out, out_ins, certainty1, config.mutual_temperature
        )
        mutual_loss = rampup * (
            config.mutual_feat_weight * mutual_feat_loss +
            config.mutual_logit_weight * mutual_logit_loss
        )

        prob1, prob2, final_fusion_prob = fuse_branch_predictions(
            out_ins, out, tf_logits=tf_logits, category_fusion_logits=category_fusion_logits
        )
        loss = loss1 + loss2 + tf_loss + category_fusion_loss + mutual_loss
        loss.backward()
        optimizer.step()
        if ema_model is not None and epoch >= config.ema_start:
            ema_model.update(model)

        loss_meter += loss.item()
        mutual_loss_meter += mutual_loss.item()
        tf_loss_meter += tf_loss.item()
        category_fusion_loss_meter += category_fusion_loss.item()
        it_count += 1

        collect_outputs(prob1, prob2, final_fusion_prob, target, outputs1, outputs2, outputsf, targets)

    avg_loss = loss_meter / it_count
    avg_mutual_loss = mutual_loss_meter / it_count
    avg_category_fusion_loss = category_fusion_loss_meter / it_count
    train_auc1, train_tpr1, train_acc1, train_f11 = compute_metrics(targets, outputs1)
    train_auc2, train_tpr2, train_acc2, train_f12 = compute_metrics(targets, outputs2)
    train_aucf, train_tprf, train_accf, train_f1f = compute_metrics(targets, outputsf)

    avg_tf_loss = tf_loss_meter / it_count
    print('train_loss1: %.4f, m_loss1: %.4f, tf_loss1: %.4f, cf_loss1: %.4f, train_macro_auc1: %.4f,  TPR1: %.4f, acc1: %.4f, f11: %.4f' % (
        avg_loss, avg_mutual_loss, avg_tf_loss, avg_category_fusion_loss, train_auc1, train_tpr1, train_acc1, train_f11))
    print('train_loss2: %.4f, train_macro_auc2: %.4f,  TPR2: %.4f, acc2: %.4f, f12: %.4f' % (
        avg_loss, train_auc2, train_tpr2, train_acc2, train_f12))
    print('train_fusion: %.4f, fusion_TPR: %.4f, fusion_acc: %.4f, fusion_f1: %.4f' % (
        train_aucf, train_tprf, train_accf, train_f1f))
    print("-----------------------------------------------------------------------------")

    return (avg_loss, train_auc1, train_tpr1, train_acc1, train_f11,
            train_auc2, train_tpr2, train_acc2, train_f12,
            train_aucf, train_tprf, train_accf, train_f1f)


def val_epoch(model, criterion, criterion_cross, val_dataloader):
    models = _as_model_list(model)
    for m in models:
        m.eval()
    loss_meter, it_count = 0, 0
    outputs1 = []
    outputs2 = []
    outputsf = []
    targets = []

    with torch.no_grad():
        for inputs1, target in val_dataloader:
            inputs1 = inputs1.to(device)
            target = target.to(device)
            loss = 0.0
            avg_prob1 = 0.0
            avg_prob2 = 0.0
            avg_probf = 0.0

            for single_model in models:
                feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits = unpack_model_outputs(single_model(inputs1))
                feat_ins = feat_ins.to(torch.float)
                out_ins = out_ins.to(torch.float)
                feat_all = feat_all.to(torch.float)
                out = out.to(torch.float)
                if tf_logits is not None:
                    tf_logits = tf_logits.to(torch.float)
                if category_fusion_logits is not None:
                    category_fusion_logits = category_fusion_logits.to(torch.float)

                loss1 = criterion(out_ins, target)
                loss2 = criterion(out, target)
                tf_loss = out_ins.new_tensor(0.0)
                if tf_logits is not None:
                    tf_loss = getattr(config, 'tf_loss_weight', 0.2) * criterion(tf_logits, target)
                category_fusion_loss = out_ins.new_tensor(0.0)
                if category_fusion_logits is not None:
                    category_fusion_loss = getattr(config, 'category_fusion_loss_weight', 0.2) * criterion(category_fusion_logits, target)
                loss += loss1 + loss2 + tf_loss + category_fusion_loss

                prob1, prob2, final_fusion_prob = fuse_branch_predictions(
                    out_ins, out, tf_logits=tf_logits, category_fusion_logits=category_fusion_logits
                )
                avg_prob1 += prob1
                avg_prob2 += prob2
                avg_probf += final_fusion_prob

            loss = loss / len(models)
            loss_meter += loss.item()
            it_count += 1
            collect_outputs(
                avg_prob1 / len(models),
                avg_prob2 / len(models),
                avg_probf / len(models),
                target, outputs1, outputs2, outputsf, targets
            )

    auc1, tpr1, acc1, f11 = compute_metrics(targets, outputs1)
    auc2, tpr2, acc2, f12 = compute_metrics(targets, outputs2)
    aucf, tprf, accf, f1f = compute_metrics(targets, outputsf)

    print('val_loss1: %.4f,   val_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc1, tpr1, acc1, f11))
    print('val_loss: %.4f, macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc2, tpr2, acc2, f12))
    print('val_fusion: %.4f, fusion_TPR: %.4f, fusion_acc: %.4f, fusion_f1: %.4f' % (
        aucf, tprf, accf, f1f))
    print("-----------------------------------------------------------------------------")
    return auc1, tpr1, acc1, f11, auc2, tpr2, acc2, f12, aucf, tprf, accf, f1f, loss_meter / it_count


def test_epoch(model, criterion, criterion_cross, test_dataloader):
    models = _as_model_list(model)
    for m in models:
        m.eval()
    loss_meter, it_count = 0, 0
    mutual_loss_meter = 0
    outputs1 = []
    outputs2 = []
    outputsf = []
    targets = []

    with torch.no_grad():
        for inputs1, target in test_dataloader:
            inputs1 = inputs1.to(device)
            target = target.to(device)
            loss = 0.0
            mutual_loss = 0.0
            avg_prob1 = 0.0
            avg_prob2 = 0.0
            avg_probf = 0.0

            for single_model in models:
                feat_ins, out_ins, feat_all, out, tf_logits, category_fusion_logits = unpack_model_outputs(single_model(inputs1))
                feat_ins = feat_ins.to(torch.float)
                out_ins = out_ins.to(torch.float)
                feat_all = feat_all.to(torch.float)
                out = out.to(torch.float)
                if tf_logits is not None:
                    tf_logits = tf_logits.to(torch.float)
                if category_fusion_logits is not None:
                    category_fusion_logits = category_fusion_logits.to(torch.float)

                loss1 = criterion(out_ins, target)
                loss2 = criterion(out, target)
                tf_loss = out_ins.new_tensor(0.0)
                if tf_logits is not None:
                    tf_loss = getattr(config, 'tf_loss_weight', 0.2) * criterion(tf_logits, target)
                category_fusion_loss = out_ins.new_tensor(0.0)
                if category_fusion_logits is not None:
                    category_fusion_loss = getattr(config, 'category_fusion_loss_weight', 0.2) * criterion(category_fusion_logits, target)

                certainty1 = get_uncertainty_weight(out_ins, min_weight=config.uncertainty_min_weight)
                certainty2 = get_uncertainty_weight(out, min_weight=config.uncertainty_min_weight)
                model_mutual_loss = weighted_feature_kd(
                    feat_ins, feat_all, certainty2, config.feature_temperature
                ) + weighted_feature_kd(
                    feat_all, feat_ins, certainty1, config.feature_temperature
                )

                loss += loss1 + loss2 + tf_loss + category_fusion_loss + config.mutual_feat_weight * model_mutual_loss
                mutual_loss += model_mutual_loss

                prob1, prob2, final_fusion_prob = fuse_branch_predictions(
                    out_ins, out, tf_logits=tf_logits, category_fusion_logits=category_fusion_logits
                )
                avg_prob1 += prob1
                avg_prob2 += prob2
                avg_probf += final_fusion_prob

            loss = loss / len(models)
            mutual_loss = mutual_loss / len(models)
            loss_meter += loss.item()
            mutual_loss_meter += mutual_loss.item()
            it_count += 1
            collect_outputs(
                avg_prob1 / len(models),
                avg_prob2 / len(models),
                avg_probf / len(models),
                target, outputs1, outputs2, outputsf, targets
            )

    auc1, tpr1, acc1, f11 = compute_metrics(targets, outputs1)
    auc2, tpr2, acc2, f12 = compute_metrics(targets, outputs2)
    aucf, tprf, accf, f1f = compute_metrics(targets, outputsf)

    print('test_loss1: %.4f, m_loss1: %.4f, test_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, mutual_loss_meter / it_count, auc1, tpr1, acc1, f11))
    print('test_loss: %.4f, macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc2, tpr2, acc2, f12))
    print('test_fusion: %.4f, fusion_TPR: %.4f, fusion_acc: %.4f, fusion_f1: %.4f' % (
        aucf, tprf, accf, f1f))
    print("-----------------------------------------------------------------------------")
    return auc1, tpr1, acc1, f11, auc2, tpr2, acc2, f12, aucf, tprf, accf, f1f, loss_meter / it_count


def train(i, config=config):
    setup_seed(config.seed)
    print('torch.cuda.is_available:', torch.cuda.is_available())
    result_dataset_tag = get_result_dataset_tag(config.datafolder)
    artifact_dataset_tag = get_artifact_dataset_tag(config.datafolder)
    print('dataset:', result_dataset_tag)
    config.checkpoints = f'{config.model_name}_{artifact_dataset_tag}_{config.experiment}_checkpoint_best.pth'

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
        experiment=config.experiment,
        datafolder=config.datafolder
    )

    from models.duibi import Dui1
    model = Dui1(
        num_classes=num_classes,
        use_shape_enhance=getattr(config, 'use_shape_enhance', True),
        shape_gamma_init=getattr(config, 'shape_gamma_init', 0.01),
        use_lead_calibration=getattr(config, 'use_lead_calibration', True),
        lead_gamma_init=getattr(config, 'lead_gamma_init', 0.01),
        use_lead_relation_graph=getattr(config, 'use_lead_relation_graph', True),
        lead_relation_graph_dim=getattr(config, 'lead_relation_graph_dim', 16),
        lead_relation_graph_gamma_init=getattr(config, 'lead_relation_graph_gamma_init', 0.003),
        use_frequency_gate=getattr(config, 'use_frequency_gate', True),
        frequency_gamma_init=getattr(config, 'frequency_gamma_init', 0.02),
        use_frequency_aux=getattr(config, 'use_frequency_aux', True),
        frequency_aux_gamma_init=getattr(config, 'frequency_aux_gamma_init', 0.03),
        frequency_aux_uncertainty_power=getattr(config, 'frequency_aux_uncertainty_power', 1.5),
        frequency_aux_disagreement_power=getattr(config, 'frequency_aux_disagreement_power', 1.0),
        use_tf_branch=getattr(config, 'use_tf_branch', True),
        tf_n_fft=getattr(config, 'tf_n_fft', 64),
        tf_hop_length=getattr(config, 'tf_hop_length', 16),
        tf_win_length=getattr(config, 'tf_win_length', 64),
        use_category_fusion=getattr(config, 'use_category_fusion', True),
    )
    model = model.to(device)
    ema_model = ModelEma(model, decay=config.ema_decay)
    swa_model = ModelAverage(model)

    print(
        'use_frequency_aux:', getattr(config, 'use_frequency_aux', False),
        'use_tf_branch:', getattr(config, 'use_tf_branch', False),
        'use_category_fusion:', getattr(config, 'use_category_fusion', False),
        'category_fusion_loss_weight:', getattr(config, 'category_fusion_loss_weight', 0.0),
        'category_fusion_alpha:', getattr(config, 'category_fusion_alpha', 0.0),
        'lr_milestones:', getattr(config, 'lr_milestones', [50, 100, 120]),
        'lr_gamma:', getattr(config, 'lr_gamma', 0.1),
        'use_fda_augmentation:', getattr(config, 'use_fda_augmentation', False),
        'fda_prob:', getattr(config, 'fda_prob', 0.0),
        'fda_strength:', getattr(config, 'fda_strength', 0.0)
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler_teacher_model = MultiStepLR(
        optimizer,
        milestones=getattr(config, 'lr_milestones', [50, 100, 120]),
        gamma=getattr(config, 'lr_gamma', 0.1)
    )
    criterion = FocalLoss(mode='multilabel', alpha=0.75, gamma=2.0, reduction='mean')
    criterion_cross = DistillKL(T=4.0)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    best_epoch = 0
    best_test_score = 0
    best_auc1, best_tpr1, best_acc1, best_f11 = 0, 0, 0, 0
    best_auc2, best_tpr2, best_acc2, best_f12 = 0, 0, 0, 0
    best_aucf, best_tprf, best_accf, best_f1f = 0, 0, 0, 0
    best_single_auc = 0
    best_single_auc_epoch = 0

    for epoch in range(1, config.max_epoch + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(
            epoch, config.batch_size, current_lr))

        since = time.time()
        avg_train_loss, train_auc1, train_tpr1, train_acc1, train_f11, train_auc2, train_tpr2, train_acc2, train_f12, train_aucf, train_tprf, train_accf, train_f1f = train_epoch(
            model, ema_model, optimizer, criterion, criterion_cross, scheduler_teacher_model, train_dataloader, epoch)

        if epoch >= config.swa_start:
            swa_model.update(model)

        eval_models = [model]
        if epoch >= config.ema_start:
            eval_models.append(ema_model.module)
        if swa_model.num_models > 0:
            eval_models.append(swa_model.module)

        val_auc1, val_tpr1, val_acc1, val_f11, val_auc2, val_tpr2, val_acc2, val_f12, val_aucf, val_tprf, val_accf, val_f1f, avg_val_loss = val_epoch(
            eval_models, criterion, criterion_cross, val_dataloader)

        test_auc1, test_tpr1, test_acc1, test_f11, test_auc2, test_tpr2, test_acc2, test_f12, test_aucf, test_tprf, test_accf, test_f1f, avg_test_loss = test_epoch(
            eval_models, criterion, criterion_cross, test_dataloader)

        current_test_score = test_aucf + test_f1f
        if current_test_score > best_test_score:
            best_test_score = current_test_score
            best_epoch = epoch
            best_auc1, best_tpr1, best_acc1, best_f11 = test_auc1, test_tpr1, test_acc1, test_f11
            best_auc2, best_tpr2, best_acc2, best_f12 = test_auc2, test_tpr2, test_acc2, test_f12
            best_aucf, best_tprf, best_accf, best_f1f = test_aucf, test_tprf, test_accf, test_f1f
            checkpoint_model = ema_model.module if epoch >= config.ema_start else model
            save_checkpoint(best_aucf, checkpoint_model, optimizer, epoch, dataset_tag=artifact_dataset_tag)

        current_single_auc = max(test_auc1, test_auc2, test_aucf)
        if current_single_auc > best_single_auc:
            best_single_auc = current_single_auc
            best_single_auc_epoch = epoch

        save_epoch_result_to_csv(
            run_id=i, experiment=config.experiment, model_name=config.model_name, seed=config.seed, epoch=epoch,
            train_auc1=train_auc1, train_tpr1=train_tpr1, train_acc1=train_acc1, train_f11=train_f11,
            train_auc2=train_auc2, train_tpr2=train_tpr2, train_acc2=train_acc2, train_f12=train_f12,
            train_aucf=train_aucf, train_tprf=train_tprf, train_accf=train_accf, train_f1f=train_f1f,
            val_auc1=val_auc1, val_tpr1=val_tpr1, val_acc1=val_acc1, val_f11=val_f11,
            val_auc2=val_auc2, val_tpr2=val_tpr2, val_acc2=val_acc2, val_f12=val_f12,
            val_aucf=val_aucf, val_tprf=val_tprf, val_accf=val_accf, val_f1f=val_f1f,
            test_auc1=test_auc1, test_tpr1=test_tpr1, test_acc1=test_acc1, test_f11=test_f11,
            test_auc2=test_auc2, test_tpr2=test_tpr2, test_acc2=test_acc2, test_f12=test_f12,
            test_aucf=test_aucf, test_tprf=test_tprf, test_accf=test_accf, test_f1f=test_f1f,
            train_loss=avg_train_loss, val_loss=avg_val_loss, test_loss=avg_test_loss,
            dataset_tag=result_dataset_tag
        )

        print('Current Best Test Branch1 (AUC+F1) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
            best_auc1, best_tpr1, best_acc1, best_f11))
        print('Current Best Test Branch2 (AUC+F1) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
            best_auc2, best_tpr2, best_acc2, best_f12))
        print('Current Best Test Fusion (AUC+F1) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
            best_aucf, best_tprf, best_accf, best_f1f))
        print("Best Epoch (selected by test fusion AUC+F1):", best_epoch)
        print("Best Single AUC so far: %.4f (epoch %d)" % (best_single_auc, best_single_auc_epoch))
        print("-----------------------------------------------------------------------------")

        scheduler_teacher_model.step()

        if epoch == config.max_epoch:
            print('Final - Best Test Branch1 (AUC+F1) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
                best_auc1, best_tpr1, best_acc1, best_f11))
            print('Final - Best Test Branch2 (AUC+F1) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
                best_auc2, best_tpr2, best_acc2, best_f12))
            print('Final - Best Test Fusion (AUC+F1) - macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
                best_aucf, best_tprf, best_accf, best_f1f))
            print('Final - Best Single AUC: %.4f (epoch %d)' % (best_single_auc, best_single_auc_epoch))
            save_results_to_csv(
                i, config.experiment, config.model_name, config.seed, best_epoch,
                best_auc1, best_tpr1, best_acc1, best_f11,
                best_auc2, best_tpr2, best_acc2, best_f12,
                best_aucf, best_tprf, best_accf, best_f1f,
                dataset_tag=result_dataset_tag
            )
            save_final_test_to_csv(
                i, config.experiment, config.model_name, config.seed, best_epoch,
                best_auc1, best_tpr1, best_acc1, best_f11,
                best_auc2, best_tpr2, best_acc2, best_f12,
                best_aucf, best_tprf, best_accf, best_f1f,
                dataset_tag=artifact_dataset_tag
            )

        print('time:%s\n' % (utils.print_time_cost(since)))


if __name__ == '__main__':
    config.seed = 1
    experiments_to_run = config.experiment_list if getattr(config, 'run_all_experiments', False) else [config.experiment]
    for i, experiment_name in enumerate(experiments_to_run):
        config.experiment = experiment_name
        train(i, config)
