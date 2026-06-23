# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F


class EvidenceReliabilityHead(nn.Module):
    """Convert logits or pooled features into evidence, opinion and reliability.

    For single-label tasks this uses a K-class Dirichlet opinion. For multi-label
    tasks it uses an independent Beta opinion per class, avoiding a simplex
    assumption across labels.
    """

    def __init__(self, num_classes, input_dim=None, mode='dirichlet'):
        super().__init__()
        self.num_classes = int(num_classes)
        self.mode = str(mode).lower()
        if self.mode not in ('dirichlet', 'beta'):
            raise ValueError(f'Unsupported evidence mode: {mode}')
        self.output_dim = self.num_classes if self.mode == 'dirichlet' else self.num_classes * 2
        self.proj = None
        if input_dim is not None and int(input_dim) != self.output_dim:
            self.proj = nn.Linear(int(input_dim), self.output_dim)

    def forward(self, x):
        logits = self.proj(x) if self.proj is not None else x
        if self.mode == 'dirichlet':
            return self._dirichlet_from_logits(logits)
        return self._beta_from_logits(logits)

    def _dirichlet_from_logits(self, logits):
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        total_evidence = alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        uncertainty = (float(self.num_classes) / total_evidence).clamp(0.0, 1.0)
        reliability = (1.0 - uncertainty).clamp(0.0, 1.0)
        prob = alpha / total_evidence
        belief = evidence / total_evidence
        return {
            'evidence': evidence,
            'alpha': alpha,
            'total_evidence': total_evidence,
            'uncertainty': uncertainty,
            'reliability': reliability,
            'prob': prob,
            'belief': belief,
        }

    def _beta_from_logits(self, logits):
        if logits.size(1) == self.num_classes:
            e_pos = F.softplus(logits)
            e_neg = F.softplus(-logits)
        elif logits.size(1) == self.num_classes * 2:
            e_pos, e_neg = torch.chunk(F.softplus(logits), 2, dim=1)
        else:
            raise ValueError(
                f'Beta evidence expects {self.num_classes} logits or '
                f'{self.num_classes * 2} projected values, got {logits.size(1)}.'
            )
        alpha_pos = e_pos + 1.0
        alpha_neg = e_neg + 1.0
        total_evidence = (alpha_pos + alpha_neg).clamp_min(1e-6)
        uncertainty = (2.0 / total_evidence).clamp(0.0, 1.0)
        reliability = (1.0 - uncertainty).clamp(0.0, 1.0)
        prob_pos = alpha_pos / total_evidence
        return {
            'evidence': torch.stack((e_pos, e_neg), dim=-1),
            'evidence_pos': e_pos,
            'evidence_neg': e_neg,
            'alpha_pos': alpha_pos,
            'alpha_neg': alpha_neg,
            'total_evidence': total_evidence,
            'uncertainty': uncertainty,
            'reliability': reliability,
            'prob': prob_pos,
        }
