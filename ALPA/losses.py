"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (based on https://github.com/HobbitLong/SupContrast)
    """

    def __init__(self, temperature=None):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def _compute_logits(self, features_a, features_b, attention=None):
        # global similarity
        if features_a.dim() == 2:
            features_a = F.normalize(features_a, dim=1, p=2)
            features_b = F.normalize(features_b, dim=1, p=2)
            contrast = torch.matmul(features_a, features_b.T)

        # spatial similarity
        elif features_a.dim() == 4:
            contrast = attention(features_a, features_b)

        else:
            raise ValueError

        # note here we use inverse temp
        contrast = contrast * self.temperature
        return contrast

    def forward(self, features_a, features_b=None, labels=None, attention=None):
        device = (torch.device('cuda') if features_a.is_cuda else torch.device('cpu'))
        num_features, num_labels = features_a.shape[0], labels.shape[0]

        # using only the current features in a given batch
        if features_b is None:
            features_b = features_a
            # mask to remove self contrasting
            logits_mask = (1. - torch.eye(num_features)).to(device)
        else:
            # contrasting different features (a & b), no need to mask the diagonal
            logits_mask = torch.ones(num_features, num_features).to(device)

        # mask to only maintain positives
        if labels is None:
            # standard self supervised case
            mask = torch.eye(num_labels, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

        # replicate the mask since the labels are just for N examples
        if num_features != num_labels:
            assert num_labels * 2 == num_features
            mask = mask.repeat(2, 2)

        # compute logits
        contrast = self._compute_logits(features_a, features_b, attention)

        # remove self contrasting
        mask = mask * logits_mask

        # normalization over number of positives
        normalization = mask.sum(1)
        normalization[normalization == 0] = 1.

        # for stability
        logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
        logits = contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        exp_logits = exp_logits * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / normalization
        loss = -mean_log_prob_pos.mean()

        return loss

