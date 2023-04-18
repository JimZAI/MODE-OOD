import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np

def projector(dim, projection_size):
    """
    projection head
    """
    return nn.Sequential(
        nn.Linear(dim, dim, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(dim, projection_size, bias=False)
        )

class AttentionSimilarity(nn.Module):
    """
    Attention alignment & spatial similarity module
    """
    def __init__(self, hidden_size, inner_size=None, drop_prob=0.0):
        super(AttentionSimilarity, self).__init__()

        self.hidden_size = hidden_size
        self.inner_size = inner_size if inner_size is not None else hidden_size//8

        self.query = projector(self.hidden_size, self.inner_size)
        self.key = projector(self.hidden_size, self.inner_size)
        self.value = projector(self.hidden_size, self.inner_size)
        self.dropout = nn.Dropout(drop_prob)

    def contrast_a_with_b(self, query_a, key_a, value_a, query_b, key_b, value_b):
        "comparing features_a to features_b"

        # 1) spatial alignement
        value_a = value_a.unsqueeze(0)
        value_b = value_b.unsqueeze(1)

        # Align features A
        att_scores = torch.matmul(query_b.unsqueeze(1), key_a.unsqueeze(0).transpose(-1, -2).contiguous())
        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
        att_probs = self.dropout(att_probs)

        # (BatchA x BatchB x HW x HW) x (BatchA x 1 x HW x C) -> (BatchA x BatchB x HW x C)
        aligned_features_a = torch.matmul(att_probs, value_a)

        # Align features B
        att_scores = torch.matmul(query_a.unsqueeze(0), key_b.unsqueeze(1).transpose(-1, -2).contiguous())
        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
        att_probs = self.dropout(att_probs)

        # (BatchA x BatchB x HW x HW) x (1 x BatchB x HW x C) -> (BatchA x BatchB x HW x C)
        aligned_features_b = torch.matmul(att_probs, value_b)

        assert aligned_features_a.size(-1) == self.inner_size
        assert aligned_features_b.size(-1) == self.inner_size
        assert value_a.size(-1) == self.inner_size
        assert value_b.size(-1) == self.inner_size

        # 2) compute the spatial similarity
        similarity = nn.CosineSimilarity(dim=-1)(value_a, aligned_features_b)
        similarity = similarity + nn.CosineSimilarity(dim=-1)(value_b, aligned_features_a)
        output = similarity.mean(-1)
        return output

    def forward(self, features_a, features_b):
        # projection of features a
        features_a = features_a.view(features_a.size(0), features_a.size(1), -1).permute(0, 2, 1).contiguous()
        query_a = self.query(features_a)
        key_a = self.key(features_a)
        value_a = self.value(features_a)

        if features_b is None:
            att_scores = query_a * key_a
            att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
            aligned_features = att_probs * value_a
            output = aligned_features.mean(-2)
        else:
            features_b = features_b.view(features_b.size(0), features_b.size(1), -1).permute(0, 2, 1).contiguous()
            query_b = self.query(features_b)
            key_b = self.key(features_b)
            value_b = self.value(features_b)
            output = self.contrast_a_with_b(query_a, key_a, value_a, query_b, key_b, value_b)

        return output
