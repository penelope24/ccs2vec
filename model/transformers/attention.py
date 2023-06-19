import math

import torch.nn.functional as F

from .funcs import *

"""todo: 如果你希望在注意力计算的归一化中采用更复杂的操作，以下是一些具体的建议： 
1. 使用自定义归一化函数：除了标准的 softmax 归一化，你可以考虑使用其他自定义的归一化函数。例如，你可以尝试使用自适应归一化（Adaptive Normalization）或自注意力归一化（Self-Attention 
Normalization）等方法。这些方法可以根据具体任务和模型的特点进行定制，以更好地捕捉注意力权重之间的关系。 
2.考虑缩放因子：除了 softmax 归一化，你可以引入一个缩放因子来调整注意力权重的范围。这个缩放因子可以是固定的常数，也可以是可学习的参数。通过调整缩放因子，可以控制注意力权重的相对大小，以适应不同的任务需求。 
3. 使用归一化策略：你可以考虑在注意力计算中引入一些归一化策略，例如层归一化（Layer Normalization）或批归一化（Batch Normalization）。这些归一化策略可以帮助提高模型的稳定性和泛化性能。 
4. 结合其他信息进行归一化：除了仅使用注意力权重本身进行归一化，你还可以考虑结合其他信息进行归一化。例如，你可以使用注意力权重与输入特征之间的相关性进行加权归一化，以更好地反映特征之间的重要性。"""


def attention(query, key, value, mask=None, dropout=None, neg_value=float('-inf')):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, neg_value)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
