import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# [N] --> [N, N]
def structure_mask(n: int, edge_index):
    mask = torch.ones(n, n)
    edge_indices = list(zip(edge_index[0], edge_index[1]))
    for n1, n2 in edge_indices:
        mask[n1, n2] = 0
        mask[n2, n1] = 0
    return mask


def structure_mask_batch(n_batched: int, batch_size: int,  edge_index_list):
    N = n_batched * batch_size
    edge_indices = []
    for edge in edge_index_list:
        edge_indices.append(edge + len(edge_indices))
    edge_index_large = torch.cat(edge_indices, dim=1)
    mask = structure_mask(N, edge_index_large)
    masks = torch.split(mask, n_batched, dim=1)
    return masks


def make_tgt_mask(tgt: torch.Tensor, pad: int):
    """mask: 1.padding; 2.future words"""
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_subsequence_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    final_mask = tgt_mask & tgt_subsequence_mask  # bit-wise AND
    return final_mask
