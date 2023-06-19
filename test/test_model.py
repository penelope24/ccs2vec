import unittest

import torch
from transformers import BartConfig
from model.transformers.funcs import *


def create_edge_index(N):
    edge_index = torch.randint(0, N, (2, N))
    return edge_index


class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_huggingface(self):
        config = BartConfig.from_pretrained('facebook/bart-base')
        print(config)

    def test_mask(self):
        tgt = torch.Tensor([[1,2,3,4],
                            [1,2,0,0]])
        print(tgt.shape)
        pad = 0
        tgt_mask = (tgt != pad).unsqueeze(-2)
        print(tgt_mask.shape)
        print(tgt_mask)
        tgt_subsequence_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        print(tgt_subsequence_mask.shape)
        final_mask = tgt_mask & tgt_subsequence_mask
        print(final_mask.shape)
        print(final_mask)

    def test_merge_edge_index(self):
        edge_index1 = create_edge_index(4)
        edge_index2 = create_edge_index(5)
        edge_index3 = create_edge_index(2)
        edge_indexes = [edge_index1, edge_index2, edge_index3]
        edge_indices = []
        for edge_index in edge_indexes:
            edge_indices.append(edge_index + len(edge_indices))
            # print(edge_index)
            # print(edge_index + len(edge_indices))
            # print("-------------")
        # print(edge_indices)
        edge_indices = torch.cat(edge_indices, dim=1)
        print(edge_indices.shape)
        mask = structure_mask(14, edge_indices)
        print(mask.shape)
        mask = torch.split(mask, edge_indices.size(1), dim=1)
        print(edge_indices.size())
        print(edge_indices.size(1))
