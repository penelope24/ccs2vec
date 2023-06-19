import json
import unittest
from graphs import *
from dataset import *


class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = "/Users/fy/Documents/workspace/ccs2vec/samples"

    def test_dataset(self):
        ds = SliceDataset(self.base_path)
        for data in ds:
            print(data)

    def test_sub_graph(self):
        import torch
        from torch_geometric.data import Data
        from torch_geometric.utils import subgraph

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6]])
        edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        subset = torch.tensor([3, 4, 5])
        sub_edge_set = torch.Tensor([[1, 3]],
                                    [0, 2])
        subgraph(subset, edge_index1, edge_attr)