import random
from typing import Optional

import torch
from torch_geometric.data import Data
# from torch_sparse import coalesce


from data.diff import *

max_node_features_len = 24


def adjust_2d_array(arr, dim2=max_node_features_len):
    result = []
    for row in arr:
        if len(row) < dim2:
            row += [-1] * (dim2 - len(row))
        elif len(row) > dim2:
            row = row[:dim2]
        result.append(row)
    return result


def mask_nodes(data, mask_ratio):
    num_nodes = data.num_nodes
    num_mask_nodes = int(num_nodes * mask_ratio)

    # 随机选择要遮掩的节点索引
    mask_indices = random.sample(range(num_nodes), num_mask_nodes)

    # 对 data.x 进行遮掩
    data.x[mask_indices] = 0.0

    # 对 data.edge_index 进行调整
    mask_mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask_mask[mask_indices] = 1
    mask_mask = mask_mask[data.edge_index[0]]
    data.edge_index = data.edge_index[:, ~mask_mask]

    # 对 data.edge_attr 进行调整（如果存在）
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[~mask_mask]

    # 重新编制索引，确保索引的连续性
    # data.edge_index, _ = coalesce(data.edge_index, None, num_nodes, num_nodes)

    return data


# todo : 1. extend node 2. y
class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge], stoi: dict):
        self.nodes = nodes
        self.edges = edges
        self.id_map = self.get_local_node_id_map()
        self.node_feature_matrix: Optional[torch.Tensor] = self.get_node_feature_mat(stoi)
        self.edge_index: Optional[torch.LongTensor] = self.get_edge_index_mat()
        self.edge_attr: Optional[torch.Tensor] = self.get_edge_attr_mat(stoi)
        self.y: Optional[torch.Tensor]
        self.pos: Optional[torch.Tensor]
        self.data = self.to_pyg_graph()

    @classmethod
    def create(cls, nodes: List[Node], edges: List[Edge], stoi: dict):
        g = cls(nodes, edges, stoi)
        return g.data

    def get_local_node_id_map(self):
        d = {}
        idx = 0
        for node in self.nodes:
            d[node.id] = idx
            idx += 1
        return d

    def get_node_feature_mat(self, stoi: dict) -> Optional[torch.Tensor]:
        features = []
        for node in self.nodes:
            tokens = node.get_tokens()
            indices = [stoi[x] for x in tokens]
            features.append(indices)
        adjusted_features = adjust_2d_array(features)
        mat = torch.Tensor(adjusted_features)
        return mat

    def get_edge_index_mat(self):
        row = []
        col = []
        for edge in self.edges:
            src, tgt = edge.src, edge.tgt
            src = self.id_map[src]
            tgt = self.id_map[tgt]
            row.append(src)
            col.append(tgt)
        edge_index = torch.LongTensor([row, col])
        return edge_index

    def get_edge_attr_mat(self, stoi: dict):
        edge_features = []
        for edge in self.edges:
            type = 0 if edge.type == "control_flow" else 1
            var = stoi[edge.var] if edge.var else -1
            edge_features.append([type, var])
        mat = torch.Tensor(edge_features)
        return mat

    def to_pyg_graph(self):
        return Data(x=self.node_feature_matrix, edge_index=self.edge_index, edge_attr=self.edge_attr)


class NodeMaskGraph(Graph):

    def __iter__(self, nodes: List[Node], edges: List[Edge], stoi: dict):
        super().__init__(nodes, edges, stoi)

