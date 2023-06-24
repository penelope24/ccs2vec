from enum import Enum
from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes

from data.seq import Seq


class NodeType(Enum):
    CH = 1
    DATA_DEP = 2
    CTRL_DEP = 3
    default = 4


class EdgeType(Enum):
    DATA_FLOW = 1
    CTRL_FLOW = 2


def parse_node_type(node: dict) -> NodeType:
    if "type" not in node.keys():
        return NodeType.default
    label = node['label']
    if label == "change":
        return NodeType.CH
    if label == "data_dep":
        return NodeType.DATA_DEP
    if label == "ctrl_dep":
        return NodeType.CTRL_DEP
    raise ValueError("not a valid node type.")

def parse_edge_type(edge: dict) -> EdgeType:
    if "label" not in edge.keys():
        raise ValueError("not a valid edge type")
    label = edge['label']
    if label == "control_flow":
        return EdgeType.CTRL_FLOW
    if label == "data_flow":
        return EdgeType.DATA_FLOW
    raise ValueError("not a valid edge type")


class Graph:

    def __init__(self, nodes: List[dict], edges: List[dict], tokenize, stoi, m_id):
        # graph structure
        self.x = None
        self.edge_index = None
        self.edge_attr = None
        # additional
        self.tokenize = tokenize
        self.stoi = stoi
        self.method_id = m_id
        self.seq = None
        # do parse
        self.parse(nodes, edges, tokenize, stoi)

    @staticmethod
    def get_x_features(nodes: List[dict], tokenize, stoi):
        features = []
        for node in nodes:
            ft = []
            t = parse_node_type(node).value
            tokens = tokenize(node)
            token_ids = [stoi[t] for t in tokens]
            ft.append(t)
            for token_id in token_ids:
                ft.append(token_id)
            features.append(ft)
        return torch.Tensor(features)

    @staticmethod
    def get_edge_index_mat(edges: List[dict]):
        row = []
        col = []
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            if src is not None and tgt is not None:
                row.append(src)
                col.append(tgt)
        return torch.LongTensor([row, col])

    @staticmethod
    def get_edge_attr_mat(edges: List[dict]):
        attrs = []
        for edge in edges:
            try:
                attr = parse_edge_type(edge).value
                attrs.append(attr)
            except ValueError:
                print("edge type error")
        return torch.Tensor(attrs)

    def parse(self, nodes: List[dict], edges: List[dict], tokenize, stoi):
        self.x = self.get_x_features(nodes, tokenize, stoi)
        self.edge_index = self.get_edge_index_mat(edges)
        self.edge_attr = self.get_edge_attr_mat(edges)

    def get_pyg_data(self):
        return Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def get_sub_edge_graph(self, sub_edges: List[dict]) -> Data:
        sub_edge_index = self.get_edge_index_mat(sub_edges)
        sub_edge_attr = self.get_edge_attr_mat(sub_edges)
        edge_index, edge_attr, mask = remove_isolated_nodes(sub_edge_index, sub_edge_attr, num_nodes=self.x.size(dim=1))
        return Data(x=self.x * mask, edge_index=edge_index, edge_attr=edge_attr)

    def get_seq(self):
        text = Seq.gen(self)
        return text

