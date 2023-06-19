from enum import Enum
from typing import List, Union

import torch
from torch_geometric.data import Data

from data.vocab import special_tokens
from utils.tokenize import tokenize


class NodeType(Enum):
    CH = 1
    DATA_DEP = 2
    CTRL_DEP = 3
    default = 4


class EdgeType(Enum):
    DATA_FLOW = 1
    CTRL_FLOW = 2


class Node:

    def __init__(self, node: dict):
        if node is not None:
            self.id = node['id']
            self.local_id = -1
            self.type = self.parse_node_type(node)
            self.line = node['line']
            self.label = node['label']
            self.tokens = self.get_tokens(self.label)
            self.emb = None

    @staticmethod
    def parse_node_type(node: dict):
        if 'type' not in node.keys():
            return NodeType.default
        else:
            node_type = node['type']
            if node_type == "change":
                return NodeType.CH
            if node_type == "data_dep":
                return NodeType.DATA_DEP
            if node_type == "ctrl_dep":
                return NodeType.CTRL_DEP
            raise ValueError("not valid node type")

    @staticmethod
    def get_tokens(stmt):
        tokens = tokenize(stmt, special_tokens)
        return tokens

    # todo
    def init_emb(self):
        pass


class Edge:

    def __init__(self, edge: dict):
        self.id: int = edge['id']
        self.src: int = edge['source']
        self.tgt: int = edge['target']
        self.type = self.parse_type(edge)
        self.var = self.parse_var(edge)

    @staticmethod
    def parse_type(edge: dict):
        if 'label' not in edge.keys():
            raise ValueError("empty edge type")
        type_str = edge['label']
        if type_str == "control_flow":
            return EdgeType.CTRL_FLOW
        if type_str == "data_flow":
            return EdgeType.DATA_FLOW
        raise ValueError("not valid edge type")

    @staticmethod
    def parse_var(edge: dict):
        if 'var' in edge.keys():
            return edge['var']
        else:
            return None


class Graph:

    def __init__(self, m_id, nodes: List[Union[dict, Node]], edges: List[Union[dict, Edge]], stoi):
        self.m_id = m_id
        if isinstance(nodes[0], dict):
            self.nodes: List[Node] = [Node(n) for n in nodes]
        else:
            self.nodes = nodes
        if isinstance(edges[0], dict):
            self.edges: List[Edge] = [Edge(e) for e in edges]
        else:
            self.edges = edges
        self.stoi = stoi
        self.x = None
        self.edge_index = None
        self.edge_attr = None
        self.y = None
        self.graph = None
        self.parse()

    def parse(self):
        # x
        if self.stoi is None:
            self.x = torch.zeros(len(self.nodes), 1)
        else:
            features = []
            for node in self.nodes:
                tokens = node.tokens
                feature = [self.stoi[t] for t in tokens]
                features.append(feature)
            self.x = torch.Tensor(features)
        # edge_index
        s = []
        t = []
        for edge in self.edges:
            src = edge.src
            tgt = edge.tgt
            s.append(src)
            t.append(tgt)
        self.edge_index = torch.Tensor([s, t], dtype=torch.long)
        # edge_attr
        edge_attr = [e.type.value for e in self.edges]
        self.edge_attr = torch.Tensor(edge_attr)
        # pyg graph
        self.graph = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr)
    
    # fixme
    def get_sub_graph(self, edges: List[Union[dict, Edge]]):
        if isinstance(edges[0], dict):
            edges = [Edge(e) for e in edges]
        sub_nodes = []
        for e in edges:
            sub_nodes.append(e.src)
            sub_nodes.append(e.tgt)
        sub_nodes = list(set(sub_nodes))

        s = []
        t = []
        for edge in edges:
            src = edge.src
            tgt = edge.tgt
            s.append(src)
            t.append(tgt)
        sub_edge_index = torch.Tensor([s, t], dtype=torch.long)
        sub_data = self.graph.edge_subgraph(sub_edge_index)
        sub_data = sub_data.subgraph(torch.Tensor(sub_nodes))
        return sub_data


class SubGraph(Graph):
    
    def __init__(self, graph: Graph, edges: List[dict]):
        self.m_id = graph.m_id
        self.stoi = graph.stoi
        self.x = None
        self.edge_index = None
        self.edge_attr = None
        self.y = None
        self.graph = None
        super().__init__(self.m_id, graph.nodes, edges, self.stoi)
