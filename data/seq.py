import random
from typing import Optional
from enum import Enum

import torch

from data.diff import *


class StmtType(Enum):
    CH = 1
    DATA_DEP = 2
    CTRL_DEP = 3
    default = 4


class StmtLinkType(Enum):
    DATA_FLOW = 1
    CTRL_FLOW = 2


class Stmt:

    def __init__(self, node: Node, id: int, type: StmtType):
        self.id = id
        self.raw_id = node.id
        self.type = type
        self.token = None
        self.seq = []
        self.parse(node)

    def parse(self, node: Node):
        self.token = "[" + str(self.id) + "]"
        self.seq.append(self.token)
        self.seq.extend(node.get_tokens())


def parse_stmts(nodes: List[Node]):
    def custom_sort_key(node):
        if node.line >= 0:
            return node.line, 0
        else:
            return float('inf'), 1

    stmts = []
    sorted_nodes = sorted(nodes, key=custom_sort_key)
    for node in sorted_nodes:
        id = len(stmts)
        if node.is_ch_node:
            stmt_type = StmtType.CH
        elif node.is_data_dep_node:
            stmt_type = StmtType.DATA_DEP
        elif node.is_ctrl_dep_node:
            stmt_type = StmtType.CTRL_DEP
        else:
            stmt_type = StmtType.default
        stmt = Stmt(node, id, stmt_type)
        stmts.append(stmt)
    return stmts


def find_stmt(node_id: int, stmts: List[Stmt]):
    res = [s for s in stmts if s.raw_id == node_id]
    if res:
        return res[0]
    else:
        return None


def get_edge_index_mat(edges: List[Edge], stmts: List[Stmt]):
    row = []
    col = []
    for edge in edges:
        src = find_stmt(edge.src, stmts)
        tgt = find_stmt(edge.tgt, stmts)
        row.append(src.id)
        col.append(tgt.id)
    edge_index = torch.LongTensor([row, col])
    return edge_index


class Sequence:

    def __init__(self, slice: Slice):
        self.node_idx_map = {}
        self.cls = '[CLS]'
        self.seq = '[SEP]'
        self.max_line_num = 24
        self.ch_stmts = []
        self.data_dep_stmts = []
        self.ctrl_dep_stmts = []
        self.default_stmts = []
        # parse res
        self.text: List[str]
        self.data_edge_index = []
        self.ctrl_edge_index = []
        self.parse(slice)

    def parse(self, slice: Slice):
        stmts = parse_stmts(slice.nodes)
        self.ch_stmts = [s for s in stmts if s.type == StmtType.CH]
        self.data_dep_stmts = [s for s in stmts if s.type == StmtType.DATA_DEP]
        self.ctrl_dep_stmts = [s for s in stmts if s.type == StmtType.CTRL_DEP]
        self.default_stmts = [s for s in stmts if s.type == StmtType.default]
        data_flow_edges = [e for e in slice.edges if e.type == "data_flow"]
        ctrl_flow_edges = [e for e in slice.edges if e.type == "control_flow"]
        self.data_edge_index = get_edge_index_mat(data_flow_edges, stmts)
        self.ctrl_edge_index = get_edge_index_mat(ctrl_flow_edges, stmts)

    def generate(self):
        ch_str_list = []
        for s in self.default_stmts:
            ch_str_list.extend(s.seq)
        print(ch_str_list)
        ch_str = " ".join(ch_str_list)
        print(ch_str)
