from typing import Optional, List

from torch_geometric.data.data import Data
from graphs import *


class Seq:

    def __init__(self, graph: Graph):
        self.graph = graph

    @staticmethod
    def gen_stmt_token(idx: int):
        st = ""
        st += "<"
        st += "s"
        st += str(idx)
        st += ">"
        return st

    @staticmethod
    def gen(graph: Graph):
        ss = []
        # round 1 for ch nodes
        for i in range(graph.x.size(0)):
            ft = graph.x[i]
            s_type = ft[0].item()
            if s_type == NodeType.CH.value:
                st = Seq.gen_stmt_token(i)
                ss.append(st)
                token_ids = [t.item() for t in ft[1:]]
                for id in token_ids:
                    ss.append(str(id))
        ss.append("[SEP]")
        # round 2 for data_dep dep nodes
        for i in range(graph.x.size(0)):
            ft = graph.x[i]
            s_type = ft[0].item()
            if s_type == NodeType.DATA_DEP.value:
                st = Seq.gen_stmt_token(i)
                ss.append(st)
                token_ids = [t.item() for t in ft[1:]]
                for id in token_ids:
                    ss.append(str(id))
        ss.append("[SEP]")
        # round 3 for ctrl dep nodes
        for i in range(graph.x.size(0)):
            ft = graph.x[i]
            s_type = ft[0].item()
            if s_type == NodeType.CTRL_DEP.value:
                st = Seq.gen_stmt_token(i)
                ss.append(st)
                token_ids = [t.item() for t in ft[1:]]
                for id in token_ids:
                    ss.append(str(id))
        return " ".join(ss)