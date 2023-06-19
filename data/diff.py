from typing import List

from torch_geometric.data import Data

from utils.tokenize import tokenize
from data.vocab import special_tokens


def get_node_by_id(nodes, id):
    res = [x for x in nodes if x.id == id]
    if res:
        return res[0]
    else:
        return None


def find_graph(hunk: dict, graphs: List[dict], v: str):
    if v == "v1":
        method_id = hunk['method_id1']
    else:
        method_id = hunk['method_id2']
    graph = get_graph_by_id(graphs, method_id)
    return graph


def get_graph_by_id(graphs: List[dict], id: int):
    res = [x for x in graphs if x['method_id'] == id]
    if res:
        return res[0]
    else:
        return None


class Node:

    def __init__(self, node: dict):
        if node is not None:
            self.id: int = node['id']
            self.label: str = node['label']
            self.line: int = node['line']
            self.is_ch_node = False
            self.is_data_dep_node = False
            self.is_ctrl_dep_node = False

    def get_tokens(self):
        stmt = self.label
        tokens = tokenize(stmt, special_tokens)
        return tokens


class Edge:

    def __init__(self, edge: dict):
        self.id: int = edge['id']
        self.type: str = edge['label']
        self.src: int = edge['source']
        self.tgt: int = edge['target']
        self.var: str = edge['var'] if 'var' in edge.keys() else None


# class Path:
#
#     def __init__(self, path: dict, nodes: List[Node]):
#         self.nodes: List[Node] = []
#         self.edges: List[Edge] = []
#         self.parse(path, nodes)
#
#     def parse(self, path: dict, nodes: List[Node]):
#         # edges
#         self.edges = [Edge(e) for e in path['edges']]
#         # nodes
#         for edge in self.edges:
#             src_node = get_node_by_id(nodes, edge.src)
#             tgt_node = get_node_by_id(nodes, edge.tgt)
#             assert src_node is not None
#             assert tgt_node is not None
#             if src_node not in self.nodes:
#                 self.nodes.append(src_node)
#             if tgt_node not in self.nodes:
#                 self.nodes.append(tgt_node)
#
#
# class Slice:
#
#     def __init__(self, slice: dict, nodes: List[Node]):
#         self.nodes: List[Node] = []
#         self.edges: List[Edge] = []
#         self.paths: List[Path] = []
#         self.parse(slice, nodes)
#
#     def parse(self, slice: dict, nodes: List[Node]):
#         # edges
#         self.edges = [Edge(e) for e in slice['edges']]
#         # nodes
#         for edge in self.edges:
#             src = get_node_by_id(nodes, edge.src)
#             tgt = get_node_by_id(nodes, edge.tgt)
#             assert src is not None
#             assert tgt is not None
#             if src not in self.nodes:
#                 self.nodes.append(src)
#             if tgt not in self.nodes:
#                 self.nodes.append(tgt)
#         # paths
#         for d in slice['paths']:
#             path = Path(d, self.nodes)
#             self.paths.append(path)
#
#
# class SliceVirtualGraph:
#
#     def __init__(self, slice: Slice):
#         self.nodes: List[Node] = []
#         self.edges: List[Edge] = []
#         self.parse(slice)
#
#     def parse(self, slice: Slice):
#         self.nodes.extend(slice.nodes)
#         # slice
#         slice_node_id = len(self.nodes)
#         slice_node = Node({
#             "id": slice_node_id,
#             "label": "[SLICE]",
#             "line": 0
#         })
#         self.nodes.append(slice_node)
#         # paths
#         for path in slice.paths:
#             path_node_id = len(self.nodes)
#             path_node = Node({
#                 "id": path_node_id,
#                 "label": "[PATH]",
#                 "line": 0
#             })
#             self.nodes.append(path_node)
#             slice_path_edge = Edge({
#                 "id": len(self.edges) + 1,
#                 "label": "contain_path",
#                 "source": slice_node_id,
#                 "target": path_node_id
#             })
#             self.edges.append(slice_path_edge)
#             for node in path.nodes:
#                 path_node_edge = Edge({
#                     "id": len(self.edges) + 1,
#                     "label": "contain_node",
#                     "source": path_node_id,
#                     "target": node.id
#                 })
#                 self.edges.append(path_node_edge)


class Change:

    def __init__(self, hunk: dict, g1: dict, g2: dict):
        self.graph1: Data = Data()
        self.graph2: Data = Data()
        self.slice1 = None
        self.slice2 = None
        self.vg1 = None
        self.vg2 = None
        self.parse(hunk, g1, g2)

    def parse(self, hunk: dict, g1: dict, g2: dict):
        pass


class File:

    def __init__(self, file: dict, file_id: int):
        self.file_name = file['file_name']
        self.file_id = file_id
        self.total_change_line_num = None
        self.total_add_line_num = None
        self.total_rmv_line_num = None
        self.changes = []
        self.tokens = []
        self.parse(file)

    def parse(self, file: dict):
        hunks = file['hunks']
        graphs1 = file['graphs1']
        graphs2 = file['graphs2']
        for g in graphs1:
            for node in g['nodes']:
                stmt_str = node['label']
                tokens = tokenize(stmt_str, special_tokens)
                self.tokens.extend(tokens)
        for g in graphs2:
            for node in g['nodes']:
                stmt_str = node['label']
                tokens = tokenize(stmt_str, special_tokens)
                self.tokens.extend(tokens)
        for hunk in hunks:
            g1 = find_graph(hunk, graphs1, "v1")
            g2 = find_graph(hunk, graphs2, "v2")
            self.changes.append(Change(hunk, g1, g2))


class Commit:

    def __init__(self, commit: dict):
        self.commit_id = None
        self.commit_msg = None
        self.files: List[File] = []
        self.parse(commit)

    def parse(self, commit: dict):
        self.commit_id = commit['commit_id']
        self.commit_msg = commit['msg']
        files: List = commit['files']
        for file in files:
            idx = files.index(file)
            self.files.append(File(file, idx))
