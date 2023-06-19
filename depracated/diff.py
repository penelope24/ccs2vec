from typing import List


class Node:

    def __init__(self, node: dict):
        self.id: int = node['id']
        self.label: str = node['label']
        self.line: int = node['line']


class Edge:

    def __init__(self, edge: dict):
        self.id: int = edge['id']
        self.type: str = edge['type']
        self.src: int = edge['src']
        self.tgt: int = edge['tgt']
        self.var: str = edge['var']


class Slice:

    def __init__(self, slice: dict, graphs):
        self.nodes: List[Node]
        self.links: List[Edge]



class CodeChange:

    def __init__(self, slice1, slice2, id1, id2):
        self.method_id1: int = id1
        self.method_id2: int = id2
        self.slice1: Slice = slice1
        self.slice2: Slice = slice2


class PDG:

    def __init__(self, nodes, links, id):
        self.method_id: int = id
        self.nodes: List[Node] = nodes
        self.links: List[Edge] = links


class CommitDiff:

    def __init__(self, graphs1: List[dict], graphs2: List[dict], changes: List[dict]):
        self.graphs1: List[PDG] = CommitDiff.parse_graphs(graphs1)
        self.graphs2: List[PDG] = CommitDiff.parse_graphs(graphs2)
        self.changes: List[CodeChange]

    @staticmethod
    def parse_graphs(graphs: List[dict]):
        res: List[PDG] = []
        for graph in graphs:
            nodes = [Node(n) for n in graph['nodes']]
            links = [Edge(e) for e in graph['edges']]
            id = graph['method_id']
            res.append(PDG(nodes, links, id))
        return res



