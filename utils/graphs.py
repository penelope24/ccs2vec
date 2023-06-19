import json

import networkx as nx
from networkx.drawing.nx_pydot import read_dot


class PDG:
    """
    a wrapper class of networkx.MultiDiGraph() for ease of use.
    """

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    @staticmethod
    def from_json_file(path):
        with open(path) as f:
            d = json.load(f)
            graph = nx.readwrite.node_link_graph(d)
            for k, v in d.items():
                graph.graph[k] = v
            return PDG(graph)

    @staticmethod
    def from_dot_file(path):
        graph = read_dot(path)
        return PDG(graph)

    @staticmethod
    def to_nx_graph(path):
        pass

    @staticmethod
    def create_empty():
        return PDG(nx.MultiDiGraph())

    def non_empty(self):
        return self.graph is not None

    def get_nodes(self):
        return self.graph.nodes

    def add_node(self, node):
        self.graph.add_node(node)

    def get_edges(self):
        return self.graph.edges

    def add_edge(self, edge):
        self.graph.add_edge(edge)

    def get_graph_properties(self):
        return self.graph.graph

    def get_graph_property(self, key):
        return self.graph.graph[key]

    def get_node_properties(self, node):
        return self.graph.nodes[node]

    def get_node_property(self, node, key):
        return self.graph.nodes[node][key]

    def get_edge_properties(self, edge):
        return self.edges[edge]

    def get_edge_property(self, edge, key):
        return self.graph.edges[edge][key]

    def get_node_features(self, key):
        return self.graph.nodes.data(key)

    # fixme
    def get_subgraph_view(self, idx: int, key: str):
        print("***")
        return nx.subgraph_view(self.graph,
                                filter_node=lambda n: idx in self.get_node_property(n, key),
                                filter_edge=lambda e: idx in self.get_edge_property(e, key))

    # def get_slice_by_index(self, idx: int):
    #
    #     def node_filter(n):
    #         return idx in self.get_node_property(n, "slices")
    #
    #     def edge_filter(e):
    #         return idx in self.get_edge_property(e, "slices")
    #
    #     return nx.subgraph_view(self.graph, filter_node=node_filter, filter_edge=edge_filter)
    #
    # def get_path_by_index(self, idx: int):
    #     def node_filter(n):
    #         return idx in self.get_node_property(n, "paths")
    #
    #     def edge_filter(e):
    #         return idx in self.get_edge_property(e, "paths")
    #
    #     return nx.subgraph_view(self.graph, filter_node=node_filter, filter_edge=edge_filter)

    def get_slices_with_paths(self):
        slices = []
        n = self.get_graph_property("slice_num")
        d: dict = self.get_graph_property("mapping")
        for i in range(n):
            slice = self.get_subgraph_view(i, "slices")
            paths = []
            key = "slice" + str(i)
            slice_indexes = d.get(key)
            if slice_indexes is not None:
                paths = [self.get_subgraph_view(x, "paths") for x in slice_indexes]

            slices.append(slice)
        return slices





class DiffSlice(PDG):

    def __init__(self, graph: nx.MultiDiGraph, idx: int):
        super(DiffSlice, self).__init__(graph)
        self.cfg_paths = []
        self.index = idx

    def add_paths(self, cfg_paths: []):
        self.cfg_paths.extend(cfg_paths)

    def get_paths(self):
        return self.cfg_paths

    def get_index(self):
        return self.index


class CFGPath(PDG):

    def __init__(self, graph: nx.MultiDiGraph, idx: int):
        super(CFGPath, self).__init__(graph)
        self.index = idx

    def get_index(self):
        return self.index
