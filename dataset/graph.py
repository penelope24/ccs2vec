import logging
import random

import networkx as nx

logger = logging.getLogger('ccs2vec')
__author__ = 'zfy'
LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


class Graph:

    """
    basic implementation & extension of networkx.MultiDiGraph()
    """
    def __init__(self, graph):
        self.graph = graph

    def non_empty(self):
        return self.graph is not None

    def nodes(self):
        """return all node id of the dataset"""
        return list(self.graph.nodes)

    def edges(self):
        """return all edges of a dataset"""
        return list(self.graph.edges)

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

    def get_adj_node_dict(self):
        """
        get the adjacent list of a multi-directed-dataset
        note that if there are multi lines between two nodes, only one is recorded in a adj_list
        e.g. (0, 1, color='red) and (0, 1, color='blue') --> 0: 1
        TODO: will multi-lines affect random walk?
        :return: dict -> {node : its adj_list}
        """
        adj_dict = {}
        for node, neibour in self.graph.adjacency():
            adj_dict.setdefault(node, [])
            for v in neibour:
                adj_dict[node].append(v)
        return adj_dict

    def remove_self_loop(self):
        """
        remove self-loop in dataset
        e.g. (1 -> 1)
        :return refined adjacent dict
        """
        removed_num = 0
        adjacent = self.get_adj_node_dict()
        for node, neibour in adjacent:
            if node in neibour:
                neibour.remove(node)
                removed_num += 1
        logger.info('remove self-loops: removed {} self loops'.format(removed_num))
        return adjacent

    def check_self_loop(self):
        """
        check if a dataset has any self-loop
        :return: true or false
        """
        adjacent = self.get_adj_node_dict()
        for node, neibour in adjacent:
            if node in neibour:
                return True
        return False

    def out_edges(self, node):
        """return all out edges of a node"""
        return list(self.graph.out_edges(node))

    def in_edges(self, node):
        """return all out edges of a node"""
        return list(self.graph.in_edges(node))

    def out_edges_with_label(self, node):
        """return all out edges in 3-tuple form"""
        out_edges_3_form = []
        out_edges = self.out_edges(node)
        for edge in out_edges:
            s = edge[0]
            t = edge[1]
            multi = self.graph[s][t]
            for i in range(len(multi)):
                label = self.graph[s][t][i]['label']
                out_edges_3_form.append((s,t,label))
        return out_edges_3_form

    def in_edges_with_label(self, node):
        """return all in edges in 3-tuple form"""
        in_edges_3_form = []
        in_edges = self.in_edges(node)
        for edge in in_edges:
            s = edge[0]
            t = edge[1]
            multi = self.graph[s][t]
            for i in range(len(multi)):
                # print('i: ' + str(i))
                # print(s, t)
                label = self.graph[s][t][i]['label']
                in_edges_3_form.append((s, t, label))
        return in_edges_3_form

    def random_walk_adjacent(self, path_length, alpha=0, rand=random.Random(), start=None):
        """
        returns a truncated random walk.
        :param path_length: length of a random walk
        :param alpha: probability of restart
        :param rand: random seed
        :param start: the start node of random walk
        :return: a generated random walk
        """
        if start:
            path = [start]
        else:
            # sampling nodes not edges
            path = [rand.choice(self.nodes())]

        while (len(path) < path_length):
            cur = path[-1]
            cur_neibour = self.get_adj_node_dict().get(cur)
            if len(cur_neibour) > 0:
                if rand.random() > alpha:
                    path.append(rand.choice(cur_neibour))
                else:
                    path.append(path[0])
            else:
                # this is how we truncate the walk
                break
        return [self.get_node_property(node, "label") for node in path]

    # fixme
    def get_slice_view(self, idx: int):
        return nx.subgraph_view(self.graph,
                                filter_node=lambda n: idx in self.get_node_property(n, "slices"),
                                filter_edge=lambda e: idx in self.get_edge_property(e, "slices"))

    def get_slice(self, idx: int):
        filtered_nodes = []
        filtered_edges = []
        for node in self.nodes():
            if idx in self.graph.nodes[node]["slices"]:
                filtered_nodes.append(node)
        for edge in self.edges():
            if idx in self.graph.edges[edge]["slices"]:
                filtered_edges.append(edge)
        # new dataset
        G = nx.MultiDiGraph()
        G.add_nodes_from(filtered_nodes)
        G.add_edges_from(filtered_edges)
        return G
