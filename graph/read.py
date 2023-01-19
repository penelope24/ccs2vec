import os
import json
import networkx as nx
from .graphs import Graph


def read_json(path):
    with open(path) as f:
        d = json.load(f)
        graph = nx.readwrite.node_link_graph(d)
        for k, v in d.items():
            graph.graph[k] = v
        return graph


def list_atom_dirs(path):
    res = []
    projects = [os.path.join(path, x) for x in os.listdir(path) if not x.startswith(".")]
    for project in projects:
        commits = [os.path.join(project, x) for x in os.listdir(project) if not x.startswith(".")]
        for commit in commits:
            methods = [os.path.join(commit, x) for x in os.listdir(commit) if not x.startswith(".")]
            for method in methods:
                res.append(method)
    return res


class MethodDiff:

    def __init__(self, atom_dir):
        self.atom_dir = atom_dir
        self.pdg1 = None
        self.pdg2 = None
        self.slices1 = []
        self.slices2 = []
        self.paths1 = []
        self.paths2 = []
        self.v1()
        self.v2()

    def v1(self):
        path = os.path.join(self.atom_dir, 'g1.json')
        if os.path.exists(path):
            g = read_json(path)
            graph = Graph(g)
            self.pdg1 = graph
            n = graph.get_graph_property('slice_num1')
            for i in range(n):
                s = graph.get_slice(i)
                self.slices1.append(s)

    def v2(self):
        path = os.path.join(self.atom_dir, 'g2.json')
        if os.path.exists(path):
            g = read_json(path)
            graph = Graph(g)
            self.pdg2 = graph
            n = graph.get_graph_property('slice_num2')
            for i in range(n):
                s = graph.get_slice(i)
                self.slices2.append(s)
