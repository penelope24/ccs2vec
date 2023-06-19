import json
import os

import networkx as nx
from networkx.drawing.nx_pydot import read_dot


def read_from_json(path):
    with open(path) as f:
        d = json.load(f)
        graph = nx.readwrite.node_link_graph(d)
        for k, v in d.items():
            graph.graph[k] = v
        return graph


def read_from_dot(path):
    return read_dot(path)


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