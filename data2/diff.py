import json
from graphs import *


class Commit:

    def __init__(self, commit: dict):
        self.files = []
        self.parse(commit)

    def parse(self, commit: dict):
        files = commit['files']
        for file in files:
            self.files.append(File(file))


class File:

    def __init__(self, file: dict):
        self.pdg_list_v1 = []
        self.pdg_list_v2 = []
        self.hunks = []
        self.parse(file)

    @staticmethod
    def get_graph_by_id(graphs: List[Graph], id: int):
        res = [x for x in graphs if x.m_id == id]
        if res:
            return res[0]
        else:
            return None

    def parse(self, file: dict):
        graphs1 = file['graphs1']
        graphs2 = file['graphs2']
        for g in graphs1:
            self.pdg_list_v1.append(Graph(g['method_id'], g['nodes'], g['links'], None))
        for g in graphs2:
            self.pdg_list_v2.append(Graph(g['method_id'], g['nodes'], g['links'], None))
        for hunk in file['hunks']:
            g1 = self.get_graph_by_id(self.pdg_list_v1, hunk['method_id1'])
            g2 = self.get_graph_by_id(self.pdg_list_v2, hunk['method_id2'])
            self.hunks.append(Hunk(hunk, g1, g2))


class Hunk:

    def __init__(self, hunk: dict, g1: Graph, g2: Graph):
        self.slice1 = None
        self.slice2 = None
        self.parse(hunk, g1, g2)

    def parse(self, hunk: dict, g1: Graph, g2: Graph):
        slice_edges1 = hunk['slice1']['edges']
        self.slice1 = g1.get_sub_graph(slice_edges1)
        slice_edges2 = hunk['slice2']['edges']
        self.slice2 = g2.get_sub_graph(slice_edges2)
