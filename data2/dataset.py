import json
import os
from abc import ABC, abstractmethod

from diff import *


def list_json_files(path):
    json_files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".json")]
    return json_files


def read_json(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)
        commit = Commit(json_data)
        return commit


class DiffDataset(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class CommitDataset(DiffDataset):

    def __init__(self, path):
        self.path = path
        self.data = []
        self.load()

    def load(self):
        json_files = list_json_files(self.path)
        for json_file in json_files:
            commit = read_json(json_file)
            self.data.append(commit)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for data in self.data:
            yield data


class FileDataset(DiffDataset):

    def __init__(self, path):
        self.path = path
        self.data = []
        self.load()

    def load(self):
        json_files = list_json_files(self.path)
        for json_file in json_files:
            commit = read_json(json_file)
            for file in commit.files:
                self.data.append(file)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for data in self.data:
            yield data


class SliceDataset(DiffDataset):

    def __init__(self, path):
        self.path = path
        self.data = []
        self.load()

    def load(self):
        json_files = list_json_files(self.path)
        for json_file in json_files:
            commit = read_json(json_file)
            for file in commit.files:
                for hunk in file.hunks:
                    self.data.append(hunk.slice1)
                    self.data.append(hunk.slice2)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for data in self.data:
            yield data
