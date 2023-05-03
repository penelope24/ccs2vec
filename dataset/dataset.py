import torch
from torch.utils.data.dataset import Dataset


class PDGDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        processed_sample = self.preprocess(sample)
        return processed_sample

    def preprocess(self, sample):
        return sample

    def __iter__(self):
        pass
