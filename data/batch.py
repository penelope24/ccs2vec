import torch
from torch.utils.data import DataLoader
from data.mask import Mask


def collate_fn(batch):
    # 将 batch 中的样本按照文本长度排序（从长到短）
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    texts, labels = zip(*batch)  # 将文本和标签分离

    # 假设你的文本数据是字符串列表
    # 对文本序列进行填充或截断，使它们具有相同的长度
    max_length = max(len(text) for text in texts)
    padded_texts = []
    for text in texts:
        padded_text = text[:max_length] + [0] * (max_length - len(text))
        padded_texts.append(padded_text)

    # 返回填充后的文本序列和对应的标签
    return torch.tensor(padded_texts), torch.tensor(labels)


def get_data_loader(dataset, batch_size, collate_fn):
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return data_loader


class BasicBatchWrapper:

    def __init__(self, dl):
        self.dl = dl


    def __iter__(self):
        for batch in self.dl:
            yield Mask(batch)