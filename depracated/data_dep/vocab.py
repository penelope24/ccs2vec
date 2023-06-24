from collections import OrderedDict, Counter

import torch

from utils.tokenize import tokenize

special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[SLICE]', '[PATH]']
min_freq_value = 1
is_special_first = True


def initialize_embeddings(vocab_size, embedding_dim):
    embeddings = torch.zeros(vocab_size, embedding_dim)
    for i in range(vocab_size):
        # 为每个单词生成一个随机初始化的向量
        random_vector = torch.FloatTensor(embedding_dim).uniform_(-0.5, 0.5)
        embeddings[i] = random_vector
    return embeddings


class Vocab:

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        self.embedding_dim = 300

    @classmethod
    def create_from_dataset(cls, dataset, specials, min_freq):
        vocab = cls()
        vocab.build_from_commit_dataset(dataset, specials, min_freq)
        return vocab

    def build_from_commit_dataset(self, dataset, specials, min_freq):
        for token in specials:
            self.word2idx[token] = len(self.word2idx)
        counter = Counter()
        for commit in dataset:
            for file in commit.files:
                counter.update(file.tokens)
        freq_dict = OrderedDict(counter)
        for word, freq in freq_dict.items():
            if freq >= min_freq:
                self.word2idx[word] = len(self.word2idx)
        self.embeddings = initialize_embeddings(len(self.word2idx), self.embedding_dim)

    def load_vocab_file(self, vocab_file):
        pass

    def load_from_pretrained(self, embedding_file):
        pass
