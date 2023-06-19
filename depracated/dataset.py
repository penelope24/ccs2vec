from torch.utils.data.dataset import IterableDataset
from torchtext.vocab import build_vocab_from_iterator

from depracated.diff import MethodDiff, list_atom_dirs
from utils.graphs import PDG


class DiffDataset(IterableDataset):

    def __init__(self, path):
        self.base = path
        self.atom_dirs = list_atom_dirs(path)
        self.special_tokens = ['<unk>', '<pad>', '<s>', '<eos>']

    def __len__(self):
        return len(self.atom_dirs)

    def __iter__(self):
        for atom_dir in self.atom_dirs:
            diff = MethodDiff(atom_dir)
            if diff is not None and not diff.is_empty():
                yield diff
                # tokens = []
                # tokens1 = diff.list_tokens("v1")
                # tokens2 = diff.list_tokens("v2")
                # tokens.extend(tokens1)
                # tokens.extend(tokens2)
                # yield tokens

    def build_vocab(self):
        # occurrences = []
        # for tokens in self:
        #     occurrences.extend(tokens[0])
        #     occurrences.extend(tokens[1])
        # counter = Counter(occurrences)
        # sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # ordered_dict = OrderedDict(sorted_by_freq_tuples)
        # v = vocab(ordered_dict, specials=self.special_tokens, special_first=True)
        vocab = build_vocab_from_iterator(self.__iter__(), specials=self.special_tokens)
        return vocab

    def build_matrix(self):
        new_graph = PDG.create_empty()
        for diff in self:
            g1:PDG = diff.graph1
            g2:PDG = diff.graph2
            for node in g1.get_nodes():
                new_graph.add_node(node)
                new_graph.add_node()
