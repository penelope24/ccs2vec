import unittest

import networkx as nx


class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = "/Users/fy/Documents/workspace/ccs2vec/samples"
        self.dataset = None
        self.vocab = None

    def setUp(self) -> None:
        self.dataset = SliceDataset(self.base_path)
        ds = CommitDataset(self.base_path)
        self.vocab = Vocab.create_from_dataset(ds, special_tokens, min_freq_value)

    def test_print(self):
        print(nx.__version__)

    # step 1
    def test_build_dataset(self):
        ds1 = SliceDataset(self.base_path)
        ds2 = FileDataset(self.base_path)
        ds3 = CommitDataset(self.base_path)
        ds4 = SliceVirtualDataset(self.base_path)
        print(len(ds1))
        print(len(ds2))
        print(len(ds3))
        print(len(ds4))
        data = next(iter(ds4))
        ids = []
        for node in data.nodes:
            ids.append(node.id)
        sids = sorted(ids)
        print(sids)

    def test_build_virtual_graph(self):
        data = next(iter(self.dataset))
        slice = data.slice1
        vg = SliceVirtualGraph(slice)

    # step 2
    def test_build_vocab(self):
        ds = CommitDataset(self.base_path)
        self.vocab = Vocab.create_from_dataset(ds, [], 1)
        print(self.vocab.word2idx)

    # step 3
    def test_build_graph(self):
        vg = next(iter(self.dataset))
        data = Graph.create(vg.nodes, vg.edges, self.vocab.word2idx)
        print(data.edge_index)
        # G = nx.Graph()
        # G.add_edges_from(data_dep.edge_index.t().tolist())
        # pos = nx.spring_layout(G)  # 布局算法可以根据需要选择
        # # pos = nx.nx_agraph.pygraphviz_layout(G, prog='dot')
        # nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=500)
        # plt.show()

    def test_build_seq(self):
        slice = next(iter(self.dataset))
        seq = Sequence(slice)
        seq.generate()

    def test_sort(self):
        ss = [1, 2, 3, -1]

        sss = sorted(ss, key=lambda i: (i, i <= 0))
        print(sss)

    def test_batch(self):
        pass

    def test_model(self):
        vg = next(iter(self.dataset))
        data = Graph.create(vg.nodes, vg.edges, self.vocab.word2idx)

        num_nodes = data.num_nodes
        num_mask_nodes = int(num_nodes * 0.15)

        # 随机选择要遮掩的节点索引
        mask_indices = random.sample(range(num_nodes), num_mask_nodes)
        print(num_nodes)
        print(num_mask_nodes)
        print(mask_indices)

        # 对 data_dep.x 进行遮掩
        data.x[mask_indices] = 0.0

        mask_mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask_mask[mask_indices] = 1
        mask_mask = mask_mask[data.edge_index[0]]
        ei = data.edge_index[:, ~mask_mask]
        print(~mask_mask)
        print(data.edge_index)
        print(ei)


        # indices = data_dep.x[11]
        # indices = [int(n) for n in indices]
        # print(indices)
        # words = [self.vocab.idx2word[i] for i in indices]
        # print(words)