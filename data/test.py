import unittest

from dataset import *


class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = "/Users/fy/Documents/workspace/ccs2vec/samples"

    def test_tensor(self):
        x = torch.randn(4, 2)
        print(x.size(dim=0))
