import torch
import numpy as np


class Mask:
    """
    hold a batch of data_dep and mask
    @:param src: tensor of size [batch_size, src_length]
    @:param tgt: tensor of size [batch_size, tgt_length]
    @:returns src_mask : tensor of size [batch_size, 1, src_length]
    @:returns tgt_mask : tensor of size [batch_size, size, size] where size = tgt_length - 1
    """
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_tgt_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_tgt_mask(tgt, pad):
        "mask: 1.padding; 2.future words"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_subsequence_mask = Mask.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        final_mask = tgt_mask & tgt_subsequence_mask # bit-wise AND
        return final_mask

    @staticmethod
    def subsequent_mask(size):
        "mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequence_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequence_mask) == 0

    @staticmethod
    def struc_mask(text):
        pass

    @staticmethod
    def data_flow_edge_mask(size):
        pass

    @staticmethod
    def ctrl_flow_edge_mask(size):
        pass

    @staticmethod
    def code_change_trans_mask(text):
        pass