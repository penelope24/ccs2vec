import torch
import torch.nn as nn
from model.transformer.Utils import Utils
from model.transformer.SublayerConnection import SublayerConnection


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = Utils.clones(SublayerConnection(size, dropout), 3)
        self.display_shape = True

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        if self.display_shape:
            print("after decoder self attn:", x.shape)
            print("memory shape:", memory.shape)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        if self.display_shape:
            print("after decode src attn:", x.shape)
        return self.sublayer[2](x, self.feed_forward)