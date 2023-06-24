import torch
import torch.nn as nn
from .Utils import Utils
from .SublayerConnection import SublayerConnection


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = Utils.clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.display_shape = True

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        if self.display_shape:
            print("input before self-attn:", x.shape)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        if self.display_shape:
            print("shape after self attn:", x.shape)
        return self.sublayer[1](x, self.feed_forward)