import torch
import torch.nn as nn
from model.transformer.Utils import Utils
from model.transformer.LayerNorm import LayerNorm


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = Utils.clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.display_shape = True

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        if self.display_shape:
            print("shape after whole decoders: ", x.size())
        return self.norm(x)