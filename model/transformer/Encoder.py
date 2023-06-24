import torch.nn as nn
from model.transformer.Utils import Utils
from model.transformer.LayerNorm import LayerNorm


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = Utils.clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.display_shape = True

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        if self.display_shape:
            print("shape after whole encoder: ", x.size())
        return self.norm(x)