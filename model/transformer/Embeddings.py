import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.display_shape = True

    def forward(self, x):
        if self.display_shape:
            print("embedding input:", x.shape)
        out = self.lut(x) * math.sqrt(self.d_model)
        if self.display_shape:
            print("embedding output:", out.shape)
        return self.lut(x) * math.sqrt(self.d_model)
