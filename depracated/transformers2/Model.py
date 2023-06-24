from .decoder import Decoder
from .decoder import DecoderLayer
from .embedding import Embeddings
from .encoder import Encoder
from .architecture import EncoderDecoder
from .encoder import EncoderLayer
from .attention import MultiHeadedAttention
from .architecture import PositionWiseFeedForward
from .PositionalEncoding import PositionalEncoding
from .Generator import Generator
# from model.transformer import *


import torch
import numpy as np
import torch.nn as nn
import copy

class Model():
    "class to make a complete transformer model"

    @staticmethod
    def make_model(src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyper-parameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

# test
if __name__ == "__main__":
    Model.make_model(100, 100, 6)