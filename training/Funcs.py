import torch
from time import time

"""
functions for training and logging
"""

class Funcs:
    def __init__(self):
        pass

    global max_src_in_batch, max_tgt_in_batch

    @staticmethod
    def batch_size_fn(new, count, sofar):
        "keep augmenting batch and calculate total number of tokens + padding"
        """insane"""
        global max_src_in_batch, max_tgt_in_batch
        if count == 1:
            max_src_in_batch = 0
            max_tgt_in_batch = 0
        max_src_in_batch = max(max_src_in_batch, len(new.src))
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        src_elements = count * max_src_in_batch
        tgt_elements = count * max_tgt_in_batch
        return max(src_elements, tgt_elements)
