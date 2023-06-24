import logging
import numpy as np
from evaluating.decode.Decode import Decode
from evaluating.metrics.F1 import F1
from evaluating.metrics.Bleu import Bleu
import torch
from op.tensor.TensorOp import TensorOp
import random
from torch.autograd import Variable
from dataset.batch.mask.Mask import Mask
from time import time

"""
1. decode method
2. max length
3. batch / single
4. measure
"""

class Eval:
    def __init__(self, model, iter, sos_id, eos_id, pad_id, unk_id):
        self.model = model
        self.iter = iter
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.max_gen_len = 5
        self.max_y_len = 5
        self.measure = "f1"
        self.decode = "beam"

    def set_measure(self, measure):
        self.measure = measure

    def set_decode_method(self, decode):
        self.decode = decode

    def set_max_gen_len(self, max):
        self.max_gen_len = max

    def set_max_y_len(self, max):
        self.max_y_len = max

    def process(self):
        if self.measure == "f1":
            if self.decode == "beam":
                self.method1()
            if self.decode == "greedy":
                self.method2()
        if self.measure == "bleu":
            if self.decode == "beam":
                self.method3()
            if self.decode == "greedy":
                self.method4()


    def method1(self):
        """
        calculate f1 using beam search
        """
        accs = []
        precs = []
        recs = []
        f1s = []
        for batch in self.iter:
            outs = []
            tgts = []
            if batch.tgt_y.size(1) <= self.max_y_len:
                max_size = 0
                BATCH_SIZE = batch.src.size(0)
                for i in range(BATCH_SIZE):
                    src = batch.src[i:i + 1]
                    src_mask = batch.src_mask[i:i + 1]
                    tgt = batch.tgt_y[i:i + 1]
                    out = Decode.beam_search_decode(self.model, src, src_mask, sos_id=self.sos_id,
                                                    eos_id=self.eos_id)
                    max_size = out.size(1) if out.size(1) > max_size else max_size
                    outs.append(out)
                    tgts.append(tgt)
                outs = [TensorOp.tpad(out, dim=1, n=max_size - out.size(1), fillvalue=self.pad_id) for out in outs]
                batch_pred = torch.cat([out for out in outs], dim=0)
                batch_tgt = torch.cat([tgt for tgt in tgts], dim=0)
                metric = F1(BATCH_SIZE, self.sos_id, self.eos_id, self.pad_id, self.unk_id)
                acc, prec, recall, f1 = metric.calculate(batch_pred, batch_tgt)
                # print("length: %d, accuracy: %0.3f, precision: %0.3f, recall: %0.3f, f1: %0.3f"
                #       % (batch.tgt_y.size(1), acc, prec, recall, f1))
                accs.append(acc)
                precs.append(prec)
                recs.append(recall)
                f1s.append(f1)
        metrics = {
            "accuracy": np.array(accs).mean(),
            "precision": np.array(precs).mean(),
            "recall": np.array(recs).mean(),
            "f1": np.array(f1s).mean()
        }
        return metrics


    def method2(self):
        """
        calculate f1 using greedy decode, in batch mode
        """
        accs = []
        precs = []
        recs = []
        f1s = []
        for batch in self.iter:
            if batch.tgt_y.size(1) <= self.max_y_len:
                BATCH_SIZE = batch.src.size(0)
                out = Decode.greedy_decode_batch(self.model, batch.src, batch.src_mask, max_len=self.max_gen_len, start_symbol=self.sos_id,
                                                 batch_size=BATCH_SIZE)
                batch_pred = out
                batch_tgt = batch.tgt_y
                metric = F1(BATCH_SIZE, self.sos_id, self.eos_id, self.pad_id, self.unk_id)
                acc, prec, recall, f1 = metric.calculate(batch_pred, batch_tgt)
                # print("length: %d, accuracy: %0.3f, precision: %0.3f, recall: %0.3f, f1: %0.3f"
                #       % (batch.tgt_y.size(1), acc, prec, recall, f1))
                accs.append(acc)
                precs.append(prec)
                recs.append(recall)
                f1s.append(f1)
            else:
                pass

        metrics = {
            "accuracy": np.array(accs).mean(),
            "precision": np.array(precs).mean(),
            "recall": np.array(recs).mean(),
            "f1": np.array(f1s).mean()
        }
        return metrics


