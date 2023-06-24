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


def eval2(model, iter, sos_id, eos_id, pad_id, unk_id, max_len=5):
    """
    measure: f1
    decode: greedy
    batch: true
    max_len: 5
    """
    accs = []
    precs = []
    recs = []
    f1s = []
    for batch in iter:
        outs = []
        tgts = []
        BATCH_SIZE = batch.src.size(0)
        for i in range(BATCH_SIZE):
            src = batch.src[i:i + 1]
            src_mask = batch.src_mask[i:i + 1]
            tgt = batch.tgt_y[i:i + 1]
            out = Decode.greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=sos_id)
            outs.append(out)
            tgts.append(tgt)
        batch_pred = torch.cat([out for out in outs], dim=0)
        batch_tgt = torch.cat([tgt for tgt in tgts], dim=0)
        metric = F1(BATCH_SIZE, sos_id, eos_id, pad_id, unk_id)
        acc, prec, recall, f1 = metric.calculate(batch_pred, batch_tgt)
        print("length: %d, accuracy: %0.3f, precision: %0.3f, recall: %0.3f, f1: %0.3f"
              % (batch.tgt_y.size(1), acc, prec, recall, f1))
        accs.append(acc)
        precs.append(prec)
        recs.append(recall)
        f1s.append(f1)

    return np.array(accs).mean(), np.array(precs).mean(), np.array(recs).mean(), np.array(f1s).mean()


def eval3(model, iter, sos_id, eos_id, pad_id, unk_id, max_len=5):
    accs = []
    precs = []
    recs = []
    f1s = []
    for batch in iter:
        BATCH_SIZE = batch.src.size(0)
        out = Decode.greedy_decode_batch(model, batch.src, batch.src_mask, max_len=max_len, start_symbol=sos_id, batch_size=BATCH_SIZE)
        batch_pred = out
        batch_tgt = batch.tgt_y
        metric = F1(BATCH_SIZE, sos_id, eos_id, pad_id, unk_id)
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


def eval4(model, iter, sos_id, eos_id, pad_id, unk_id, max_len=5):
    accs = []
    precs = []
    recs = []
    f1s = []
    for batch in iter:
        if batch.tgt_y.size(1) <= max_len:
            BATCH_SIZE = batch.src.size(0)
            out = Decode.greedy_decode_batch(model, batch.src, batch.src_mask, max_len=max_len, start_symbol=sos_id, batch_size=BATCH_SIZE)
            batch_pred = out
            batch_tgt = batch.tgt_y
            metric = F1(BATCH_SIZE, sos_id, eos_id, pad_id, unk_id)
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

def eval5(model, iter, sos_id, eos_id, pad_id, unk_id, max_len=5):
    accs = []
    precs = []
    recs = []
    f1s = []
    for batch in iter:
        max_size = 0
        outs = []
        tgts = []
        if batch.tgt_y.size(1) <= max_len:
            BATCH_SIZE = batch.src.size(0)
            for i in range(BATCH_SIZE):
                src = batch.src[i:i + 1]
                src_mask = batch.src_mask[i:i + 1]
                tgt = batch.tgt_y[i:i + 1]
                out = Decode.beam_search_decode(model, src, src_mask, sos_id=sos_id,
                                                eos_id=eos_id)
                max_size = out.size(1) if out.size(1) > max_size else max_size
                outs.append(out)
                tgts.append(tgt)
            outs = [TensorOp.tpad(out, dim=1, n=max_size-out.size(1), fillvalue=pad_id) for out in outs]
            batch_pred = torch.cat([out for out in outs], dim=0)
            batch_tgt = torch.cat([tgt for tgt in tgts], dim=0)
            metric = F1(BATCH_SIZE, sos_id, eos_id, pad_id, unk_id)
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


def eval6(model, iter, sos_id, eos_id, pad_id, unk_id, max_len=5):
    accs = []
    precs = []
    recs = []
    f1s = []
    for batch in iter:
        max_size = 0
        outs = []
        tgts = []
        BATCH_SIZE = batch.src.size(0)
        for i in range(BATCH_SIZE):
            src = batch.src[i:i + 1]
            src_mask = batch.src_mask[i:i + 1]
            tgt = batch.tgt_y[i:i + 1]
            out = Decode.beam_search_decode(model, src, src_mask, sos_id=sos_id,
                                            eos_id=eos_id)
            max_size = out.size(1) if out.size(1) > max_size else max_size
            outs.append(out)
            tgts.append(tgt)
        outs = [TensorOp.tpad(out, dim=1, n=max_size-out.size(1), fillvalue=pad_id) for out in outs]
        batch_pred = torch.cat([out for out in outs], dim=0)
        batch_tgt = torch.cat([tgt for tgt in tgts], dim=0)
        metric = F1(BATCH_SIZE, sos_id, eos_id, pad_id, unk_id)
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

def eval7(model, iter, sos_id, eos_id, pad_id, unk_id):
    for batch in iter:
        max_size = 0
        outs = []
        tgts = []
        BATCH_SIZE = batch.src.size(0)
        for i in range(BATCH_SIZE):
            src = batch.src[i:i + 1]
            src_mask = batch.src_mask[i:i + 1]
            tgt = batch.tgt_y[i:i + 1]
            out = Decode.greedy_decode(model, src, src_mask, max_len=10, start_symbol=sos_id)
            max_size = out.size(1) if out.size(1) > max_size else max_size
            outs.append(out)
            tgts.append(tgt)
        outs = [TensorOp.tpad(out, dim=1, n=max_size - out.size(1), fillvalue=pad_id) for out in outs]
        batch_pred = torch.cat([out for out in outs], dim=0)
        batch_tgt = torch.cat([tgt for tgt in tgts], dim=0)
        metric = Bleu(BATCH_SIZE, sos_id, eos_id, pad_id, unk_id)
        score = metric.calculate(batch_pred, batch_tgt)
        # print("length: %d, accuracy: %0.3f, precision: %0.3f, recall: %0.3f, f1: %0.3f"
        #       % (batch.tgt_y.size(1), acc, prec, recall, f1))
    return score