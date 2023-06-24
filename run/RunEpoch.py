from time import time
from tqdm import tqdm
from op.model.utils import RunningAverage, load_checkpoint, save_checkpoint, save_dict_to_json
import numpy as np
import logging
import os


class RunEpoch:
    def __init__(self):
        pass

    @staticmethod
    def run_epoch(data_iter, model, loss_compute):
        "standard training and logging function"

        total_tokens = 0
        total_loss = 0
        count = 0

        # for batch in tqdm(data_iter):
        for batch in data_iter:
            count += 1
            # print(batch.src.size())
            # print(batch.tgt.size())
            # print(batch.src_mask.size())
            # print(batch.tgt_mask.size())
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            # print(out.size())
            loss = loss_compute(out, batch.tgt_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
        return total_loss / total_tokens

    @staticmethod
    def  run_single_step(data_iter, model, loss_compute):
        batch = next(data_iter.__iter__())
        print("raw input size: ", batch.src.size())
        print("raw target size: ", batch.tgt.size())
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)

    @staticmethod
    def train(model, loss_compute, data_iterator, metrics, params, save_summary_steps):
        """Train the model on `num_steps` batches

        Args:
            model: (torch.nn.Module) the neural network
            loss_compute: include loss compute and optimizer update
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyper-parameters
            num_steps: (int) number of batches to run on, each of size params.batch_size
        """

        # set model to training mode
        model.train()

        # summary for current training loop and a running average object for loss
        summ = []
        loss_avg = RunningAverage()

        i = 0
        t = tqdm(data_iterator)
        for batch in t:
            output_batch = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = loss_compute(output_batch, batch.tgt_y, batch.ntokens)

            # update the average loss
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
