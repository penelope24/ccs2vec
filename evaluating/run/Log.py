import numpy as np
from evaluating.decode.Decode import Decode


class Log:

    def __init__(self, model, iter, sos_id, eos_id, pad_id, unk_id):
        self.model = model
        self.iter = iter
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.max_gen_len = 5
        self.max_y_len = 100000
        self.decode = "beam"

    def set_decode_method(self, decode):
        self.decode = decode

    def set_max_gen_len(self, max):
        self.max_gen_len = max

    def set_max_y_len(self, max):
        self.max_y_len = max

    def filter_impossible_names(self, names_list):
        result = []
        for name in names_list:
            if name not in [self.sos_id, self.eos_id, self.pad_id, self.unk_id]:
                result.append(name)
        return result

    def show(self):
        if self.decode == "beam":
            for batch in self.iter:
                if batch.tgt_y.size(1) <= self.max_y_len:
                    BATCH_SIZE = batch.src.size(0)
                    for i in range(BATCH_SIZE):
                        src = batch.src[i:i + 1]
                        src_mask = batch.src_mask[i:i + 1]
                        tgt = batch.tgt_y[i:i + 1]
                        out = Decode.beam_search_decode(self.model, src, src_mask, sos_id=self.sos_id,
                                                        eos_id=self.eos_id)
                        out = self.filter_impossible_names(out.tolist()[0])
                        tgt = self.filter_impossible_names(tgt.tolist()[0])
                        print(out, tgt)
                        print("--------------------------------------------")
        else:
            for batch in self.iter:
                if batch.tgt_y.size(1) <= self.max_y_len:
                    BATCH_SIZE = batch.src.size(0)
                    for i in range(BATCH_SIZE):
                        src = batch.src[i:i + 1]
                        src_mask = batch.src_mask[i:i + 1]
                        tgt = batch.tgt_y[i:i + 1]
                        out = Decode.greedy_decode(self.model, src, src_mask, max_len=self.max_gen_len, start_symbol=self.sos_id)
                        out = self.filter_impossible_names(out.tolist()[0])
                        tgt = self.filter_impossible_names(tgt.tolist()[0])
                        print(out, tgt)
                        print("--------------------------------------------")

    def get_statics(self):
        outs = []
        tgts = []
        if self.decode == "beam":
            for batch in self.iter:
                if batch.tgt_y.size(1) <= self.max_y_len:
                    BATCH_SIZE = batch.src.size(0)
                    for i in range(BATCH_SIZE):
                        src = batch.src[i:i + 1]
                        src_mask = batch.src_mask[i:i + 1]
                        tgt = batch.tgt_y[i:i + 1]
                        out = Decode.beam_search_decode(self.model, src, src_mask, sos_id=self.sos_id,
                                                        eos_id=self.eos_id)
                        out = self.filter_impossible_names(out.tolist()[0])
                        tgt = self.filter_impossible_names(tgt.tolist()[0])
                        outs.append(out)
                        tgts.append(tgt)
        else:
            for batch in self.iter:
                if batch.tgt_y.size(1) <= self.max_y_len:
                    BATCH_SIZE = batch.src.size(0)
                    for i in range(BATCH_SIZE):
                        src = batch.src[i:i + 1]
                        src_mask = batch.src_mask[i:i + 1]
                        tgt = batch.tgt_y[i:i + 1]
                        out = Decode.greedy_decode(self.model, src, src_mask, max_len=self.max_gen_len, start_symbol=self.sos_id)
                        out = self.filter_impossible_names(out.tolist()[0])
                        tgt = self.filter_impossible_names(tgt.tolist()[0])
                        outs.append(out)
                        tgts.append(tgt)

        metrics = {
            "prediction_mean": np.array(outs).mean(),
            "prediction_max": np.array(outs).max(),
            "target_mean": np.array(tgts).mean(),
            "target_max": np.array(tgts).max()
        }
        return metrics
