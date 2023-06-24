import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import sacrebleu
from sacremoses import MosesTokenizer, MosesDetokenizer
from evaluating.decode.Decode import Decode


class Bleu:

    def __init__(self, model, iter, sos_id, eos_id, pad_id, unk_id):
        self.model = model
        self.iter = iter
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.max_gen_len = 10
        self.mt = MosesTokenizer(lang="en")
        self.md = MosesDetokenizer(lang="en")

    def set_max_gen_len(self, max):
        self.max_gen_len = max

    def process(self, path):
        bleu1 = self.read_and_cal_nltk_corpus_bleu(path)
        # bleu2 = self.read_and_cal_sacre_corpus_bleu(path)
        return bleu1

    def write1(self, path):
        """
        write beam search results
        """
        with open(path + "/bleu.pred", "w") as f_pred:
            with open(path + "/bleu.target", "w") as f_tgt:
                for batch in self.iter:
                    BATCH_SIZE = batch.src.size(0)
                    for i in range(BATCH_SIZE):
                        src = batch.src[i:i + 1]
                        src_mask = batch.src_mask[i:i + 1]
                        tgt = batch.tgt_y[i:i + 1]
                        out = Decode.beam_search_decode(self.model, src, src_mask, sos_id=self.sos_id,
                                                        eos_id=self.eos_id)
                        out = out[0].tolist()
                        tgt = tgt[0].tolist()
                        out = [str(t) for t in out]
                        tgt = [str(t) for t in tgt]
                        f_pred.write(" ".join(out))
                        f_pred.write("\n")
                        f_tgt.write(" ".join(tgt))
                        f_tgt.write("\n")

    def write2(self, path):
        """
        write greedy decode results.
        """
        with open(path + "/bleu.pred", "w") as f_pred:
            with open(path + "/bleu.target", "w") as f_tgt:
                for batch in self.iter:
                    BATCH_SIZE = batch.src.size(0)
                    for i in range(BATCH_SIZE):
                        src = batch.src[i:i + 1]
                        src_mask = batch.src_mask[i:i + 1]
                        tgt = batch.tgt_y[i:i + 1]
                        out = Decode.greedy_decode(self.model, src, src_mask, max_len=self.max_gen_len,
                                                   start_symbol=self.sos_id)
                        out = out[0].tolist()
                        tgt = tgt[0].tolist()
                        out = [str(t) for t in out]
                        tgt = [str(t) for t in tgt]
                        f_pred.write(" ".join(out))
                        f_pred.write("\n")
                        f_tgt.write(" ".join(tgt))
                        f_tgt.write("\n")

    def read_and_cal_nltk_corpus_bleu(self, path):
        list_of_refs = []
        preds = []
        with open(path + "/bleu.pred", "r") as f_pred:
            for line in f_pred.readlines():
                line = line.strip().split()
                line = self.md.detokenize(line,return_str=False)
                preds.append(line)
        with open(path + "/bleu.target", "r") as f_tgt:
            for line in f_tgt.readlines():
                line = line.strip().split()
                line = self.md.detokenize(line, return_str=False)
                list_of_refs.append([line])
        score = Bleu.nltk_corpus_bleu(list_of_refs, preds)
        return score

    def read_and_cal_sacre_corpus_bleu(self, path):
        refs = []
        preds = []
        with open(path + "/bleu.pred", "r") as f_pred:
            for line in f_pred.readlines():
                line = line.strip().split()
                line = self.md.detokenize(line,return_str=True)
                preds.append(line)
        with open(path + "/bleu.target", "r") as f_tgt:
            for line in f_tgt.readlines():
                line = line.strip().split()
                line = self.md.detokenize(line, return_str=True)
                refs.append(line)
        list_of_refs = [refs]
        return Bleu.sacre_corpus_bleu(list_of_refs, preds)

    def filter_impossible_names(self, names_list):
        result = []
        for name in names_list:
            if name not in [self.sos_id, self.eos_id, self.pad_id, self.unk_id]:
                result.append(name)
        return result

    @staticmethod
    def nltk_sentence_bleu(refs, pred):
        """
        :param refs: tuple lists
        :param pred: list
        :return:
        """
        smooth = SmoothingFunction()
        return sentence_bleu(refs, pred, smoothing_function=smooth.method2)

    @staticmethod
    def nltk_corpus_bleu(list_of_refs, preds):
        """
        :param list_of_refs: triple lists
        :param preds: tuple lists
        :return:
        """
        # print(preds)
        # print(list_of_refs)
        smooth = SmoothingFunction()
        return corpus_bleu(list_of_refs, preds,smoothing_function=smooth.method2)

    @staticmethod
    def sacre_corpus_bleu(list_of_refs, preds):
        """
        'floor': 0.0,
        'add-k': 1,
        'exp': None,    # No value is required
        'none': None,   # No value is required
        """
        # print(preds)
        # print(list_of_refs)
        bleu = sacrebleu.corpus_bleu(preds, list_of_refs)
        return bleu.score
