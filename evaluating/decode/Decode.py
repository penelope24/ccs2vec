import torch
from math import log
import operator
from queue import PriorityQueue
from torch.autograd import Variable
from dataset.batch.mask.Mask import Mask
from op.tensor.TensorOp import TensorOp


class BeamSearchNode(object):
    def __init__(self, y_input, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.ys = y_input
        self.prevNode = previousNode
        self.wordId = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        reward = -log(self.leng) * log(self.leng / 2)

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
        # return self.logp


class Decode:
    def __init__(self):
        pass

    @staticmethod
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(memory, src_mask,
                               Variable(ys),
                               Variable(Mask.subsequent_mask(ys.size(1))
                                        .type_as(src.data)))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys

    @staticmethod
    def greedy_decode_batch(model, src, src_mask, max_len, start_symbol, batch_size):
        memory = model.encode(src, src_mask)
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(memory, src_mask,
                               Variable(ys),
                               Variable(Mask.subsequent_mask(ys.size(1))
                                        .type_as(src.data)))
            # print(out.size(), out[:,-1].size())
            # print(out[:, -1])
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data
            next_word = next_word.unsqueeze(1)
            # print(next_word.size())
            ys = torch.cat([ys, next_word.type_as(src.data)], dim=1)
        #     print(ys)
        # print('--------')
        return ys

    @staticmethod
    def beam_search_decode(model, src, src_mask, sos_id, eos_id, beam_width=3):
        decoded_batch = []
        # topk = beam_width
        topk = 1
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(sos_id).type_as(src.data)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  decoder input, previous node, word id, logp, length
        node = BeamSearchNode(ys, None, ys, 0.0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            # if n.prevNode is not None:
            #     print("current node: ", n.wordId, n.prevNode.wordId, n.ys, score)

            if n.wordId.item() == eos_id and n.prevNode is not None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
            ys = n.ys
            out = model.decode(memory, src_mask,
                               Variable(ys),
                               Variable(Mask.subsequent_mask(ys.size(1))
                                        .type_as(src.data)))
            prob = model.generator(out[:, -1])

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(prob, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                y = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                # print(log_p)
                ys = n.ys
                ys = torch.cat([ys, y.type_as(src.data)], dim=1)
                node = BeamSearchNode(ys, n, y, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                # print(node.wordId, score)
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                # print(score, nn)
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordId)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordId)

            utterance = utterance[::-1]
            # print(utterance)
            utterances.append(utterance)
        t = torch.tensor(utterances[0]).unsqueeze(0)
        return t

