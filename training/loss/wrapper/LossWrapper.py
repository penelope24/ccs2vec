import torch
import torch.nn as nn
from torch.autograd import Variable


class LossWrapper(nn.Module):
    """implements any loss from torch.nn.Loss with label smoothing"""
    def __init__(self, size, criterion, padding_idx, smoothing=0.0):
        """
        :param size: size of axe along which loos compute is performing
        :param criterion: any subclass of torch.nn.Loss
        :param padding_idx: index that need to be padded
        :param smoothing: smoothing constant
        """
        super(LossWrapper, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        :param x: of size (batch_size, n)
        :param target: same size as x
        :return:
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



