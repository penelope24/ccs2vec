import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.w_1(x)
        # x2 = F.relu(x1)
        # print(x2.shape)
        # x3 = self.w_2(self.dropout(x2))
        # print(x3.shape)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))