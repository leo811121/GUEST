import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
import numpy as np
import math

from model.fi_gnn import Fi_GNN

class biLSTM(nn.Module):
     def __init__(self, ninput, nhid):  
        super(biLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=ninput, hidden_size=nhid, bidirectional=True)
        self.W = nn.Linear(nhid*2, 1, bias=False)
     def forward(self, input):
        input = input.unsqueeze(-1)
        h = self.lstm(input)[0]
        a = F.softmax(torch.tanh(self.W(self.lstm(input)[0])), dim=1)
        h = a * h
        output = torch.sum(h, dim=1)
        
        return output