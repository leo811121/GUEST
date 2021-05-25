import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
import numpy as np
import math

##############################Fi-GNN########################################################
class message_pass(nn.Module):
    def __init__(self, hidden_size, node_num, sen_num, graph=None):
        super(message_pass, self).__init__()
        self.graph = graph
        self.hidden_size = hidden_size
        self.node_num = node_num
        self.sen_num = sen_num
        self.w_state1 = nn.ModuleList([Linear(hidden_size, hidden_size) for i in range(sen_num)]) #
        self.w_state3 = nn.ModuleList([Linear(hidden_size, hidden_size) for i in range(sen_num)]) #
    def forward(self, input_feature):
        self.state1 = torch.stack([self.w_state1[i](input_feature[:,i,:]) for i in range(self.sen_num)], dim=2).permute(0,2,1)
        self.state2 = torch.sum(self.state1, dim=1).reshape(-1, 1, self.hidden_size).repeat(1,self.sen_num,1) #  
        self.state2 -= self.state1
        self.state3 = torch.stack([self.w_state3[i](self.state2[:,i,:]) for i in range(self.sen_num)], dim=2).permute(0,2,1) #

        return self.state3
    
class GNN(nn.Module):
    def __init__(self, feature_input_len, hidden_size, node_num, sen_num, graph=None):
        super(GNN, self).__init__()
        self.graph = graph
        self.hidden_size = hidden_size
        self.node_num = node_num
        self.messagePass = message_pass(hidden_size, node_num, sen_num)
        self.gru_cell = nn.GRUCell(feature_input_len, hidden_size)
        self.sen_num = sen_num
    def forward(self, feature_input):
        self.h0 = feature_input
        self.state = feature_input
        feature_input_new = self.messagePass(feature_input)
        state_new = self.gru_cell(feature_input_new.reshape(-1,self.hidden_size), self.state.reshape(-1,self.hidden_size))
        state_new = state_new.reshape(-1, self.sen_num, self.hidden_size) #
        self.state = self.h0 + state_new
        return self.state


class normalize(nn.Module):
    def __init__(self, input_shape):
        super(normalize, self).__init__()
        self.inputs_shape = input_shape
        self.instaNorm = nn.InstanceNorm1d(input_shape)
        self.params_shape = input_shape
        self.beta = nn.Parameter(torch.zeros(self.params_shape))
        self.gamma = nn.Parameter(torch.ones(self.params_shape))
        
    def forward(self, feature_input):
        self.normalized = self.instaNorm(feature_input)
        self.outputs = self.gamma * self.normalized + self.beta
        return self.outputs

class multihead_attention(nn.Module):
    def __init__(self, feature_input_shape, num_units, num_heads=4, has_residual=True):
        super(multihead_attention, self).__init__()
        self.num_heads = num_heads
        self.has_residual = has_residual
        self.w_Q = nn.Sequential(Linear(feature_input_shape, num_units),
                      ReLU(True))
        self.w_K = nn.Sequential(Linear(feature_input_shape, num_units),
                      ReLU(True))
        self.w_V = nn.Sequential(Linear(feature_input_shape, num_units),
                      ReLU(True))
        self.w_V_res = nn.Sequential(Linear(feature_input_shape, num_units),
                      ReLU(True))
        self.Relu = ReLU(True)
        self.softmax = torch.nn.Softmax()
        self.normalize = normalize(feature_input_shape)
     
    def forward(self, feature_input):
        self.Q = self.w_Q(feature_input)
        self.Q = torch.cat(torch.split(self.Q, int(self.Q.size()[-1]/self.num_heads), dim=2), dim=0)
        self.K = self.w_K(feature_input)
        self.K = torch.cat(torch.split(self.K, int(self.K.size()[-1]/self.num_heads), dim=2), dim=0)
        self.V = self.w_V(feature_input)
        self.V = torch.cat(torch.split(self.V, int(self.V.size()[-1]/self.num_heads), dim=2), dim=0)
        
        if self.has_residual:
          self.V_res = self.w_V_res(feature_input)

        weights = self.softmax(torch.matmul(self.Q, self.K.permute(0,2,1)) / (self.K.size()[-1]**0.5))
        self.outputs = torch.matmul(weights, self.V)
        self.outputs = torch.cat(torch.split(self.outputs, int(self.outputs.size()[0]/self.num_heads), dim=0), dim=2)
        if self.has_residual:
            self.outputs += self.V_res
        self.outputs = self.Relu(self.outputs)
        self.outputs = self.normalize(self.outputs)

        return self.outputs
    
class Fi_GNN(nn.Module):
    def __init__(self, feature_input_len, hidden_size, node_num, nums_comm, feature_all=[2,2,2], graph=None):
        super(Fi_GNN, self).__init__()
        self.feature_all = feature_all
        self.feature_input_len = feature_input_len
        self.hidden_size = hidden_size
        self.node_num = node_num
        self.nums_comm = nums_comm
        hidden_stdv = np.sqrt(1. / self.feature_input_len)

        self.embed = nn.Sequential(
            Linear(node_num, self.hidden_size),
            nn.Tanh()
          )
        
        #Multi_attention
        self.attention = multihead_attention(feature_input_len, hidden_size)
        #GNN 
        self.GNN = GNN(self.feature_input_len, self.hidden_size, self.node_num, self.nums_comm+1)
        #Attentional Scoring Layer
        self.MLP1 =  nn.Sequential(
            Linear(hidden_size, self.hidden_size),
            nn.Tanh()
          )
        self.MLP2 =  nn.Sequential(
            Linear(hidden_size, self.hidden_size),
            nn.LeakyReLU(0.01)
          )
        
    def forward(self, inputs):
        # feature embedding
        self.feature_input = self.embed(inputs)
        self.attn_input = self.attention(self.feature_input)
        self.final_state = self.GNN(self.attn_input)
        self.final_state1 = self.GNN(self.final_state)
        self.final_state2 = self.GNN(self.final_state1)
        self.final_state3 = self.GNN(self.final_state2)
        self.cred = torch.sum(torch.mul(self.MLP1(self.final_state3), self.MLP2(self.final_state3)), dim=-1)
        binary_out = torch.sigmoid(self.cred) #attention
        return binary_out, self.final_state