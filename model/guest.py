import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
import numpy as np
import math

from model.fi_gnn import Fi_GNN
from model.bilstm import biLSTM

################################GUEST################################################
class SelfAttentionLayer(nn.Module):
    def __init__(self, nhid, nins):  #(768*2,5)
        super(SelfAttentionLayer, self).__init__()
        self.nhid = nhid #768*2
        self.nins = nins #5
        
        self.project = nn.Sequential(
            Linear(nhid, 64),
            nn.LeakyReLU(0.1),
            Linear(64, 1)
        )

    def forward(self, inputs, index, claims):
        tmp = None
        if index > -1:
            idx = torch.LongTensor([index]).cuda()
            own = torch.index_select(inputs, 1, idx)  #[batch, 1, 768]
            own = own.repeat(1, self.nins, 1)  #[batch, 5, 768]
            tmp = torch.cat((own, inputs), 2)
        else:
            claims = claims.repeat(1, self.nins, 1) #[batch, 5, 768]
            tmp = torch.cat((claims, inputs), 2) #[batch, 5, 2*768]
        # before
        attention = self.project(tmp) #[batch, 5, 1] 
        weights = F.softmax(attention.squeeze(-1), dim=1) #[batch, 5]
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1) #[batch, 1, 768]
        return (outputs, torch.unsqueeze(weights, dim=-1))

class AttentionLayer(nn.Module):
    def __init__(self, nins, nhid):   #(5,768)
        super(AttentionLayer, self).__init__()
        self.nins = nins   #5
        self.attentions = [SelfAttentionLayer(nhid=nhid * 2, nins=nins) for _ in range(nins)] #[batch, 1, 768]*5
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, inputs):
        outputs = []
        attns4visual = []
        for i in range(self.nins):
            attns_edges = self.attentions[i](inputs, i, None)
            outputs.append(attns_edges[0])
            attns4visual.append(attns_edges[1])
        outputs = torch.cat(outputs, dim=1)
        attns4visual = torch.cat(attns4visual, dim=2)
        outputs = outputs.view(inputs.shape) #[batch, 5, 768]
        return outputs, attns4visual
    

class GUEST(nn.Module):                                             
    def __init__(self, nfeat, bi_num_hidden, nums_feat,  nembd_1, nembd_2,  nins, nclass, nlayer, pool):
        super(GUEST, self).__init__()
        self.nlayer = nlayer
        self.comm_num = nins
        self.attentions = [AttentionLayer(nins, nfeat) for _ in range(nlayer)] #[batch, 5, 768]*1
        self.batch_norms = [BatchNorm1d(nins) for _ in range(nlayer)] #[batch, 5, 768]*1
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.pool = pool
        if pool == 'att':
            self.aggregate = SelfAttentionLayer(nfeat * 2, nins)
        self.index = torch.LongTensor([0]).cuda()
        
        #BiLSTM
        self.biLSTM = biLSTM(1, nembd_2)

        self.weight = nn.Parameter(torch.FloatTensor(nfeat, nclass)) #[768,2]
        self.bias = nn.Parameter(torch.FloatTensor(nclass))
        
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

                
        #Fi-GNN for comment
        self.test = Fi_GNN(nembd_1, nembd_1, nums_feat, self.comm_num, feature_all=[2, 2, 2, 3, 3, 3, 2 ,2 ,2 ,2 ,2])
        
        #Co-attention
        self.W_aff1 = nn.Parameter(torch.FloatTensor(nfeat, nembd_1))
        self.W_aff1.data.uniform_(-stdv, stdv)
        self.W_sem1 = nn.Linear(nfeat, bi_num_hidden, bias=True)
        self.W_fea1 = nn.Linear(nembd_1, bi_num_hidden, bias=True)
        self.W_sf1 = nn.Linear(bi_num_hidden, 1, bias=True)
        
        self.W_aff2 = nn.Parameter(torch.FloatTensor(nfeat, nembd_1))
        self.W_aff2.data.uniform_(-stdv, stdv)
        self.W_sem2 = nn.Linear(nfeat, bi_num_hidden, bias=True)
        self.W_fea2 = nn.Linear(nembd_1, bi_num_hidden, bias=True)
        self.W_sf2 = nn.Linear(bi_num_hidden, 1, bias=True)
        
        
        #entailment-attention
        self.W_entail = nn.Linear((nfeat+nembd_1)*4, (nfeat+nembd_1)*2, bias=True)
        self.W_entail2 = nn.Linear((nfeat+nembd_1)*2, 1, bias=True)
        
        #final output     
        self.FFN_final = nn.Sequential(
            nn.Linear((nfeat+nembd_1)*2, (nfeat+nembd_1)*1, bias=True),
            nn.Linear((nfeat+nembd_1)*1, 2, bias=True),
            )
        
        
    def forward(self, inputs, claims, claim_features, comm_features, comm_masks, tweet_struct_temp):
        for i in range(self.nlayer):
            inputs, attns4visual = self.attentions[i](inputs)
 
        #temporal
        struct_BiSLTM = self.biLSTM(tweet_struct_temp)
        
        #fea embedding
        all_feas = torch.cat((claim_features.unsqueeze(1), comm_features), dim=1).float()
        binary_cred, inputs_FiGNNs = self.test(all_feas)
        
        #GEAR+FiGNN_claim
        inputs_FiGNN_claim = inputs_FiGNNs[:,0,:].unsqueeze(1)
        claim_fs = torch.cat((claims, inputs_FiGNN_claim), dim=-1)
        
        #GEAR+FiGNN_comm
        inputs_FiGNN_comm = inputs_FiGNNs[:,1:,:]
        binary_cred = binary_cred[:,1:]  
        inputs_FiGNN_comm = torch.mul(binary_cred.unsqueeze(-1), inputs_FiGNN_comm)
        
         
        Comm_fs = torch.cat((inputs, inputs_FiGNN_comm), dim=-1)
        
        #entailment-attention
        claim_fs = claim_fs.repeat(1,inputs.size()[1],1)
        fs_claim_comm = torch.cat((claim_fs, Comm_fs, claim_fs*Comm_fs, torch.abs(claim_fs-Comm_fs)), dim=-1)
        fs_claim_comm = torch.tanh(self.W_entail(fs_claim_comm))
        fs_W = F.softmax(torch.tanh(self.W_entail2(fs_claim_comm)), 1)
        fs_final = torch.sum(fs_W * fs_claim_comm, dim=1)
       
        #final output
        output = self.FFN_final(fs_final)
        
        return F.log_softmax(output, dim=1), attns4visual, binary_cred, fs_W