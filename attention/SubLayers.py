''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_input_k,d_k, d_v, dropout=0.1):
        super().__init__()

        '''
        n_head 几个头
        d_model q的输入维度
        d_input_k k的输入维度
        d_k k的输出维度
        d_v v的输出维度
        '''
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #线性
        self.w_qs = nn.Linear(d_model, n_head * d_k)   #64，4*16
        self.w_ks = nn.Linear(d_input_k, n_head * d_k)
        self.w_vs = nn.Linear(d_input_k, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_input_k + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_input_k + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)  #初始化

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None): #（batch*32*32,3,64）

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_k, len_k, _ = k.size()
        sz_v, len_v, _ = v.size()

        residual = q
        #(batch*32*32,3,64)*(64,4*16)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  #(batch*32*32,3,4,16)
        k = self.w_ks(k).view(sz_k, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_v, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk  (batch*32*32*4,3,16)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk  (4*batch*32*32,3,16)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x .. 每个维度重复n次

        output, attn = self.attention(q, k, v, mask=mask)
        #output(batch*32*32*4,3,16)  attn(batch*32*32*4,3,3) q*k
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv) (batch*32*32,3,4*16)

        output = self.dropout(self.fc(output))  #(3,4*16)*(4*16,64)（batch*32*32,3,64）
        # print("output")
        # print(output)
        output = self.layer_norm(output + residual)

        return output, attn,q

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise (64,256)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise  (256,64)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  #(batch*32*32,3,64)
        residual = x
        output = x.transpose(1, 2)  #(batch*32*32,64,3)
        output = self.w_2(F.relu(self.w_1(output)))   #self.w_1(output)  (batch*32*32,256,3) (batch*32*32,64,3)
        output = output.transpose(1, 2)   #(batch*32*32,3,64)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
