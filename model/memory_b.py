import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
import numpy as np
import h5py
import os
import sys

sys.path.append('../')
sys.path.append('../attention')
from attention.SubLayers import MultiHeadAttention, PositionwiseFeedForward

#print("全部的类别特征直接为类别特征15层resnet")
class memory(nn.Module):
    def __init__(self, map_height, map_width, num_classes, feats_channels=64):
        super(memory, self).__init__()
        self.num_classes = num_classes
        self.feats_channels = feats_channels

        self.map_height = map_height
        self.map_width = map_width

        # 获得类别特征改变维度为64
        _, center, _, _, _, _ = read_cache()
        self.m = torch.FloatTensor(center)
        self.memory = nn.Parameter(data=torch.Tensor(center),
                                    requires_grad=False)
        print(self.memory.shape)

        self.slf_attn = MultiHeadAttention(
            n_head=1, d_model=self.feats_channels, d_input_k=self.memory.shape[1], d_k=self.memory.shape[1],
            d_v=self.feats_channels)

        self.slf_attn2 = MultiHeadAttention(
            n_head=4, d_model=self.feats_channels, d_input_k=self.feats_channels, d_k=64,
            d_v=self.feats_channels)

        self.slf_attn1 = MultiHeadAttention(
            n_head=4, d_model=self.feats_channels, d_input_k=self.feats_channels, d_k=64,
            d_v=self.feats_channels)
        self.pos_ffn = PositionwiseFeedForward(d_in=self.feats_channels, d_hid=256)

        self.update_attn = MultiHeadAttention(
            n_head=1, d_model=self.memory.shape[1], d_input_k=self.feats_channels, d_k=self.memory.shape[1],
            d_v=self.memory.shape[1])
        # 对每个类进行卷积，获得每个类的卷积核
        self.class_conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 8, kernel_size=4,
                                stride=1, padding=0, bias=True)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(8, 64, kernel_size=3,
                                stride=1, padding=0, bias=True))]))

        self.w_b = nn.Linear(64, 5)
        self.w_x=nn.Linear(64, 96)
        self.w_n=nn.Linear(96,64)

    def forward(self, x):  #(b,n,d)
        b, c, H, W = x.size()  # (b,c,H,W)
        x = x.permute(0, 2, 3, 1).reshape(b, H * W, c)

        #通过全连接层求概率
        weight=self.w_b(x)
        weight1=F.softmax(weight,dim=-1)  #(n,8)
        weight2=F.softmax(weight.permute(0,2,1),dim=-1)

        self.class_feat1=torch.matmul(weight2,x)   #(8,64)

        # attention 返回class_feat(32,8,64)
        class_feat, self.weight1,self.class_feat2 = self.slf_attn(self.class_feat1, self.memory.unsqueeze(0), self.memory.unsqueeze(0))

        class_feat = torch.matmul(weight1, class_feat)
        #空间
        #print("空间")
        class_feat,_,_=self.slf_attn1(class_feat,class_feat,class_feat)
        # 获得卷积核#每个类别的特征没求

        class_feat=class_feat.permute(0,2,1).reshape(-1,64,H,W)
        return class_feat












