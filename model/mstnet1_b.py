import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import torch.optim as optim
import time
import torch.nn.init as init
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import sys
sys.path.append('../../')
#from preprocessing.TaxiBJ import load_data
from torch.utils.data import TensorDataset, DataLoader

from memory_b import memory
from gcn.GCN import GCN
from gcn.cheb_gcn import cheb_conv
#官方的stnet


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias= True)

class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn = False):
        super(_bn_relu_conv, self).__init__()
        self.has_bn = bn
        #self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        #if self.has_bn:
        #    x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x

class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual # short cut

        return out

class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x



# Matrix-based fusion
class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w),
                                    requires_grad = True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        x = x * self.weights # element-wise multiplication

        return x




class  mstnet1(nn.Module):
    def __init__(self,
                 chebpolynomials="",
                 learning_rate=0.0001,
                 c_conf=(3, 2, 32, 32), p_conf=(4, 2, 32, 32),
                 t_conf=(4, 2, 32, 32),
                 external_dim=28,
                 nb_residual_unit=2,
                ):

        super(mstnet1, self).__init__()
        logger = logging.getLogger(__name__)
        logger.info('initializing net params and ops ...')

        self.learning_rate = learning_rate
        self.external_dim = external_dim

        self.nb_flow, self.map_height, self.map_width = t_conf[1], t_conf[2], t_conf[3]
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf

        self.len_closeness = self.c_conf[0]
        self.len_period = self.p_conf[0]
        self.len_trend = self.t_conf[0]

        self.nb_residual_unit = nb_residual_unit
        self.external_dim = external_dim
        self.logger = logging.getLogger(__name__)

        if self.c_conf is not None:
            self.c_way = self.make_one_way(in_channels = self.c_conf[0] * self.nb_flow)
        # Branch p
        if self.p_conf is not None:
            self.p_way = self.make_one_way(in_channels = self.p_conf[0] * self.nb_flow)
        # Branch t
        if self.t_conf is not None:
            self.t_way = self.make_one_way(in_channels = self.t_conf[0] * self.nb_flow)


        # Operations of external component
        if self.external_dim != None and self.external_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
                ('embd', nn.Linear(self.external_dim, 128, bias=True)),
                ('relu1', nn.ReLU()),
                ('fc', nn.Linear(128, 64* self.map_height * self.map_width, bias=True)),
                ('relu2', nn.ReLU()),
            ]))

        self.conv = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.gcn=cheb_conv(3,chebpolynomials,64,64)

        self.layer_norm = nn.LayerNorm(64)

    def make_one_way(self, in_channels):

        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels=in_channels, out_channels=64)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter=64, repetations=self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('memory1',memory(self.map_width,self.map_width,5, 64)),
            ('conv2', conv3x3(in_channels=64, out_channels=64)),
            ('FusionLayer', TrainableEltwiseLayer(n=64, h=self.map_height, w=self.map_width))
        ]))




    def forward(self, input_c, input_p, input_t,ext,adjust_learning_rate=0.0001,label='Train'): #输入（128，6，32，32） （128，2，32，32）  （128，2，32，32）

        # Three-way Convolution
        main_output = 0
        res=0
        if self.c_conf is not None:  # （3，2，32，32）--（1，6，32，32）
            #input_c = input_c.view(-1, self.c_conf[0] * 2, self.map_height, self.map_width)
            out_c = self.c_way(input_c)
            res+= out_c
        if self.p_conf is not None:
            #input_p = input_p.view(-1, self.p_conf[0] * 2, self.map_height, self.map_width)
            out_p = self.p_way(input_p)
            res+= out_p
        if self.t_conf is not None:
            #input_t = input_t.view(-1, self.t_conf[0] * 2, self.map_height, self.map_width)
            out_t = self.t_way(input_t)
            res+= out_t

        # fusing with external component
        if self.external_dim != None and self.external_dim > 0:
            # external input
            # print(input_ext.shape)
            external_output = self.external_ops(ext)
            external_output = torch.relu(external_output)
            external_output = external_output.view(-1, 64, self.map_height, self.map_width)
            # main_output = torch.add(main_output, external_output)
            res += external_output

        else:
            print('external_dim:', self.external_dim)


        class_feat= self.memory(res)  # (b,h*w,c)
        class_feat = class_feat.permute(0, 2, 3, 1).reshape(-1, self.map_height * self.map_width, 64)

        class_feat = self.gcn(class_feat)
        res = class_feat.permute(0, 2, 1).reshape(-1, 64, self.map_height, self.map_width)
        res = self.conv(res)
        res = torch.tanh(res)

        return res

