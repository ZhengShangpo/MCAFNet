# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from torch.nn import init, Sequential
from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn
from models.common import *


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class feature_fusion_at(nn.Module):
    def __init__(self, inchannel):
        super(feature_fusion_at, self).__init__()
        self.fusion_1 = Conv(inchannel, inchannel, 1)
        self.fusion_2 = Conv(inchannel, inchannel, 1)
        self.fusion_3 = Conv(inchannel, inchannel, 1)
        self.fusion_4 = Conv(inchannel * 3, 3, 1)

    def forward(self, x1, x2, x3):
        fusion = torch.softmax(
            self.fusion_4(torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)), dim=1)
        x1_weight, x2_weight, x3_weight = torch.split(fusion, [1, 1, 1], dim=1)
        return x1 * x1_weight + x2 * x2_weight + x3 * x3_weight


class feature_fusion(nn.Module):
    def __init__(self, inchannel):
        super(feature_fusion, self).__init__()
        self.fusion_1 = Conv(inchannel, inchannel, 1)
        self.fusion_2 = Conv(inchannel, inchannel, 1)
        self.fusion_3 = Conv(inchannel * 2, 2, 1)

    def forward(self, x1, x2):
        fusion = torch.softmax(self.fusion_3(torch.cat([self.fusion_1(x1), self.fusion_1(x2)], dim=1)), dim=1)
        x1_weight, x2_weight = torch.split(fusion, [1, 1], dim=1)
        return x1 * x1_weight + x2 * x2_weight


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes)
            self.relu = nn.SiLU() if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.SiLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=False,
                 bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.D = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.S = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            self.C = nn.Sequential(self.D, max_pool_layer)
            self.S = nn.Sequential(self.S, max_pool_layer)
        self.CAM = ChannelAttention(self.inter_channels)
        self.SP = SpatialAttention(self.inter_channels)


class NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=False,
                 bn_layer=False):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.D = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.S = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.C = conv_nd(in_channels=self.in_channels,
                 out_channels=self.inter_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            self.C = nn.Sequential(self.C, max_pool_layer)
            self.S = nn.Sequential(self.S, max_pool_layer)
        self.sim = SimAM(self.inter_channels)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''
        batch_size = x.size(0)
        x1 = self.sim(x)

        S_x = self.S(x1).view(batch_size, self.inter_channels, -1)
        S_x = S_x.permute(0, 2, 1)

        D_x = self.D(x1).view(batch_size, self.inter_channels, -1)
        D_x = D_x.permute(0, 2, 1)

        C_x = self.C(x1).view(batch_size, self.inter_channels, -1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)


        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f1 = torch.matmul(theta_x, phi_x)
        f_div_C1 = F.softmax(f1, dim=-1)

        f2 = torch.matmul(D_x, phi_x)
        f_div_C2 = F.softmax(f2, dim=-1)

        y1 = torch.matmul(f_div_C1, g_x)
        y2 = torch.matmul(f_div_C2, S_x)
        y = y1 + y2
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class Conv_NonLocal(nn.Module):
    # Standard convolution·
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        # Standard convolution
        super(Conv_NonLocal, self).__init__()
        self.NonLocalBlockND = NonLocalBlockND(c1, c2)
        # self.NonLocal = NonLocalBlockND(in_channels=c1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class CAN(nn.Module):
    def __init__(self, c1, kernel_size=16, bn_layer=True, sub_sample=True):
        super().__init__()
        self.in_channels = c1
        self.inter_channels = c1 // 2
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d
        self.ASPP = ASPP(c1)

        self.avgpool = nn.AdaptiveAvgPool2d((kernel_size, kernel_size))

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels,
                          out_channels=self.in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        # self.W1= nn.Conv2d(in_channels=2*self.inter_channels, out_channels=self.in_channels,
        #                    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''
        batch_size = x.size(0)
        avg_x = self.avgpool(x)
        x1 = self.ASPP(x)

        g_x = self.g(avg_x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)  # [bs, w*h,c]

        theta_x = self.theta(avg_x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        theta_x = theta_x.permute(0, 2, 1)  # [bs, w*h,c]

        phi_x = self.phi(avg_x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]

        f = torch.matmul(theta_x, phi_x)  # [bs,  w*h, w*h]

        N = f.size(-1)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)  # [bs, w*h, c]
        y = y.permute(0, 2, 1).contiguous()  # [bs, c, w*h]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x1
        # out = torch.concat([z,x1],dim=1)
        # out = self.W1(out)
        return z


class ASPP(nn.Module):
    def __init__(self, in_channel=512, padding_dilation=(2)):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        # self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        # self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
        # self.atrous_block3 = BasicConv(in_channel, out_channel, 3, 1, padding=3, dilation=3,bn=False)
        self.relu_b = nn.ReLU(inplace=True)
        self.branch0 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        # for i,rate in enumerate(padding_dilation):
        #     if i ==1 :
        #         self.branch1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=rate,
        #                              dilation=rate, bias=False)
        #     elif i == 2:
        #         self.branch2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=rate,
        #                              dilation=rate, bias=False)
        #     else:
        #         self.branch3 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=rate,
        #                              dilation=rate, bias=False)
        # self.conv_1x1_output = nn.Conv2d(in_channel * 4, in_channel, 1, 1)
        # self.atrous_block5 = BasicConv(in_channel, out_channel, 3, 1, padding=5, dilation=5)
        # self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        # self.cbam = CBAM(out_channel)

    def forward(self, x):
        # size = x.shape[2:]
        # image_features = self.mean(x)
        # image_features = self.conv(image_features)
        # image_features = F.upsample(image_features, size=size, mode='bilinear')
        # image_features = self.mean(x)
        # image_features = self.conv(x)
        # x1 = self.branch0(x)
        # x2_1 = self.branch1(x)
        x = self.branch0(x)
        # x2_3 = self.branch3(x)
        # x = self.conv_1x1_output(torch.cat([x1,x2_1, x2_2,x2_3], dim=1))
        # x = self.cbam(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # # Convolution layers to generate dynamic weights A_c and B_c
        # self.conv_a = nn.Conv2d(1, c1, kernel_size=1, bias=False)
        # self.conv_b = nn.Conv2d(1, c1, kernel_size=1, bias=False)
        #
        # # Initializing the weights
        # nn.init.kaiming_normal_(self.conv_a.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv_b.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        # 1*h*w
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)
        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)
        x1 = torch.cat([avg_out1, max_out1], dim=1)
        x2 = torch.cat([avg_out2, max_out2], dim=1)
        # 2*h*w
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        # A_c = self.conv_a(x)
        # B_c = self.conv_b(x)
        # 1*h*w
        weight1 = self.sigmoid(x1)

        weight2 = self.sigmoid(x2)
        return weight1,weight2


class Add_SN(nn.Module):
    #  Add two tensorss
    def __init__(self, arg):
        super(Add_SN, self).__init__()
        self.SimAM = SimAM()
        self.CAM_SAM = CAM_SAM(arg)
        # self.feature_fusion = feature_fusion(arg)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        # x1 = self.NonLocal(x1)
        # x2 = self.NonLocal(x2)
        x1 = self.SimAM(x1)
        x2 = self.SimAM(x2)
        x = self.CAM_SAM(x1, x2)
        return x


class CAM_SAM(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.CAM = ChannelAttention(c1)
        self.SAM = SpatialAttention(c1)
        # self.feature_fusion = feature_fusion()
        # self.conv = nn.Conv2d(c1 * 2, c1, 1, padding=0, bias=False)

    def forward(self, x1, x2):
        x1 = x1 * self.CAM(x1)
        x2 = x2 * self.CAM(x2)
        x3 = torch.cat([x1, x2], dim=1)
        weight1,weight2 = self.SAM(x1, x2)
        x =  weight1*x1+weight2*x2
        return x


class Add2_SN(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index
        self.SimAM = SimAM()
        #    self.NonLocalBlockND = NonLocalBlockND(c1, c1)
        self.CAM_SAM = CAM_SAM(c1)
        self.feature_fusion = feature_fusion(c1)

    def forward(self, x):
        if self.index == 0:
            x1 = x[0]
            x2 = x[1][0]
            x1 = self.SimAM(x1)
            # x = self.feature_fusion(x1, x2)
            x = self.CAM_SAM(x1, x2)
            #     x = self.NonLocalBlockND(x)
            # x = self.NonLocalBlockND(x)
            #   x = self.feature_fusion(x1, x2)
            # x1_weight, x2_weight = torch.split(x, [1, 1], dim=1)
            # x = torch.add(x1_weight * x1, x2_weight * x2)
            return x
        elif self.index == 1:
            x1 = x[0]
            x2 = x[1][1]
            x1 = self.SimAM(x1)
            x = self.CAM_SAM(x1, x2)
            #  x = self.feature_fusion(x1, x2)
            # x = self.CAM_SAM(x1, x2)
            #   x = self.NonLocalBlockND(x)
            #        x = self.NonLocalBlockND(x)
            # x1_weight, x2_weight = torch.split(x, [1, 1], dim=1)
            # x = torch.add(x1_weight * x1, x2_weight * x2)
            return x