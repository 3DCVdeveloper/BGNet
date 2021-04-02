# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function
import torch.nn as nn
import math
def convbn_2d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(kernel_size, kernel_size),
                  stride=(stride, stride), padding=(pad, pad), dilation=(dilation, dilation), bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True))
 
def convbn_2d_Tanh(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(kernel_size, kernel_size),
                  stride=(stride, stride), padding=(pad, pad), dilation=(dilation, dilation), bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.Tanh())

def deconvbn_2d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation=1,bias=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                               dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),                   
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

def convbn_3d_lrelu(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=(pad, pad, pad),
                                   stride=(1, stride, stride), bias=False),
                         nn.BatchNorm3d(out_planes),
                         nn.LeakyReLU(0.1, inplace=True))


def conv_relu(in_planes, out_planes, kernel_size, stride, pad, bias=True):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad, bias=bias),
                         nn.ReLU(inplace=True))


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_relu(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))


def convbn_transpose_3d(inplanes, outplanes, kernel_size, padding, output_padding, stride, bias):
    return nn.Sequential(nn.ConvTranspose3d(inplanes, outplanes, kernel_size, padding=padding,
                                            output_padding=output_padding, stride=stride, bias=bias),
                         nn.BatchNorm3d(outplanes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = convbn_relu(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
