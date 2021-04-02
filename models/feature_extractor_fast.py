# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
def convbn_relu(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes)) 
                         
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
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # if self.use_bn:
            # x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d: 
            kernel = (3, 4, 4)

        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=False, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):

        x = self.conv1(x)
        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.inplanes = 32
#         self.firstconv = convbn_relu(1, 32, 7, 2, 3, 1)
        self.firstconv = nn.Sequential(convbn_relu(1, 32, 3, 2, 1, 1),
                               convbn_relu(32, 32, 3, 1, 1, 1),
                               convbn_relu(32, 32, 3, 1, 1, 1))
        self.layer1 = self._make_layer(BasicBlock, 32, 1, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 1, 1, 1, 1)
        self.reduce = convbn_relu(128, 32, 3, 1, 1, 1)
        
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
#         self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

#         self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
#         self.conv4b = Conv2x(96, 128)

#         self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)
    def forward(self, x):
        #1/2
        x = self.firstconv(x)
        x = self.layer1(x)
        conv0a = x
        x = self.layer2(x) #1/4
        conv1a = x
        x = self.layer3(x) #1/8 * 128
        feat0 = x
        x = self.layer4(x) #1/8 * 128
        feat1 = x
        x = self.reduce(x) #1/8 * 32
        feat2 = x
        rem0 = x
        #1/2 * 1/2 * 48
        x = self.conv1a(x)
        rem1 = x
        #1/4 * 1/4 * 64
        x = self.conv2a(x)
        rem2 = x
        #1/8 * 1/8 * 96
        x = self.conv3a(x)
        rem3 = x
        #1/16 * 1/16 * 128
#         x = self.conv4a(x)
#         rem4 = x
#         x = self.deconv4a(x, rem3)
#         rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        feat3 = x
        rem0 = x
        #1/2
        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        #1/16
#         x = self.conv4b(x, rem4)

#         x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)
        feat4 = x
        gwc_feature = torch.cat((feat0,feat1,feat2,feat3,feat4),dim = 1)
        return conv0a,gwc_feature

