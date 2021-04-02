# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from models.submodules import SubModule, convbn_3d_lrelu, convbn_transpose_3d
class HourGlass(SubModule):
    def __init__(self, inplanes=16):
        super(HourGlass, self).__init__()

        self.conv1 = convbn_3d_lrelu(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1)
        self.conv2 = convbn_3d_lrelu(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv1_1 = convbn_3d_lrelu(inplanes * 2, inplanes * 4, kernel_size=3, stride=2, pad=1)
        self.conv2_1 = convbn_3d_lrelu(inplanes * 4, inplanes * 4, kernel_size=3, stride=1, pad=1)

        self.conv3 = convbn_3d_lrelu(inplanes * 4, inplanes * 8, kernel_size=3, stride=2, pad=1)
        self.conv4 = convbn_3d_lrelu(inplanes * 8, inplanes * 8, kernel_size=3, stride=1, pad=1)

        self.conv5 = convbn_transpose_3d(inplanes * 8, inplanes * 4, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv6 = convbn_transpose_3d(inplanes * 4, inplanes * 2, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv7 = convbn_transpose_3d(inplanes * 2, inplanes, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.last_for_guidance = convbn_3d_lrelu(inplanes, 32, kernel_size=3, stride=1, pad=1)
        self.weight_init()
#modify from Deeppruner code
class CoeffsPredictor(HourGlass):

    def __init__(self, hourglass_inplanes=16):
        super(CoeffsPredictor, self).__init__(hourglass_inplanes)

    def forward(self, input):
        output0 = self.conv1(input)
        output0_a = self.conv2(output0) + output0

        output0 = self.conv1_1(output0_a)
        output0_c = self.conv2_1(output0) + output0

        output0 = self.conv3(output0_c)
        output0 = self.conv4(output0) + output0

        output1 = self.conv5(output0) + output0_c
        output1 = self.conv6(output1) + output0_a
        output1 = self.conv7(output1)
        #[B,G,D,H,W] -> [B,D,G,H,W]
        coeffs = self.last_for_guidance(output1).permute(0,2,1,3,4).contiguous()
        return coeffs




