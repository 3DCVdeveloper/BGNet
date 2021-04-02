# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function
from models.feature_extractor_fast import feature_extraction
from models.submodules3d import CoeffsPredictor
from models.submodules2d import HourglassRefinement
from models.submodules import SubModule,convbn_2d_lrelu,convbn_3d_lrelu,convbn_2d_Tanh
from nets.warp import disp_warp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time        
class Slice(SubModule):
    def __init__(self):
        super(Slice, self).__init__()
    def forward(self, bilateral_grid, wg, hg, guidemap): 
        guidemap = guidemap.permute(0,2,3,1).contiguous() #[B,C,H,W]-> [B,H,W,C]
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide,align_corners =False)
        return coeff.squeeze(2) #[B,1,H,W]

      
class GuideNN(SubModule):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = convbn_2d_lrelu(32, 16, 1, 1, 0)
        self.conv2 = convbn_2d_Tanh(16, 1, 1, 1, 0)

    def forward(self, x):
        return self.conv2(self.conv1(x)) 
        
def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    #[B,G,D,H,W]
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
def correlation(fea1, fea2):
    B, C, H, W = fea1.shape
    cost = (fea1 * fea2).mean(dim=1)
    assert cost.shape == (B, H, W)
    return cost

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp + 1, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp + 1, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)
class BGNet(SubModule):
    def __init__(self):
        super(BGNet, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        # self.refinement_net = HourglassRefinement()
      
        self.feature_extraction = feature_extraction()
        self.coeffs_disparity_predictor = CoeffsPredictor()


        self.dres0 = nn.Sequential(convbn_3d_lrelu(44, 32, 3, 1, 1),
                                   convbn_3d_lrelu(32, 16, 3, 1, 1))
        self.guide = GuideNN()
        self.slice = Slice()
        self.weight_init()





 
    def forward(self, left_input, right_input):         
        left_low_level_features_1,  left_gwc_feature  = self.feature_extraction(left_input)
        _,                          right_gwc_feature = self.feature_extraction(right_input)
        
        guide = self.guide(left_low_level_features_1) #[B,1,H,W]
        # torch.cuda.synchronize()
        # start = time.time()
        cost_volume = build_gwc_volume(left_gwc_feature,right_gwc_feature,25,44)
        cost_volume = self.dres0(cost_volume)
        #coeffs:[B,D,G,H,W]
        coeffs = self.coeffs_disparity_predictor(cost_volume)

        list_coeffs = torch.split(coeffs,1,dim = 1)
        index = torch.arange(0,97)
        index_float = index/4.0
        index_a = torch.floor(index_float)
        index_b = index_a + 1
        
        index_a = torch.clamp(index_a, min=0, max= 24)
        index_b = torch.clamp(index_b, min=0, max= 24)
        
        wa = index_b - index_float
        wb = index_float   - index_a

        list_float = []  
        device = list_coeffs[0].get_device()
        wa  = wa.view(1,-1,1,1)
        wb  = wb.view(1,-1,1,1)
        wa = wa.to(device)
        wb = wb.to(device)
        wa = wa.float()
        wb = wb.float()

        N, _, H, W = guide.shape
        #[H,W]
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        #[B,H,W,1]
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        slice_dict = []
        # torch.cuda.synchronize()
        # start = time.time()
        for i in range(25):
            slice_dict.append(self.slice(list_coeffs[i], wg, hg, guide)) #[B,1,H,W]
        slice_dict_a = []
        slice_dict_b = []
        for i in range(97):
            inx_a = i//4
            inx_b = inx_a + 1
            inx_b  = min(inx_b,24)
            slice_dict_a.append(slice_dict[inx_a])
            slice_dict_b.append(slice_dict[inx_b])
            
        final_cost_volume = wa * torch.cat(slice_dict_a,dim = 1) + wb * torch.cat(slice_dict_b,dim = 1)
        slice = self.softmax(final_cost_volume)
        disparity_samples = torch.arange(0, 97, dtype=slice.dtype, device=slice.device).view(1, 97, 1, 1)
        
        disparity_samples = disparity_samples.repeat(slice.size()[0],1,slice.size()[2],slice.size()[3])
        half_disp = torch.sum(disparity_samples * slice,dim = 1).unsqueeze(1)
        out2 = F.interpolate(half_disp * 2.0, scale_factor=(2.0, 2.0),
                                      mode='bilinear',align_corners =False).squeeze(1)
                                            
        return out2,out2


