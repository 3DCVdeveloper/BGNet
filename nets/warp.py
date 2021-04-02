# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import time
def disp_warp(right_input, disparity_samples, padding_mode='border'):
    device = right_input.get_device()
    left_y_coordinate = torch.arange(0.0, right_input.size()[3], device=device).repeat(right_input.size()[2])
    left_y_coordinate = left_y_coordinate.view(right_input.size()[2], right_input.size()[3])
    left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=right_input.size()[3] - 1)
    left_y_coordinate = left_y_coordinate.expand(right_input.size()[0], -1, -1) #[B,H,W]

    right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
    disparity_samples = disparity_samples.float()
    #[B,C,H,W] - [B,C,H,W]
    right_y_coordinate = left_y_coordinate.expand(
        disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples

    right_y_coordinate_1 = right_y_coordinate
    right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)
    # torch.cuda.synchronize()
    # start  = time.time()
    warped_right_feature_map = torch.gather(right_feature_map, dim=4, index=right_y_coordinate.expand(
        right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long()) 
    # torch.cuda.synchronize()
    # temp_time = time.time() - start
    # print('gather_time = {:3f}'.format(temp_time))
    right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1)
    warped_right_feature_map = (1 - ((right_y_coordinate_1 < 0) +
                                     (right_y_coordinate_1 > right_input.size()[3] - 1)).float()) * \
        (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)

    return warped_right_feature_map.squeeze(2)

