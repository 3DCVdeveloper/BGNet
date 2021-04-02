# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function, division
import argparse
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import time
from datasets import __datasets__
import gc
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
parser = argparse.ArgumentParser(description='BGNet')
parser.add_argument('--model', default='bgnet', help='select a model structure')
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--savepath', required=True, help='save path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--resume', required=True, help='the directory to save logs and checkpoints')

args = parser.parse_args()
datapath = args.datapath
StereoDataset = __datasets__[args.dataset]
kitti_real_test = args.testlist
kitti_real_test_dataset = StereoDataset(datapath, kitti_real_test, False)
TestImgLoader = DataLoader(kitti_real_test_dataset, batch_size= 1, shuffle=False, num_workers=1, drop_last=False)

if(args.model == 'bgnet'):
    model = BGNet().cuda()
elif(args.model == 'bgnet_plus'):
    model = BGNet_Plus().cuda()
# else:
    # print('wrong model')
    # return -1
sub_name = None    
if(args.dataset == 'kitti_12'):
    sub_name = 'testing/colored_0/'
elif(args.dataset == 'kitti'):
    sub_name = 'testing/image_2/'
# else:
    # print('wrong dataset')
    # return -1
checkpoint = torch.load(args.resume,map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint) 
model.eval()
for batch_idx, sample in enumerate(TestImgLoader):
    print('predict the sample:',batch_idx)
    imgL, imgR = sample['left'], sample['right']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    pred,_ = model(imgL, imgR)
    pred = pred[0].data.cpu().numpy() * 256
    # print('pred',pred.shape)
    filename = datapath + sub_name + '{:0>6d}'.format(batch_idx) +'_10.png'
    gt = Image.open(filename)
    w, h = gt.size
    top_pad = 384 - h
    right_pad = 1280 - w
    temp = pred[top_pad:, :-right_pad]
    os.makedirs(args.savepath,exist_ok=True)
    skimage.io.imsave(args.savepath + '{:0>6d}'.format(batch_idx) +'_10.png',temp.astype('uint16'))
    