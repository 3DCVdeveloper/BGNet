# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import cv2
import torchvision
from PIL import ImageFilter
class KITTI_12_Dataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames,self.mask_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None
    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        elif len(splits[0]) == 3:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images,None
        else:
            disp_images = [x[2] for x in splits]
            mask_images = [x[3] for x in splits]
            return left_images, right_images, disp_images,mask_images
    def load_image(self, filename):
        return Image.open(filename).convert('L')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        if self.mask_filenames:
            mask = self.load_image(os.path.join(self.datapath, self.mask_filenames[index]))
        else:
            mask = None
        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
            if self.mask_filenames:
                mask = np.asarray(mask)
                temp = mask > 0
                disparity = disparity * temp
        else:
            disparity = None

        if self.training:
            #rgb2gray
            left_img = left_img.convert('L')
            right_img = right_img.convert('L')
                
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            
            left_img = np.ascontiguousarray(left_img, dtype=np.float32)
            right_img = np.ascontiguousarray(right_img, dtype=np.float32)
            # to tensor, normalize
            preprocess = get_transform()
            left_img = preprocess(left_img)
            right_img = preprocess(right_img)
            disparity = np.expand_dims(disparity, 0)
            # return [left_img,right_img],-disparity
            return {"left": left_img,
                   "right": right_img,
                   "disparity": disparity}

        else:
            w, h = left_img.size
            top_pad = 384 - h
            right_pad = 1280 - w
            assert top_pad > 0 and right_pad > 0

                                   
            left_img = np.ascontiguousarray(left_img, dtype=np.float32)
            right_img = np.ascontiguousarray(right_img, dtype=np.float32)
                        
            left_img = np.lib.pad(left_img, ((top_pad, 0), (0, right_pad)), mode='symmetric', )
            right_img = np.lib.pad(right_img, ((top_pad, 0), (0, right_pad)), mode='symmetric')
            disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            
            preprocess = get_transform()
            left_img = preprocess(left_img)
            right_img = preprocess(right_img)

            disparity = np.expand_dims(disparity, 0)
            # return [left_img,right_img],-disparity
            return {"left": left_img,
                   "right": right_img,
                   "disparity": disparity}