# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
import torch 
import numpy as np
import re
import torchvision.transforms as transforms

def mean_std_transform(img,flag):

    size = np.shape(img)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([3, height, width], 'float32')
    
    img = np.ascontiguousarray(img)
    r = img[:, :, 0]/255.0
    g = img[:, :, 1]/255.0
    b = img[:, :, 2]/255.0
    

    if(flag ==1):
        temp_data[0, :, :] = (r - 0.353720247746) 
        temp_data[1, :, :] = (g - 0.384273201227) 
        temp_data[2, :, :] = (b - 0.405834376812) 
    else:
        temp_data[0, :, :] = (r - 0.353581100702) 
        temp_data[1, :, :] = (g - 0.384512037039) 
        temp_data[2, :, :] = (b - 0.406228214502) 
    
    
    img = temp_data[0: 3, :, :]
    #img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    return img

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

def get_transform():

    normalize = __imagenet_stats
    t_list = [
        transforms.ToTensor(),
        #transforms.Normalize(**normalize),
    ]

    return transforms.Compose(t_list)


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
