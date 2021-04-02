# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from .kitti_dataset import KITTIDataset
from .kitti12_dataset import KITTI_12_Dataset
__datasets__ = {
    "kitti": KITTIDataset,
    "kitti_12": KITTI_12_Dataset,
}
