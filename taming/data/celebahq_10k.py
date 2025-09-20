"""
CelebA-HQ 10K subset dataset for training
"""

import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CelebAHQ10KTrain(Dataset):
    """CelebA-HQ 10K training dataset"""
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        super().__init__()
        self.coord = coord
        
        # Load CelebA-HQ 10K training data
        root = "data/celebahq"
        with open("data/celebahqtrain_10k.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
        
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h, w, _ = ex["image"].shape
                coord = np.arange(h*w).reshape(h, w, 1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        
        # Convert from HWC to CHW format for PyTorch
        if ex["image"].shape[-1] == 3:  # HWC format
            ex["image"] = ex["image"].transpose(2, 0, 1)  # Convert to CHW
        
        ex["class"] = 0  # CelebA-HQ class
        return ex


class CelebAHQ10KValidation(Dataset):
    """CelebA-HQ 10K validation dataset"""
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        super().__init__()
        self.coord = coord
        
        # Load CelebA-HQ 10K validation data
        root = "data/celebahq"
        with open("data/celebahqvalidation_10k.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
        
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h, w, _ = ex["image"].shape
                coord = np.arange(h*w).reshape(h, w, 1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        
        # Convert from HWC to CHW format for PyTorch
        if ex["image"].shape[-1] == 3:  # HWC format
            ex["image"] = ex["image"].transpose(2, 0, 1)  # Convert to CHW
        
        ex["class"] = 0  # CelebA-HQ class
        return ex
