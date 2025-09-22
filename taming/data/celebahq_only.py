"""
CelebA-HQ only dataset for training
"""

import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CelebAHQOnlyTrain(Dataset):
    """CelebA-HQ only training dataset"""
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        super().__init__()
        self.coord = coord
        
        # Load CelebA-HQ training data
        root = "data/celebahq"
        with open("data/celebahqtrain.txt", "r") as f:
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
        
        ex["class"] = 0  # CelebA-HQ class
        return ex


class CelebAHQOnlyValidation(Dataset):
    """CelebA-HQ only validation dataset"""
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        super().__init__()
        self.coord = coord
        
        # Load CelebA-HQ validation data
        root = "data/celebahq"
        with open("data/celebahqvalidation.txt", "r") as f:
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
        
        ex["class"] = 0  # CelebA-HQ class
        return ex









