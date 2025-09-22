"""
Tiny CelebA-HQ dataset for debugging (50 train, 10 val images)
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CelebAHQTinyTrain(Dataset):
    def __init__(self, size=256, crop_size=256, **kwargs):
        self.size = size
        self.crop_size = crop_size
        
        # Read image paths
        with open("data/celebahqtrain_tiny.txt", "r") as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.image_paths)} training images for tiny dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Resize and crop
        image = image.resize((self.size, self.size), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format for PyTorch
        if image.shape[-1] == 3:  # HWC format
            image = image.transpose(2, 0, 1)  # Convert to CHW
        
        ex = {
            "image": image,
            "class": 0  # CelebA-HQ class
        }
        
        return ex

class CelebAHQTinyValidation(Dataset):
    def __init__(self, size=256, crop_size=256, **kwargs):
        self.size = size
        self.crop_size = crop_size
        
        # Read image paths
        with open("data/celebahqvalidation_tiny.txt", "r") as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.image_paths)} validation images for tiny dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Resize and crop
        image = image.resize((self.size, self.size), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format for PyTorch
        if image.shape[-1] == 3:  # HWC format
            image = image.transpose(2, 0, 1)  # Convert to CHW
        
        ex = {
            "image": image,
            "class": 0  # CelebA-HQ class
        }
        
        return ex






