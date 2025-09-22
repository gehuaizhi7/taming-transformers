import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TinyImageNetTrain(Dataset):
    """Tiny ImageNet Training Dataset"""
    
    def __init__(self, data_path, size=64, crop_size=None):
        self.data_path = data_path
        self.size = size
        self.crop_size = crop_size if crop_size is not None else size
        
        # Load class IDs
        with open(os.path.join(data_path, 'wnids.txt'), 'r') as f:
            self.class_ids = [line.strip() for line in f.readlines()]
        
        # Create class to index mapping
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        
        train_dir = os.path.join(data_path, 'train')
        for class_id in self.class_ids:
            class_dir = os.path.join(train_dir, class_id, 'images')
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_id])
        
        print(f"Loaded {len(self.image_paths)} training images from {len(self.class_ids)} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = (image * 2.0) - 1.0  # Convert to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        return {"image": image, "label": label}

class TinyImageNetValidation(Dataset):
    """Tiny ImageNet Validation Dataset"""
    
    def __init__(self, data_path, size=64, crop_size=None):
        self.data_path = data_path
        self.size = size
        self.crop_size = crop_size if crop_size is not None else size
        
        # Load class IDs
        with open(os.path.join(data_path, 'wnids.txt'), 'r') as f:
            self.class_ids = [line.strip() for line in f.readlines()]
        
        # Create class to index mapping
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        
        # Load validation annotations
        val_annotations = {}
        with open(os.path.join(data_path, 'val', 'val_annotations.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                val_annotations[img_name] = class_id
        
        # Load all validation image paths and labels
        self.image_paths = []
        self.labels = []
        
        val_dir = os.path.join(data_path, 'val', 'images')
        for img_name in os.listdir(val_dir):
            if img_name.endswith('.JPEG') and img_name in val_annotations:
                self.image_paths.append(os.path.join(val_dir, img_name))
                class_id = val_annotations[img_name]
                self.labels.append(self.class_to_idx[class_id])
        
        print(f"Loaded {len(self.image_paths)} validation images from {len(self.class_ids)} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = (image * 2.0) - 1.0  # Convert to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        return {"image": image, "label": label}



