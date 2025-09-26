import os
import glob
import numpy as np
from PIL import Image
import albumentations
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from taming.data.base import ImagePaths
import taming.data.utils as bdu


class ImageNet256Base(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self._prepare()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        # Ensure image is in CHW format (Channels, Height, Width)
        if 'image' in item:
            image = item['image']
            if len(image.shape) == 3 and image.shape[2] == 3:  # HWC format
                item['image'] = image.transpose(2, 0, 1)  # Convert to CHW
        return item

    def _prepare(self):
        raise NotImplementedError()

    def _load(self):
        # Get all image paths
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        self.abspaths = []
        self.relpaths = []
        self.class_labels = []
        self.class_names = []
        
        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(self.datadir) 
                            if os.path.isdir(os.path.join(self.datadir, d))])
        
        # Create class mapping
        class_dict = dict((class_name, i) for i, class_name in enumerate(class_dirs))
        
        for class_name in class_dirs:
            class_path = os.path.join(self.datadir, class_name)
            
            # Find all images in this class directory
            for ext in image_extensions:
                pattern = os.path.join(class_path, ext)
                images = glob.glob(pattern)
                
                for img_path in images:
                    self.abspaths.append(img_path)
                    rel_path = os.path.relpath(img_path, self.datadir)
                    self.relpaths.append(rel_path)
                    self.class_labels.append(class_dict[class_name])
                    self.class_names.append(class_name)

        print(f"Found {len(self.abspaths)} images in {len(class_dirs)} classes")
        
        # Apply subset if specified
        max_samples = retrieve(self.config, "max_samples", default=None)
        if max_samples is not None and max_samples < len(self.abspaths):
            print(f"Using subset of {max_samples} images (out of {len(self.abspaths)} total)")
            # Take first max_samples images
            self.abspaths = self.abspaths[:max_samples]
            self.relpaths = self.relpaths[:max_samples]
            self.class_labels = self.class_labels[:max_samples]
            self.class_names = self.class_names[:max_samples]
        
        labels = {
            "relpath": np.array(self.relpaths),
            "class_label": np.array(self.class_labels),
            "class_name": np.array(self.class_names),
        }
        
        self.data = ImagePaths(self.abspaths,
                               labels=labels,
                               size=retrieve(self.config, "size", default=0),
                               random_crop=self.random_crop)
        


class ImageNet256Train(ImageNet256Base):
    def __init__(self, config=None, **kwargs):
        # Handle data_path parameter and merge kwargs into config
        if config is None:
            config = {}
        # Merge kwargs into config (this is how instantiate_from_config passes params)
        config.update(kwargs)
        super().__init__(config)
    
    def _prepare(self):
        self.random_crop = retrieve(self.config, "ImageNet256Train/random_crop", default=True)
        self.datadir = retrieve(self.config, "data_path", default="data/imagenet-256")
        
        if not os.path.exists(self.datadir):
            raise FileNotFoundError(f"ImageNet-256 dataset not found at {self.datadir}")


class ImageNet256Validation(ImageNet256Base):
    def __init__(self, config=None, **kwargs):
        # Handle data_path parameter and merge kwargs into config
        if config is None:
            config = {}
        # Merge kwargs into config (this is how instantiate_from_config passes params)
        config.update(kwargs)
        super().__init__(config)
    
    def _prepare(self):
        self.random_crop = retrieve(self.config, "ImageNet256Validation/random_crop", default=False)
        self.datadir = retrieve(self.config, "data_path", default="data/imagenet-256")
        
        if not os.path.exists(self.datadir):
            raise FileNotFoundError(f"ImageNet-256 dataset not found at {self.datadir}")


def retrieve(config, key, default=None):
    """Helper function to retrieve values from config"""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    else:
        return getattr(config, key, default)
