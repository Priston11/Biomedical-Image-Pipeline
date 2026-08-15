import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class NucleiDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name
        
        # Load image in grayscale and normalize to [0, 1]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0) # Add channel dimension -> (1, 256, 256)
        
        # Load mask if it exists, otherwise generate a dummy blank mask for pipeline testing
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((256, 256), dtype=np.float32)
            
        mask = np.expand_dims(mask, axis=0) # Add channel dimension -> (1, 256, 256)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)