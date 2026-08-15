import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from preprocessing import load_and_preprocess

class NucleiDataset(Dataset):
    """PyTorch Dataset loader for nuclei images and their binary ground-truth masks."""
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(list(Path(image_dir).glob("*.png")))
        self.mask_paths = sorted(list(Path(mask_dir).glob("*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and normalize to [0, 1]
        img = load_and_preprocess(self.image_paths[idx], is_mask=False).astype(np.float32) / 255.0
        # Load mask and binarize to [0, 1]
        mask = load_and_preprocess(self.mask_paths[idx], is_mask=True).astype(np.float32)

        # PyTorch expects channel-first format: (Channels, Height, Width)
        img_tensor = torch.tensor(img).unsqueeze(0)  # Shape: (1, 256, 256)
        mask_tensor = torch.tensor(mask).unsqueeze(0) # Shape: (1, 256, 256)

        return img_tensor, mask_tensor


class SimpleUNet(nn.Module):
    """A compact, lightweight U-Net architecture for fast educational training."""
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder (Contracting path)
        self.enc1_conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2_conv = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        
        # Decoder (Expanding path)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2_conv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1_conv = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        
        # Final output layer
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1_conv(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2_conv(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        bn = self.bottleneck(p2)
        
        # Decoder with skip connections
        u2 = self.up2(bn)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2_conv(u2)
        
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1_conv(u1)
        
        return torch.sigmoid(self.out_conv(d1))


def calculate_metrics(pred, target, threshold=0.5):
    """Calculates Dice Coefficient and Intersection over Union (IoU)."""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    intersection = (pred_bin * target_bin).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + target_bin.sum(dim=(1,2,3)) - intersection
    
    dice = (2.0 * intersection + 1e-6) / (pred_bin.sum(dim=(1,2,3)) + target_bin.sum(dim=(1,2,3)) + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return dice.mean().item(), iou.mean().item()