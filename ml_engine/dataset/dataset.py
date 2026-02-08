import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ReferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, downscale_factor=4):
        """
        Args:
            root_dir (str): Path to directory containing HR frames.
            transform (callable, optional): Optional transform to be applied on a sample.
            downscale_factor (int): Factor to downscale HR frames to create LR.
        """
        self.root_dir = Path(root_dir)
        self.image_files = sorted([f for f in self.root_dir.glob('*.png')])
        self.transform = transform
        self.downscale_factor = downscale_factor
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Load HR Image
        hr_path = self.image_files[idx]
        hr_image = Image.open(hr_path).convert('RGB')
        
        # 2. Select Reference Image
        # Strategy: Select a random frame from the same video sequence
        # For simplicity, we assume all images in root_dir belong to the same sequence
        # In a real scenario, we'd need to group by video ID
        ref_idx = random.randint(0, len(self.image_files) - 1)
        ref_path = self.image_files[ref_idx]
        ref_image = Image.open(ref_path).convert('RGB')

        # 3. Create LR Image (Downscale HR)
        w, h = hr_image.size
        new_w, new_h = w // self.downscale_factor, h // self.downscale_factor
        lr_image = hr_image.resize((new_w, new_h), Image.BICUBIC)

        # 4. Apply Transforms (if any)
        # Note: Keeps as PIL images if no transform, or converts here
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
            ref_image = self.transform(ref_image)
        else:
            hr_image = self.to_tensor(hr_image)
            lr_image = self.to_tensor(lr_image)
            ref_image = self.to_tensor(ref_image)

        return {
            'lr': lr_image,
            'hr': hr_image,
            'ref': ref_image
        }
