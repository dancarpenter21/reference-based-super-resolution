import argparse
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ml_engine.dataset.dataset import ReferenceDataset
from ml_engine.models.generator import RefUNet

def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset
    train_set = ReferenceDataset(opt.data_root, downscale_factor=4)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    print(f"Dataset size: {len(train_set)}")

    # 2. Model
    model = RefUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(opt.output_dir, exist_ok=True)

    # 3. Training Loop
    for epoch in range(opt.epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            ref = batch['ref'].to(device)
            
            # Forward
            sr = model(lr, ref)
            loss = criterion(sr, hr)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{opt.epochs}] [Batch {i}/{len(train_loader)}] [Loss: {loss.item():.4f}]")
                
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, f"model_epoch_{epoch}.pth"))
        
        # Save Sample Result
        with torch.no_grad():
            save_image(torch.cat([hr[0], sr[0]], dim=2), os.path.join(opt.output_dir, f"val_{epoch}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to directory with training frames")
    parser.add_argument("--checkpoint_dir", type=str, default="ml_engine/checkpoints")
    parser.add_argument("--output_dir", type=str, default="data/results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    
    opt = parser.parse_args()
    train(opt)
