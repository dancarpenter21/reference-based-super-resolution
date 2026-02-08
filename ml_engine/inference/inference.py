import argparse
import sys
import os
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ml_engine.models.generator import RefUNet

def inference(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = RefUNet().to(device)
    if os.path.exists(opt.checkpoint_path):
        model.load_state_dict(torch.load(opt.checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {opt.checkpoint_path}")
    else:
        print(f"Checkpoint not found at {opt.checkpoint_path}, using random init (for testing)")
    
    model.eval()
    
    input_dir = Path(opt.input_dir)
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input frames
    frames = sorted([f for f in input_dir.glob('*.png')])
    if not frames:
        print("No frames found in input directory")
        return

    # Transform
    to_tensor = transforms.ToTensor()
    
    # Select Reference (Naive: Use first frame of input sequence or provided path)
    if opt.ref_path:
        ref_img = Image.open(opt.ref_path).convert('RGB')
    else:
        # Fallback: use the first frame as reference (Naive)
        ref_img = Image.open(frames[0]).convert('RGB')
        
    ref_tensor = to_tensor(ref_img).unsqueeze(0).to(device)

    print(f"Starting inference on {len(frames)} frames...")
    
    with torch.no_grad():
        for i, frame_path in enumerate(frames):
            lr_img = Image.open(frame_path).convert('RGB')
            lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
            
            # Forward
            sr_tensor = model(lr_tensor, ref_tensor)
            
            # Save
            save_path = output_dir / f"sr_{frame_path.name}"
            save_image(sr_tensor.squeeze(0), save_path)
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(frames)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing LR frames")
    parser.add_argument("--ref_path", type=str, help="Path to reference HR image")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save SR frames")
    parser.add_argument("--checkpoint_path", type=str, default="ml_engine/checkpoints/model_epoch_9.pth")
    
    opt = parser.parse_args()
    inference(opt)
