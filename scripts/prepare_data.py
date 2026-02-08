import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.services.video_processor import VideoProcessor

def prepare_data(video_path, output_dir=None):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    print(f"Processing {video_path}...")
    
    # Use backend service to extract frames
    # Note: VideoProcessor.extract_frames returns the output directory path
    # It defaults to save in 'data/processed' if called directly without modification, 
    # but let's check its implementation.
    # It uses PROCESSED_DIR which is project_root/data/processed.
    
    # If we want a custom output dir for training data (e.g. data/train_hr), 
    # we might need to move the files or modify the processor.
    # For now, let's just use the default processing and print where it went.
    
    try:
        # We can pass a specific FPS if needed, defaulting to 1 for now or 30?
        # For training, we probably want all frames or a high sampling rate.
        # Let's extract at native FPS (by passing a high number? logic says: if fps < video_fps else 1)
        # To get ALL frames, we need to modify logic or pass a very high FPS.
        # Looking at VideoProcessor: `frame_interval = int(video_fps / fps) if fps < video_fps else 1`
        # So passing fps=1000 (likely > video_fps) results in interval=1 (every frame).
        
        output_path = VideoProcessor.extract_frames(str(video_path), fps=1000)
        print(f"Frames extracted to: {output_path}")
        print("You can now run training using this directory as --data_root")
        
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to input video file")
    
    args = parser.parse_args()
    prepare_data(args.video_path)
