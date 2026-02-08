from app.services.video_processor import VideoProcessor
import os

try:
    print(f"Testing extraction on test_video.mp4")
    if not os.path.exists("test_video.mp4"):
        print("test_video.mp4 does not exist")
        exit(1)
        
    output_dir = VideoProcessor.extract_frames("test_video.mp4")
    print(f"Extraction successful. Output dir: {output_dir}")
    print(f"Files: {os.listdir(output_dir)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
