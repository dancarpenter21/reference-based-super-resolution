import os
import cv2
import uuid
from pathlib import Path
from fastapi import UploadFile


# Define project root relative to this file
# absolute path of this file is backend/app/services/video_processor.py
# project root is backend/../..
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

UPLOAD_DIR = PROJECT_ROOT / "data/uploads"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class VideoProcessor:
    @staticmethod
    async def save_upload(file: UploadFile) -> str:
        """Saves an uploaded file to the data/uploads directory."""
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{file_id}{file_ext}"
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        return str(file_path)

    @staticmethod
    def extract_frames(video_path: str, fps: int = 1) -> str:
        """Extracts frames from the video at the given fps."""
        video_name = Path(video_path).stem
        output_dir = PROCESSED_DIR / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        count = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                frame_path = output_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(frame_path), frame)
                frame_idx += 1
            
            count += 1

        cap.release()
        return str(output_dir)
