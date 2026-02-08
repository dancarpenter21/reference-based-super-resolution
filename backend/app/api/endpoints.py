from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.services.video_processor import VideoProcessor
import shutil

router = APIRouter()

@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a video file. The file is saved and then processed in the background
    to extract frames.
    """
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video.")

        file_path = await VideoProcessor.save_upload(file)
        
        # Trigger background processing
        # In a real app with heavy load, use Celery/Redis queue instead
        background_tasks.add_task(VideoProcessor.extract_frames, file_path)

        return {
            "message": "Video uploaded successfully. Processing started.",
            "file_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
