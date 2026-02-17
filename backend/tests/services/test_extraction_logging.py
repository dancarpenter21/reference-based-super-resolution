import pytest
import logging
import os
from app.services.video_processor import VideoProcessor

# Path to the specific test video
TEST_VIDEO_PATH = os.path.join(os.path.dirname(__file__), '../resources/videos/SS-24.mp4')

def test_extract_frames_logging(caplog):
    """
    Verifies that extract_frames logs the video_fps and target_fps.
    """
    if not os.path.exists(TEST_VIDEO_PATH):
        pytest.skip(f"Test video not found at {TEST_VIDEO_PATH}")

    caplog.set_level(logging.DEBUG)
    
    # Run extraction
    VideoProcessor.extract_frames(TEST_VIDEO_PATH)
    
    # Check logs
    # We expect a log message like "Extracting all frames: video_fps=..."
    assert "Extracting all frames: video_fps=" in caplog.text
    
    print(f"\nCaptured Logs:\n{caplog.text}")
