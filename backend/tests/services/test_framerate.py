import pytest
import cv2
import subprocess
import shutil
import os

# Path to the specific test video
TEST_VIDEO_PATH = os.path.join(os.path.dirname(__file__), '../resources/videos/SS-24.mp4')

def get_cv2_fps(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video with cv2: {video_path}")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def get_ffmpeg_fps(video_path):
    # Check if ffmpeg/ffprobe is available
    ffprobe_cmd = shutil.which("ffprobe")
    if not ffprobe_cmd:
        # Fallback to trying 'ffmpeg' if ffprobe not found directly (though usually packaged together)
        if shutil.which("ffmpeg"):
            # Check if we can use ffmpeg to get info
            cmd = ["ffmpeg", "-i", video_path]
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            # Parse output for "X fps"
            for line in result.stderr.split('\n'):
                if "fps," in line:
                    parts = line.split(',')
                    for part in parts:
                        if "fps" in part:
                            return float(part.strip().split(' ')[0])
        
        pytest.skip("ffprobe/ffmpeg not found in PATH")

    # Use ffprobe for cleaner output
    cmd = [
        ffprobe_cmd, 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=r_frame_rate", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
    # ffprobe returns "num/den" (e.g., 24000/1001 or 30/1)
    r_frame_rate = result.stdout.strip()
    if '/' in r_frame_rate:
        num, den = map(int, r_frame_rate.split('/'))
        return num / den
    return float(r_frame_rate)

@pytest.mark.skipif(not os.path.exists(TEST_VIDEO_PATH), reason="Test video SS-24.mp4 not found")
def test_framerate_consistency():
    """
    Validates that OpenCV and FFmpeg report the same framerate for the test video.
    """
    print(f"Testing video: {TEST_VIDEO_PATH}")
    
    cv2_fps = get_cv2_fps(TEST_VIDEO_PATH)
    print(f"OpenCV FPS: {cv2_fps}")
    
    ffmpeg_fps = get_ffmpeg_fps(TEST_VIDEO_PATH)
    print(f"FFmpeg FPS: {ffmpeg_fps}")
    
    # Assert they are close (floating point differences)
    # Using a small tolerance
    assert abs(cv2_fps - ffmpeg_fps) < 0.01, \
        f"Mismatch in FPS: OpenCV={cv2_fps}, FFmpeg={ffmpeg_fps}"
