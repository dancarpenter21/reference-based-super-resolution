import cv2
import numpy as np


def create_dummy_video(filename, duration=2, fps=30):
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(duration * fps):
        # Create a frame with a moving rectangle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i*2 % width, 100), (i*2 % width + 50, 150), (0, 255, 0), -1)
        out.write(frame)

    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_video("test_video.mp4")
