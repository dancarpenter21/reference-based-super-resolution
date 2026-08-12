import cv2
import numpy as np

from ml_engine.alignment import align_videos
from ml_engine.media import probe


def write_video(path, seed, frames=90, fps=10):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    for index in range(frames):
        frame = np.roll(base, index // 2, axis=1)
        cv2.putText(frame, str(index), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_unrelated_videos_fall_back(tmp_path):
    low, ref = tmp_path / "low.mp4", tmp_path / "ref.mp4"
    write_video(low, 1)
    write_video(ref, 2)
    report = align_videos(low, ref, probe(low), probe(ref), sample_seconds=1)
    assert report.mode == "unpaired"
    assert report.warning
