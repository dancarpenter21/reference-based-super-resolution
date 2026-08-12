from pathlib import Path

import cv2
import numpy as np
import pytest

from ml_engine.media import (
    MediaError,
    MediaInfo,
    normalize_reference_frame,
    probe,
    validate_input,
)


def make_video(path: Path, size=(80, 60), frames=12, fps=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), index * 10, np.uint8)
        writer.write(frame)
    writer.release()


def test_probe_and_validate(tmp_path):
    path = tmp_path / "sample.mp4"
    make_video(path)
    info = probe(path)
    assert (info.width, info.height, info.frame_count) == (80, 60, 12)
    assert info.fps == pytest.approx(12, abs=0.1)
    validate_input(info)


def test_reference_normalization_corrects_geometry():
    info = MediaInfo("x", 720, 480, 24, 1, 1 / 24, "h264", False, "10:11", False)
    result = normalize_reference_frame(np.zeros((480, 720, 3), np.uint8), info)
    assert result.shape == (480, 640, 3)


def test_rejects_wrong_container():
    info = MediaInfo("x.avi", 640, 480, 24, 10, 1, "xvid", False, "1:1", False)
    with pytest.raises(MediaError):
        validate_input(info)
