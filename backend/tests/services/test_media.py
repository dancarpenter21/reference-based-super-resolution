from pathlib import Path

import cv2
import numpy as np
import pytest

from ml_engine.media import (
    audio_fingerprints,
    detect_content_bounds,
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
    assert audio_fingerprints(path).shape == (0, 128)


def test_reference_normalization_corrects_geometry():
    info = MediaInfo("x", 720, 480, 24, 1, 1 / 24, "h264", False, "10:11", False)
    result = normalize_reference_frame(np.zeros((480, 720, 3), np.uint8), info)
    assert result.shape == (480, 640, 3)


def test_reference_normalization_removes_persistent_black_border(tmp_path):
    path = tmp_path / "bordered.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (120, 90))
    for index in range(24):
        frame = np.zeros((90, 120, 3), np.uint8)
        content = np.full((80, 100, 3), (60 + index * 3, 110, 180), np.uint8)
        cv2.circle(content, (20 + index, 40), 8, (220, 220, 220), -1)
        frame[5:85, 10:110] = content
        writer.write(frame)
    writer.release()

    info = probe(path)
    left, top, right, bottom = detect_content_bounds(info)
    assert (left, top) == pytest.approx((10, 5), abs=1)
    assert (right, bottom) == pytest.approx((110, 85), abs=1)

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 12)
    ok, frame = cap.read()
    cap.release()
    assert ok
    normalized = normalize_reference_frame(frame, info)
    assert normalized.shape == (480, 640, 3)
    assert normalized[0].mean() > 25
    assert normalized[-1].mean() > 25


def test_rejects_wrong_container():
    info = MediaInfo("x.avi", 640, 480, 24, 10, 1, "xvid", False, "1:1", False)
    with pytest.raises(MediaError):
        validate_input(info)
