import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ml_engine.inference.composition import compose_timeline
from ml_engine.media import ffmpeg_executable, probe


class TinyUpscaler(torch.nn.Module):
    def forward(self, value):
        return F.interpolate(value, scale_factor=2, mode="nearest")


def make_solid_video(path, color, frames=3, fps=6):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (80, 60))
    for _ in range(frames):
        writer.write(np.full((60, 80, 3), color, np.uint8))
    writer.release()


def test_composer_keeps_reference_then_upscaled_supplemental_difference(tmp_path):
    low, reference = tmp_path / "low.mp4", tmp_path / "reference.mp4"
    output, checkpoint = tmp_path / "result.mp4", tmp_path / "checkpoint.pth"
    make_solid_video(low, (20, 220, 20))
    make_solid_video(reference, (20, 20, 220))
    torch.save({}, checkpoint)
    spans = [{
        "id": "different", "kind": "difference",
        "low_range": {"start_frame": 0, "end_frame": 3},
        "reference_range": {"start_frame": 0, "end_frame": 3},
        "status": None,
    }]

    report = compose_timeline(
        low, reference, checkpoint, output, spans, output_resolution="reference",
        device=torch.device("cpu"), model_factory=TinyUpscaler,
    )

    info = probe(output)
    assert (info.width, info.height, info.frame_count) == (80, 60, 6)
    assert report["upscaled_low_frames"] == 3
    assert [clip["source"] for clip in report["clips"]] == ["reference", "low"]
    cap = cv2.VideoCapture(str(output))
    ok, reference_frame = cap.read()
    assert ok
    cap.set(cv2.CAP_PROP_POS_FRAMES, 4)
    ok, low_frame = cap.read()
    cap.release()
    assert ok
    assert reference_frame[..., 2].mean() > reference_frame[..., 1].mean() * 2
    assert low_frame[..., 1].mean() > low_frame[..., 2].mean() * 2


def test_composer_follows_clip_audio_and_inserts_silence(tmp_path):
    low, silent_reference = tmp_path / "low.mp4", tmp_path / "silent-reference.mp4"
    reference, output, checkpoint = tmp_path / "reference.mp4", tmp_path / "result.mp4", tmp_path / "checkpoint.pth"
    make_solid_video(low, (20, 220, 20), frames=6)
    make_solid_video(silent_reference, (20, 20, 220), frames=6)
    subprocess.run([
        ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(silent_reference), "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(reference),
    ], check=True)
    torch.save({}, checkpoint)
    spans = [{
        "id": "different", "kind": "difference",
        "low_range": {"start_frame": 0, "end_frame": 6},
        "reference_range": {"start_frame": 0, "end_frame": 6}, "status": None,
    }]

    report = compose_timeline(
        low, reference, checkpoint, output, spans, output_resolution="reference",
        device=torch.device("cpu"), model_factory=TinyUpscaler,
    )

    assert probe(output).has_audio
    assert report["audio"] == "source_following"
    decoded = subprocess.run([
        ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-i", str(output),
        "-vn", "-ac", "1", "-ar", "48000", "-f", "s16le", "pipe:1",
    ], capture_output=True, check=True).stdout
    samples = np.frombuffer(decoded, dtype="<i2").astype(np.float32)
    midpoint = len(samples) // 2
    assert np.sqrt(np.mean(samples[:midpoint] ** 2)) > 100
    assert np.sqrt(np.mean(samples[midpoint:] ** 2)) < 20
