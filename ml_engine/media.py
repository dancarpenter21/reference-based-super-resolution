from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from .config import OUTPUT_HEIGHT, OUTPUT_WIDTH


class MediaError(ValueError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    has_audio: bool
    sample_aspect_ratio: str
    variable_frame_rate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def ffmpeg_executable() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"Video does not exist: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    cap.release()
    if width <= 0 or height <= 0 or not (0 < fps <= 240) or frames <= 0:
        raise MediaError("Invalid or unsupported video stream")
    # OpenCV does not expose every container field consistently. Ask FFmpeg for audio/SAR
    # without requiring a system ffprobe binary.
    proc = subprocess.run(
        [ffmpeg_executable(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, check=False,
    )
    details = proc.stderr
    has_audio = " Audio: " in details
    sar = "1:1"
    import re

    match = re.search(r"SAR (\d+[:/]\d+)", details)
    if match:
        sar = match.group(1).replace("/", ":")
    avg_match = re.search(r",\s*([0-9.]+) fps,", details)
    tbr_match = re.search(r",\s*([0-9.]+) tbr,", details)
    vfr = False
    if avg_match and tbr_match:
        vfr = abs(float(avg_match.group(1)) - float(tbr_match.group(1))) > 0.02
    return MediaInfo(
        str(path.resolve()), width, height, fps, frames, frames / fps, codec,
        has_audio, sar, vfr,
    )


def validate_input(info: MediaInfo) -> None:
    if Path(info.path).suffix.lower() not in {".mp4", ".mov", ".m4v"}:
        raise MediaError("Version 1 accepts MP4/MOV containers")
    if info.variable_frame_rate:
        raise MediaError("Variable-frame-rate video is not supported; transcode to constant FPS")
    if info.width * info.height > 3840 * 2160:
        raise MediaError("Inputs larger than 4K are not supported")


def sar_value(sar: str) -> float:
    try:
        return float(Fraction(sar.replace(":", "/")))
    except (ValueError, ZeroDivisionError):
        return 1.0


def normalize_reference_frame(frame: np.ndarray, info: MediaInfo) -> np.ndarray:
    """Correct SAR, center-crop to 4:3, and resize to the canonical HR canvas."""
    display_width = max(1, round(info.width * sar_value(info.sample_aspect_ratio)))
    if display_width != info.width:
        frame = cv2.resize(frame, (display_width, info.height), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    target_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT
    if w / h > target_ratio:
        new_w = round(h * target_ratio)
        left = (w - new_w) // 2
        frame = frame[:, left:left + new_w]
    elif w / h < target_ratio:
        new_h = round(w / target_ratio)
        top = (h - new_h) // 2
        frame = frame[top:top + new_h]
    return cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def sampled_frames(path: str | Path, every_seconds: float = 1.0) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps * every_seconds))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % interval == 0:
            yield index / fps, frame
        index += 1
    cap.release()


def encode_frames(
    frame_pattern: str | Path,
    low_video: str | Path,
    output_path: str | Path,
    fps: float,
) -> None:
    """Encode numbered PNGs and map the original low-video audio when available."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.8f}", "-i", str(frame_pattern), "-i", str(low_video),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise MediaError(f"FFmpeg encode failed: {proc.stderr[-2000:]}")


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
