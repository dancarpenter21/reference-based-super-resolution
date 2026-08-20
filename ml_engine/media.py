from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
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
    fps_numerator: int = 0
    fps_denominator: int = 1
    time_base: str = ""

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
    rate = Fraction(fps).limit_denominator(100_000)
    return MediaInfo(
        str(path.resolve()), width, height, fps, frames, frames / fps, codec,
        has_audio, sar, vfr, rate.numerator, rate.denominator,
        f"{rate.denominator}/{rate.numerator}",
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


@lru_cache(maxsize=64)
def detect_content_bounds(info: MediaInfo) -> tuple[int, int, int, int]:
    """Find a persistent near-black outer border using fixed samples from the video."""
    full = (0, 0, info.width, info.height)
    if not Path(info.path).is_file():
        return full
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        return full
    row_profiles: list[np.ndarray] = []
    column_profiles: list[np.ndarray] = []
    sample_count = min(12, info.frame_count)
    indices = np.linspace(
        round((info.frame_count - 1) * 0.08),
        round((info.frame_count - 1) * 0.92),
        sample_count,
        dtype=int,
    )
    try:
        for index in np.unique(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != (info.height, info.width):
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # A true black bar is dark across almost its entire row/column. Using
            # the 90th percentile avoids treating ordinary dark scenery as a bar.
            row_profiles.append(np.percentile(gray, 90, axis=1))
            column_profiles.append(np.percentile(gray, 90, axis=0))
    finally:
        cap.release()
    if len(row_profiles) < 3:
        return full

    persistent_rows = np.percentile(np.stack(row_profiles), 75, axis=0) < 16
    persistent_columns = np.percentile(np.stack(column_profiles), 75, axis=0) < 16

    def edge_run(mask: np.ndarray, reverse: bool, limit: int) -> int:
        count = 0
        for dark in (mask[::-1] if reverse else mask):
            if not dark:
                break
            count += 1
        return count if count <= limit else 0

    top = edge_run(persistent_rows, False, max(1, round(info.height * 0.2)))
    bottom = edge_run(persistent_rows, True, max(1, round(info.height * 0.2)))
    left = edge_run(persistent_columns, False, max(1, round(info.width * 0.2)))
    right = edge_run(persistent_columns, True, max(1, round(info.width * 0.2)))
    # Ignore isolated codec-darkened edge pixels; they are not a meaningful border.
    top = top if top >= 2 else 0
    bottom = bottom if bottom >= 2 else 0
    left = left if left >= 2 else 0
    right = right if right >= 2 else 0
    if info.width - left - right < info.width // 2 or info.height - top - bottom < info.height // 2:
        return full
    return left, top, info.width - right, info.height - bottom


def normalize_reference_frame(frame: np.ndarray, info: MediaInfo) -> np.ndarray:
    """Remove persistent borders, correct SAR, crop to 4:3, and resize to the HR canvas."""
    left, top, right, bottom = detect_content_bounds(info)
    frame = frame[top:bottom, left:right]
    source_width = frame.shape[1]
    display_width = max(1, round(source_width * sar_value(info.sample_aspect_ratio)))
    if display_width != source_width:
        frame = cv2.resize(frame, (display_width, frame.shape[0]), interpolation=cv2.INTER_AREA)
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


def read_frame(path: str | Path, frame_index: int) -> np.ndarray:
    """Decode a specific CFR frame, seeking to its preceding keyframe internally."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise MediaError(f"Could not decode frame {frame_index}")
    return frame


def create_navigation_proxy(input_path: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path),
        "-vf", "scale='min(640,iw)':-2", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "27", "-force_key_frames", "expr:gte(t,n_forced*1)",
        "-movflags", "+faststart", str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise MediaError(f"Navigation proxy encode failed: {proc.stderr[-2000:]}")


def audio_fingerprints(path: str | Path, every_seconds: float = 1.0, sample_rate: int = 8000) -> np.ndarray:
    """Return compact, compression-tolerant audio descriptors; silence/no-audio returns no rows."""
    cmd = [
        ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode or not proc.stdout:
        return np.empty((0, 128), dtype=np.float32)
    audio = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
    chunk_size = max(1, round(sample_rate * every_seconds))
    descriptors: list[np.ndarray] = []
    for start in range(0, len(audio) - chunk_size + 1, chunk_size):
        chunk = audio[start:start + chunk_size]
        if float(np.sqrt(np.mean(chunk * chunk))) < 1e-4:
            descriptors.append(np.zeros(128, dtype=np.float32))
            continue
        blocks = np.array_split(np.abs(chunk), 64)
        envelope = np.array([block.mean() for block in blocks], dtype=np.float32)
        spectrum = np.log1p(np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))))
        spectral_bins = np.array([block.mean() for block in np.array_split(spectrum, 64)], dtype=np.float32)
        descriptor = np.concatenate((envelope, spectral_bins))
        descriptor = (descriptor - descriptor.mean()) / (descriptor.std() + 1e-6)
        descriptors.append(descriptor / (np.linalg.norm(descriptor) + 1e-6))
    return np.stack(descriptors) if descriptors else np.empty((0, 128), dtype=np.float32)


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
