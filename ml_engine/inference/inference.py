from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from ml_engine.config import OUTPUT_HEIGHT, OUTPUT_WIDTH
from ml_engine.media import MediaError, ffmpeg_executable, probe
from ml_engine.models.generator import RRDBNet, load_weights, output_4_by_3
from ml_engine.training.train import require_rocm


@torch.inference_mode()
def upscale_frame(model: torch.nn.Module, frame: np.ndarray, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        result = output_4_by_3(model, tensor)
    result = result.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def _stabilize(
    current_lr: np.ndarray,
    current_sr: np.ndarray,
    previous_lr: np.ndarray | None,
    previous_sr: np.ndarray | None,
) -> np.ndarray:
    if previous_lr is None or previous_sr is None:
        return current_sr
    small_prev = cv2.resize(previous_lr, (160, 120), interpolation=cv2.INTER_AREA)
    # Scene cuts must not propagate stale detail.
    small_cur = cv2.resize(current_lr, (160, 120), interpolation=cv2.INTER_AREA)
    if np.mean(cv2.absdiff(small_prev, small_cur)) > 45:
        return current_sr
    prev_gray = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(small_cur, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    flow = cv2.resize(flow, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= OUTPUT_WIDTH / 160
    flow[..., 1] *= OUTPUT_HEIGHT / 120
    yy, xx = np.mgrid[0:OUTPUT_HEIGHT, 0:OUTPUT_WIDTH].astype(np.float32)
    warped = cv2.remap(previous_sr, xx - flow[..., 0], yy - flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    disagreement = cv2.cvtColor(cv2.absdiff(current_sr, warped), cv2.COLOR_BGR2GRAY)
    confidence = np.clip((18.0 - disagreement) / 18.0, 0, 1)[..., None]
    weight = confidence * 0.15
    return np.clip(current_sr * (1 - weight) + warped * weight, 0, 255).astype(np.uint8)


def upscale_video(
    low_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    progress: Callable[[float, str, dict | None], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    device: torch.device | None = None,
    model_factory: Callable[[], torch.nn.Module] = RRDBNet,
) -> dict:
    info = probe(low_path)
    device = device or require_rocm()
    model = model_factory().to(device)
    load_weights(model, checkpoint_path)
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "-r", f"{info.fps:.8f}", "-i", "pipe:0", "-i", str(low_path),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-frames:v", str(info.frame_count),
        "-movflags", "+faststart", str(output_path),
    ]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(str(low_path))
    index = 0
    previous_lr = previous_sr = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cancel and cancel():
                raise InterruptedError("Upscaling cancelled")
            sr = upscale_frame(model, frame, device)
            sr = cv2.resize(sr, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
            current_lr = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
            sr = _stabilize(current_lr, sr, previous_lr, previous_sr)
            assert encoder.stdin is not None
            encoder.stdin.write(sr.tobytes())
            previous_lr = current_lr
            previous_sr = sr
            index += 1
            if progress and (index % 30 == 0 or index == info.frame_count):
                progress(index / info.frame_count, f"frame {index}/{info.frame_count}", None)
    finally:
        cap.release()
        if encoder.stdin:
            encoder.stdin.close()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    code = encoder.wait()
    if code:
        output_path.unlink(missing_ok=True)
        raise MediaError(f"FFmpeg encode failed: {stderr[-2000:]}")
    return {"frames": index, "fps": info.fps, "width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT}
