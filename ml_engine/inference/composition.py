from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import torch

from ml_engine.alignment import build_edit_manifest
from ml_engine.config import OUTPUT_HEIGHT, OUTPUT_WIDTH
from ml_engine.media import (
    MediaError, crop_to_fill, ffmpeg_executable, prepare_output_frame, probe,
    resolve_output_geometry,
)
from ml_engine.models.generator import RRDBNet, load_weights
from ml_engine.training.train import require_rocm

from .inference import _stabilize, upscale_frame


def _audio_filter(clips: list[dict], low_info, reference_info) -> tuple[str, str]:
    filters: list[str] = []
    labels: list[str] = []
    for index, clip in enumerate(clips):
        info = reference_info if clip["source"] == "reference" else low_info
        input_index = 2 if clip["source"] == "reference" else 1
        start = clip["source_range"]["start_frame"] / info.fps
        end = clip["source_range"]["end_frame"] / info.fps
        duration = clip["output_duration_seconds"]
        label = f"a{index}"
        if info.has_audio:
            filters.append(
                f"[{input_index}:a:0]atrim=start={start:.9f}:end={end:.9f},"
                f"asetpts=PTS-STARTPTS,aresample=48000,"
                f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                f"apad,atrim=duration={duration:.9f}[{label}]"
            )
        else:
            filters.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.9f}[{label}]"
            )
        labels.append(f"[{label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]")
    return ";".join(filters), "[aout]"


def _run_cancellable(cmd: list[str], cancel: Callable[[], bool] | None) -> tuple[int, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        try:
            _, stderr = proc.communicate(timeout=0.25)
            return proc.returncode, stderr
        except subprocess.TimeoutExpired:
            if cancel and cancel():
                proc.terminate()
                try:
                    proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                raise InterruptedError("Combined rendering cancelled")


def compose_timeline(
    low_path: str | Path,
    reference_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    spans: list[dict],
    output_resolution: str = "reference",
    progress: Callable[[float, str, dict | None], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    device: torch.device | None = None,
    model_factory: Callable[[], torch.nn.Module] = RRDBNet,
) -> dict:
    low_info, reference_info = probe(low_path), probe(reference_path)
    clips = build_edit_manifest(spans, low_info, reference_info)
    if not clips:
        raise MediaError("The reviewed timeline does not contain renderable footage")
    width, height = resolve_output_geometry(reference_info, output_resolution)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_path = output_path.with_name(f".{output_path.stem}-video.mp4")
    output_path.unlink(missing_ok=True)
    video_path.unlink(missing_ok=True)

    low_clips = [clip for clip in clips if clip["source"] == "low"]
    model = None
    if low_clips:
        device = device or require_rocm()
        model = model_factory().to(device)
        load_weights(model, checkpoint_path)
        model.eval()

    total_frames = sum(clip["output_end_frame"] - clip["output_start_frame"] for clip in clips)
    encoder_cmd = [
        ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{reference_info.fps:.8f}", "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-frames:v", str(total_frames),
        "-movflags", "+faststart", str(video_path),
    ]
    encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    caps = {
        "low": cv2.VideoCapture(str(low_path)),
        "reference": cv2.VideoCapture(str(reference_path)),
    }
    emitted = 0
    try:
        for clip in clips:
            source = clip["source"]
            info = reference_info if source == "reference" else low_info
            cap = caps[source]
            source_range = clip["source_range"]
            source_count = source_range["end_frame"] - source_range["start_frame"]
            output_count = clip["output_end_frame"] - clip["output_start_frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_range["start_frame"])
            decoded_index = source_range["start_frame"] - 1
            decoded = rendered = None
            previous_lr = previous_sr = None
            for offset in range(output_count):
                if cancel and cancel():
                    raise InterruptedError("Combined rendering cancelled")
                wanted = source_range["start_frame"] + min(
                    source_count - 1, int(offset * source_count / output_count),
                )
                while decoded_index < wanted:
                    ok, decoded = cap.read()
                    decoded_index += 1
                    if not ok or decoded is None:
                        raise MediaError(f"Could not decode {source} frame {decoded_index}")
                    if source == "low":
                        assert model is not None and device is not None
                        sr = upscale_frame(model, decoded, device)
                        sr = cv2.resize(sr, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                        current_lr = cv2.resize(
                            decoded, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC,
                        )
                        sr = _stabilize(current_lr, sr, previous_lr, previous_sr)
                        previous_lr, previous_sr = current_lr, sr
                        rendered = crop_to_fill(sr, (width, height))
                    else:
                        rendered = prepare_output_frame(decoded, info, (width, height))
                assert rendered is not None and encoder.stdin is not None
                encoder.stdin.write(rendered.tobytes())
                emitted += 1
                if progress and (emitted % 30 == 0 or emitted == total_frames):
                    progress(
                        emitted / total_frames, f"frame {emitted}/{total_frames}",
                        {"output_frames": emitted, "total_output_frames": total_frames},
                    )
    except Exception:
        if encoder.stdin:
            encoder.stdin.close()
        encoder.kill()
        encoder.wait()
        video_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    finally:
        for cap in caps.values():
            cap.release()
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    if encoder.wait():
        video_path.unlink(missing_ok=True)
        raise MediaError(f"FFmpeg video encode failed: {stderr[-2000:]}")

    try:
        has_any_audio = low_info.has_audio or reference_info.has_audio
        if has_any_audio:
            filter_graph, audio_label = _audio_filter(clips, low_info, reference_info)
            mux_cmd = [
                ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path), "-i", str(low_path), "-i", str(reference_path),
                "-filter_complex", filter_graph, "-map", "0:v:0", "-map", audio_label,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", str(output_path),
            ]
            code, mux_stderr = _run_cancellable(mux_cmd, cancel)
            if code:
                raise MediaError(f"FFmpeg audio mux failed: {mux_stderr[-2000:]}")
        else:
            video_path.replace(output_path)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        video_path.unlink(missing_ok=True)

    return {
        "frames": total_frames, "fps": reference_info.fps,
        "width": width, "height": height, "duration_seconds": total_frames / reference_info.fps,
        "resolution_choice": output_resolution, "audio": "source_following" if has_any_audio else "none",
        "reference_frames": sum(
            clip["source_range"]["end_frame"] - clip["source_range"]["start_frame"]
            for clip in clips if clip["source"] == "reference"
        ),
        "upscaled_low_frames": sum(
            clip["source_range"]["end_frame"] - clip["source_range"]["start_frame"]
            for clip in low_clips
        ),
        "clips": clips,
    }
