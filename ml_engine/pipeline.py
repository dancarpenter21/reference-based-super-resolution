from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .alignment import align_videos
from .inference.inference import upscale_video
from .media import probe, validate_input, write_json
from .training.train import train_model
from .weights import ensure_pretrained


class Cancelled(InterruptedError):
    pass


def run_pipeline(
    low_path: str | Path,
    reference_path: str | Path,
    job_dir: str | Path,
    preset: str = "balanced",
    update: Callable[[str, float, str, dict | None], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    def emit(stage: str, progress: float, message: str, metrics: dict | None = None) -> None:
        if cancel and cancel():
            raise Cancelled("Job cancelled")
        if update:
            update(stage, progress, message, metrics)

    emit("analyzing", 0.01, "Probing input videos")
    low_info, ref_info = probe(low_path), probe(reference_path)
    validate_input(low_info)
    validate_input(ref_info)
    media_report = {"low": low_info.to_dict(), "reference": ref_info.to_dict()}
    write_json(job_dir / "media.json", media_report)

    emit("aligning", 0.04, "Searching for trustworthy overlap")
    alignment = align_videos(low_path, reference_path, low_info, ref_info)
    write_json(job_dir / "alignment.json", alignment.to_dict())

    emit("training", 0.08, "Preparing pretrained restoration model", alignment.to_dict())
    pretrained = ensure_pretrained()
    checkpoint, training = train_model(
        reference_path, low_path, pretrained, job_dir / "checkpoints", preset,
        progress=lambda p, m, metrics: emit("training", 0.08 + p * 0.42, m, metrics),
        cancel=cancel,
        paired_anchors=[anchor.__dict__ for anchor in alignment.anchors] if alignment.mode == "paired" else None,
    )
    write_json(job_dir / "training.json", training)

    output = job_dir / "result.mp4"
    emit("upscaling", 0.50, "Upscaling complete low-resolution video")
    inference = upscale_video(
        low_path, checkpoint, output,
        progress=lambda p, m, metrics: emit("upscaling", 0.50 + p * 0.47, m, metrics),
        cancel=cancel,
    )
    report = {
        "media": media_report,
        "alignment": alignment.to_dict(),
        "training": training,
        "inference": inference,
        "output": str(output),
    }
    write_json(job_dir / "report.json", report)
    emit("muxing", 0.99, "Finalizing playable MP4")
    return report
