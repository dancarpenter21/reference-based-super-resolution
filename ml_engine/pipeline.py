from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .alignment import (
    AlignmentReport, analyze_alignment, build_pair_manifest, segment_from_dict, validate_segments,
)
from .inference.inference import upscale_video
from .media import create_navigation_proxy, probe, validate_input, write_json
from .training.train import train_model
from .weights import ensure_pretrained


class Cancelled(InterruptedError):
    pass


def analyze_pipeline(
    low_path: str | Path,
    reference_path: str | Path,
    job_dir: str | Path,
    update: Callable[[str, float, str, dict | None], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    def emit(progress: float, message: str) -> None:
        if cancel and cancel():
            raise Cancelled("Job cancelled")
        if update:
            update("analyzing", progress, message, None)

    emit(0.01, "Probing input videos")
    low_info, ref_info = probe(low_path), probe(reference_path)
    validate_input(low_info)
    validate_input(ref_info)
    media_report = {"low": low_info.to_dict(), "reference": ref_info.to_dict()}
    write_json(job_dir / "media.json", media_report)
    emit(0.03, "Building navigation media")
    create_navigation_proxy(low_path, job_dir / "low-proxy.mp4")
    create_navigation_proxy(reference_path, job_dir / "reference-proxy.mp4")
    emit(0.05, "Finding shared sections")
    review = analyze_alignment(low_path, reference_path, low_info, ref_info)
    review["media"] = media_report
    review["proxy_urls"] = {
        "low": "low-proxy.mp4", "reference": "reference-proxy.mp4",
    }
    write_json(job_dir / "match-review.json", review)
    return review


def process_pipeline(
    low_path: str | Path,
    reference_path: str | Path,
    job_dir: str | Path,
    preset: str,
    review: dict,
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

    low_info, ref_info = probe(low_path), probe(reference_path)
    validate_input(low_info)
    validate_input(ref_info)
    media_report = {"low": low_info.to_dict(), "reference": ref_info.to_dict()}
    write_json(job_dir / "media.json", media_report)
    mode = review.get("approved_mode", "paired")
    segments = validate_segments(review.get("segments", []), low_info, ref_info) if mode == "paired" else []
    manifest = build_pair_manifest(segments, low_info, ref_info)
    write_json(job_dir / "pair-manifest.json", {"pairs": manifest})
    confidence = sum(s.confidence for s in segments) / len(segments) if segments else 0.0
    verified = sum(s.low_end.time_seconds - s.low_start.time_seconds for s in segments)
    alignment = AlignmentReport(
        mode="paired" if manifest else "unpaired", confidence=confidence,
        verified_seconds=verified, anchors=(), segments=tuple(segments), pair_count=len(manifest),
        review_revision=int(review.get("revision", 1)),
        warning=None if manifest else (
            "Reference-only mode selected; using synthetic reference pairs."
            if review.get("matching_mode") == "reference_only"
            else "User approved unpaired adaptation; using synthetic reference pairs."
        ),
    )
    write_json(job_dir / "alignment.json", alignment.to_dict())
    emit("training", 0.08, "Preparing pretrained restoration model", alignment.to_dict())
    pretrained = ensure_pretrained()
    checkpoint, training = train_model(
        reference_path, low_path, pretrained, job_dir / "checkpoints", preset,
        progress=lambda p, m, metrics: emit("training", 0.08 + p * 0.42, m, metrics),
        cancel=cancel, paired_manifest=manifest or None,
    )
    write_json(job_dir / "training.json", training)
    output = job_dir / "result.mp4"
    emit("upscaling", 0.50, "Upscaling complete low-resolution video")
    inference = upscale_video(
        low_path, checkpoint, output,
        progress=lambda p, m, metrics: emit("upscaling", 0.50 + p * 0.47, m, metrics), cancel=cancel,
    )
    report = {
        "media": media_report, "alignment": alignment.to_dict(),
        "matching_mode": review.get("matching_mode", "guided"),
        "review": {"revision": review.get("revision"), "approved_mode": mode, "segments": review.get("segments", [])},
        "training": training, "inference": inference, "output": str(output),
    }
    write_json(job_dir / "report.json", report)
    emit("muxing", 0.99, "Finalizing playable MP4")
    return report


def run_pipeline(
    low_path: str | Path,
    reference_path: str | Path,
    job_dir: str | Path,
    preset: str = "balanced",
    update: Callable[[str, float, str, dict | None], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict:
    """CLI compatibility: analyze, then conservatively run unpaired unless review was approved."""
    job_dir = Path(job_dir)
    review_path = job_dir / "match-review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else analyze_pipeline(
        low_path, reference_path, job_dir, update, cancel
    )
    if "approved_mode" not in review:
        review["approved_mode"] = "unpaired"
    return process_pipeline(low_path, reference_path, job_dir, preset, review, update, cancel)
