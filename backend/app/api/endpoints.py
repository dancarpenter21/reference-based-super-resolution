from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.services.job_store import store
from app.services.gpu_diagnostics import gpu_diagnostics
from app.services.job_worker import worker
from ml_engine.config import PRESETS
from ml_engine.alignment import frame_ref, validate_segments
from ml_engine.media import normalize_reference_frame, probe, read_frame

router = APIRouter()
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v"}


async def save_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            stream.write(chunk)
    await upload.close()
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty")


def public_job(job: dict) -> dict:
    job_id = job["id"]
    value = {
        key: job[key] for key in (
            "id", "state", "stage", "progress", "message", "preset", "metrics",
            "warning", "error", "created_at", "updated_at",
        )
    }
    value["alignment_mode"] = job.get("alignment_mode") or (job["metrics"] or {}).get("mode")
    value["matching_mode"] = job.get("matching_mode", "guided")
    value["needs_review"] = job["state"] == "awaiting_match_review"
    review = job.get("review_data") or {}
    value["match_summary"] = review.get("summary")
    value["eta_seconds"] = None
    if 0 < job["progress"] < 1 and job["state"] not in {"failed", "cancelled", "awaiting_match_review"}:
        elapsed = (datetime.now(UTC) - datetime.fromisoformat(job["created_at"])).total_seconds()
        value["eta_seconds"] = round(elapsed * (1 - job["progress"]) / job["progress"])
    if job["state"] == "completed":
        value.update({
            "result_url": f"/api/v1/jobs/{job_id}/result",
            "report_url": f"/api/v1/jobs/{job_id}/report",
        })
    return value


@router.post("/jobs", status_code=202)
async def create_job(
    low_video: Annotated[UploadFile, File()],
    reference_video: Annotated[UploadFile, File()],
    preset: Annotated[str, Form()] = "balanced",
    matching_mode: Annotated[str, Form()] = "guided",
):
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}")
    if matching_mode not in {"guided", "reference_only"}:
        raise HTTPException(status_code=400, detail=f"Unknown matching mode: {matching_mode}")
    low_suffix = Path(low_video.filename or "").suffix.lower()
    ref_suffix = Path(reference_video.filename or "").suffix.lower()
    if low_suffix not in ALLOWED_SUFFIXES or ref_suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Both inputs must be MP4 or MOV videos")
    _, job_dir, low_path, reference_path = store.new_paths(low_suffix, ref_suffix)
    try:
        await save_upload(low_video, low_path)
        await save_upload(reference_video, reference_path)
        job = store.create(
            str(low_path), str(reference_path), str(job_dir), preset, matching_mode=matching_mode,
        )
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    worker.notify()
    return public_job(job)


@router.get("/jobs")
def list_jobs():
    return {"jobs": [public_job(job) for job in store.list_all()]}


@router.post("/jobs/cancel-all")
def cancel_all_jobs():
    affected, jobs = store.cancel_all()
    return {"affected": affected, "jobs": [public_job(job) for job in jobs]}


@router.get("/system/status")
def system_status():
    return {
        "backend": {"state": "online"},
        "worker": worker.status(),
        "gpu": gpu_diagnostics.status(),
        "queue": store.active_counts(),
    }


@router.post("/system/gpu/recheck", status_code=202)
def recheck_gpu():
    gpu_diagnostics.start_check()
    return gpu_diagnostics.status()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return public_job(store.get(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


def review_job(job_id: str) -> dict:
    try:
        job = store.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("matching_mode", "guided") != "guided":
        raise HTTPException(status_code=409, detail="Frame matching is disabled for this job")
    if not job.get("review_data"):
        raise HTTPException(status_code=409, detail="Frame-match analysis is not ready")
    return job


def public_review(job_id: str, review: dict, revision: int) -> dict:
    value = {**review, "revision": revision}
    value["proxy_urls"] = {
        "low": f"/api/v1/jobs/{job_id}/navigation/low",
        "reference": f"/api/v1/jobs/{job_id}/navigation/reference",
    }
    return value


@router.get("/jobs/{job_id}/match-review")
def get_match_review(job_id: str):
    job = review_job(job_id)
    return public_review(job_id, job["review_data"], job["review_revision"])


@router.put("/jobs/{job_id}/match-review")
def save_match_review(job_id: str, payload: dict = Body(...)):
    job = review_job(job_id)
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise HTTPException(status_code=422, detail="segments must be a list")
    try:
        # Confirmed ranges receive strict validation; proposals may still be edited.
        validate_segments(segments, probe(job["low_path"]), probe(job["reference_path"]))
        matched = sum(
            max(0.0, float(s["low_end"]["time_seconds"]) - float(s["low_start"]["time_seconds"]))
            for s in segments if s.get("status") == "confirmed"
        )
        review = {
            **job["review_data"], "segments": segments,
            "summary": {
                "proposed_segments": sum(s.get("status") == "proposed" for s in segments),
                "confirmed_segments": sum(s.get("status") == "confirmed" for s in segments),
                "matched_seconds": matched,
                "warning": None,
            },
        }
        saved = store.save_review(job_id, review, int(payload.get("revision", -1)))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return public_review(job_id, saved["review_data"], saved["review_revision"])


@router.post("/jobs/{job_id}/match-review/expand")
def expand_match_seed(job_id: str, payload: dict = Body(...)):
    job = review_job(job_id)
    low_info, ref_info = probe(job["low_path"]), probe(job["reference_path"])
    try:
        low_index = int(payload["low_frame_index"])
        ref_index = int(payload["reference_frame_index"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Both seed frame indices are required")
    low_radius, ref_radius = round(low_info.fps * 5), round(ref_info.fps * 5)
    import uuid
    return {
        "id": str(uuid.uuid4()),
        "low_start": frame_ref(max(0, low_index - low_radius), low_info).__dict__,
        "low_end": frame_ref(min(low_info.frame_count - 1, low_index + low_radius), low_info).__dict__,
        "reference_start": frame_ref(max(0, ref_index - ref_radius), ref_info).__dict__,
        "reference_end": frame_ref(min(ref_info.frame_count - 1, ref_index + ref_radius), ref_info).__dict__,
        "confidence": 1.0, "origin": "manual", "status": "proposed",
    }


@router.post("/jobs/{job_id}/match-review/snap")
def snap_match_frame(job_id: str, payload: dict = Body(...)):
    job = review_job(job_id)
    fixed_stream = payload.get("fixed_stream")
    target_stream = "reference" if fixed_stream == "low" else "low" if fixed_stream == "reference" else None
    if target_stream is None:
        raise HTTPException(status_code=422, detail="fixed_stream must be low or reference")
    try:
        fixed_index = int(payload["fixed_frame_index"])
        target_index = int(payload["target_frame_index"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Both fixed and target frame indices are required")

    def comparison_frame(stream: str, index: int):
        path = job["low_path"] if stream == "low" else job["reference_path"]
        info = probe(path)
        frame = read_frame(path, index)
        frame = normalize_reference_frame(frame, info) if stream == "reference" else cv2.resize(frame, (640, 480))
        return cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA).astype("float32"), info

    try:
        fixed, _ = comparison_frame(fixed_stream, fixed_index)
        target_path = job["reference_path"] if target_stream == "reference" else job["low_path"]
        target_info = probe(target_path)
        target_index = max(0, min(target_index, target_info.frame_count - 1))
        radius = max(2, round(target_info.fps))
        best_index, best_error = target_index, float("inf")
        for candidate in range(max(0, target_index - radius), min(target_info.frame_count, target_index + radius + 1)):
            frame, _ = comparison_frame(target_stream, candidate)
            error = float(cv2.norm(fixed, frame, cv2.NORM_L1) / fixed.size)
            if error < best_error:
                best_index, best_error = candidate, error
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error))
    return {"stream": target_stream, "frame": frame_ref(best_index, target_info).__dict__, "score": best_error}


@router.post("/jobs/{job_id}/match-review/approve", status_code=202)
def approve_match_review(job_id: str, payload: dict = Body(...)):
    job = review_job(job_id)
    mode = payload.get("mode")
    if mode not in {"paired", "unpaired"}:
        raise HTTPException(status_code=422, detail="mode must be paired or unpaired")
    segments = job["review_data"].get("segments", [])
    if any(s.get("status") == "proposed" for s in segments):
        raise HTTPException(status_code=422, detail="Confirm or reject every proposed segment first")
    try:
        confirmed = validate_segments(segments, probe(job["low_path"]), probe(job["reference_path"]))
        if mode == "paired" and not confirmed:
            raise ValueError("Confirm at least one segment or continue unpaired")
        updated = store.approve_review(job_id, mode, int(payload.get("revision", -1)))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))
    worker.notify()
    return public_job(updated)


@router.get("/jobs/{job_id}/frames/{stream}/{frame_index}")
def exact_frame(job_id: str, stream: str, frame_index: int):
    job = review_job(job_id)
    if stream not in {"low", "reference"}:
        raise HTTPException(status_code=404, detail="Unknown video stream")
    path = job["low_path"] if stream == "low" else job["reference_path"]
    info = probe(path)
    if frame_index < 0 or frame_index >= info.frame_count:
        raise HTTPException(status_code=416, detail="Frame index is outside the video")
    try:
        frame = read_frame(path, frame_index)
        if stream == "reference":
            frame = normalize_reference_frame(frame, info)
        else:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("JPEG encode failed")
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error))
    return Response(
        content=encoded.tobytes(), media_type="image/jpeg",
        headers={"X-Frame-Index": str(frame_index), "X-Frame-Time": f"{frame_index / info.fps:.9f}"},
    )


@router.get("/jobs/{job_id}/navigation/{stream}")
def navigation_proxy(job_id: str, stream: str):
    if stream not in {"low", "reference"}:
        raise HTTPException(status_code=404, detail="Unknown video stream")
    return artifact(job_id, f"{stream}-proxy.mp4", "video/mp4")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return public_job(store.cancel(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


def artifact(job_id: str, filename: str, media_type: str):
    try:
        job = store.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    path = Path(job["job_dir"]) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact is not ready")
    return FileResponse(path, media_type=media_type, filename=filename)


@router.get("/jobs/{job_id}/result")
def result(job_id: str):
    return artifact(job_id, "result.mp4", "video/mp4")


@router.get("/jobs/{job_id}/report")
def report(job_id: str):
    return artifact(job_id, "report.json", "application/json")


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        job = store.delete(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))
    shutil.rmtree(job["job_dir"], ignore_errors=True)
