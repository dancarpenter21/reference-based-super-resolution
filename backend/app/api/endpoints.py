from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.services.job_store import store
from app.services.job_worker import worker
from ml_engine.config import PRESETS

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
    value["alignment_mode"] = (job["metrics"] or {}).get("mode")
    value["eta_seconds"] = None
    if 0 < job["progress"] < 1 and job["state"] not in {"failed", "cancelled"}:
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
):
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}")
    low_suffix = Path(low_video.filename or "").suffix.lower()
    ref_suffix = Path(reference_video.filename or "").suffix.lower()
    if low_suffix not in ALLOWED_SUFFIXES or ref_suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Both inputs must be MP4 or MOV videos")
    _, job_dir, low_path, reference_path = store.new_paths(low_suffix, ref_suffix)
    try:
        await save_upload(low_video, low_path)
        await save_upload(reference_video, reference_path)
        job = store.create(str(low_path), str(reference_path), str(job_dir), preset)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    worker.notify()
    return public_job(job)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return public_job(store.get(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


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
