from io import BytesIO

import cv2
import numpy as np

from app.api import endpoints
from app.services.job_store import JobStore


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_requires_two_supported_videos(client):
    response = client.post(
        "/api/v1/jobs",
        files={
            "low_video": ("low.txt", BytesIO(b"bad"), "text/plain"),
            "reference_video": ("reference.mp4", BytesIO(b"bad"), "video/mp4"),
        },
    )
    assert response.status_code == 400


def test_create_rejects_unknown_matching_mode(client):
    response = client.post(
        "/api/v1/jobs",
        data={"matching_mode": "automatic_unreviewed"},
        files={
            "low_video": ("low.mp4", BytesIO(b"placeholder"), "video/mp4"),
            "reference_video": ("reference.mp4", BytesIO(b"placeholder"), "video/mp4"),
        },
    )
    assert response.status_code == 400
    assert "matching mode" in response.json()["detail"]


def test_unknown_job_is_404(client):
    assert client.get("/api/v1/jobs/not-real").status_code == 404


def test_list_and_cancel_all_jobs(client, tmp_path, monkeypatch):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    queued = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "queued"), "quick")
    completed = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "completed"), "balanced")
    jobs.update(completed["id"], state="completed", stage="completed", progress=1.0)
    monkeypatch.setattr(endpoints, "store", jobs)

    listed = client.get("/api/v1/jobs")
    cancelled = client.post("/api/v1/jobs/cancel-all")

    assert listed.status_code == 200
    assert [job["id"] for job in listed.json()["jobs"]] == [queued["id"], completed["id"]]
    assert all(job["matching_mode"] == "guided" for job in listed.json()["jobs"])
    assert cancelled.status_code == 200
    assert cancelled.json()["affected"] == 1
    states = {job["id"]: job["state"] for job in cancelled.json()["jobs"]}
    assert states == {queued["id"]: "cancelled", completed["id"]: "completed"}


def test_system_status_and_gpu_recheck(client, monkeypatch):
    class Diagnostics:
        def __init__(self):
            self.started = 0

        def status(self):
            return {"state": "checking", "error": None}

        def start_check(self):
            self.started += 1
            return True

    diagnostics = Diagnostics()
    monkeypatch.setattr(endpoints, "gpu_diagnostics", diagnostics)
    monkeypatch.setattr(endpoints.worker, "status", lambda: {
        "state": "idle", "current_job_id": None, "started_at": "now", "fatal_error": None,
    })
    monkeypatch.setattr(endpoints.store, "active_counts", lambda: {
        "queued": 2, "processing": 0, "outstanding": 2,
    })

    status = client.get("/api/v1/system/status")
    recheck = client.post("/api/v1/system/gpu/recheck")

    assert status.status_code == 200
    assert status.json()["backend"] == {"state": "online"}
    assert status.json()["worker"]["state"] == "idle"
    assert status.json()["queue"]["queued"] == 2
    assert recheck.status_code == 202
    assert recheck.json()["state"] == "checking"
    assert diagnostics.started == 1


def test_match_review_save_frame_and_approve(client, tmp_path, monkeypatch):
    def make_video(path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
        for index in range(30):
            writer.write(np.full((60, 80, 3), index * 4, np.uint8))
        writer.release()

    low, reference = tmp_path / "low.mp4", tmp_path / "reference.mp4"
    make_video(low)
    make_video(reference)
    jobs = JobStore(tmp_path / "review.sqlite3")
    job = jobs.create(str(low), str(reference), str(tmp_path / "job"), "quick")
    segment = {
        "id": "segment-1",
        "low_start": {"frame_index": 1, "pts": 1, "time_seconds": .1},
        "low_end": {"frame_index": 20, "pts": 20, "time_seconds": 2.0},
        "reference_start": {"frame_index": 1, "pts": 1, "time_seconds": .1},
        "reference_end": {"frame_index": 20, "pts": 20, "time_seconds": 2.0},
        "confidence": .9, "origin": "automatic", "status": "proposed",
    }
    review = {
        "revision": 1, "segments": [segment],
        "media": {"low": probe_dict(low), "reference": probe_dict(reference)},
        "summary": {"proposed_segments": 1, "matched_seconds": 1.9},
    }
    jobs.update(
        job["id"], state="awaiting_match_review", stage="awaiting_match_review", phase="review",
        review_data=review, review_revision=1,
    )
    monkeypatch.setattr(endpoints, "store", jobs)
    monkeypatch.setattr(endpoints.worker, "notify", lambda: None)

    fetched = client.get(f"/api/v1/jobs/{job['id']}/match-review")
    frame = client.get(f"/api/v1/jobs/{job['id']}/frames/low/5")
    snapped = client.post(
        f"/api/v1/jobs/{job['id']}/match-review/snap",
        json={"fixed_stream": "low", "fixed_frame_index": 5, "target_frame_index": 10},
    )
    segment["status"] = "confirmed"
    saved = client.put(
        f"/api/v1/jobs/{job['id']}/match-review", json={"revision": 1, "segments": [segment]},
    )
    approved = client.post(
        f"/api/v1/jobs/{job['id']}/match-review/approve",
        json={"revision": saved.json()["revision"], "mode": "paired"},
    )

    assert fetched.status_code == 200
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    assert snapped.status_code == 200
    assert abs(snapped.json()["frame"]["frame_index"] - 5) <= 1
    assert saved.status_code == 200
    assert approved.status_code == 202
    assert approved.json()["state"] == "queued"
    assert jobs.get(job["id"])["phase"] == "processing"


def probe_dict(path):
    from ml_engine.media import probe
    return probe(path).to_dict()
