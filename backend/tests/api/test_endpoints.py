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


def test_create_requires_review_and_valid_output_resolution(client):
    files = {
        "low_video": ("low.mp4", BytesIO(b"placeholder"), "video/mp4"),
        "reference_video": ("reference.mp4", BytesIO(b"placeholder"), "video/mp4"),
    }
    skipped = client.post("/api/v1/jobs", data={"matching_mode": "reference_only"}, files=files)
    assert skipped.status_code == 400
    assert "timeline review is required" in skipped.json()["detail"]

    invalid_size = client.post(
        "/api/v1/jobs", data={"output_resolution": "8k"},
        files={
            "low_video": ("low.mp4", BytesIO(b"placeholder"), "video/mp4"),
            "reference_video": ("reference.mp4", BytesIO(b"placeholder"), "video/mp4"),
        },
    )
    assert invalid_size.status_code == 400
    assert "output resolution" in invalid_size.json()["detail"]


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


def test_v2_alignment_edit_validation_and_approval(client, tmp_path, monkeypatch):
    def make_video(path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
        for index in range(30):
            writer.write(np.full((60, 80, 3), index * 4, np.uint8))
        writer.release()

    low, reference = tmp_path / "v2-low.mp4", tmp_path / "v2-reference.mp4"
    make_video(low)
    make_video(reference)
    jobs = JobStore(tmp_path / "v2.sqlite3")
    job = jobs.create(str(low), str(reference), str(tmp_path / "v2-job"), "quick")
    review = {
        "schema_version": 2, "revision": 1,
        "spans": [
            {"id": "before", "kind": "difference", "low_range": {"start_frame": 0, "end_frame": 5}, "reference_range": {"start_frame": 0, "end_frame": 5}, "origin": "automatic", "status": None, "confidence": None, "sequence_start_seconds": 0, "sequence_duration_seconds": .5},
            {"id": "shared", "kind": "match", "low_range": {"start_frame": 5, "end_frame": 20}, "reference_range": {"start_frame": 5, "end_frame": 20}, "origin": "automatic", "status": "proposed", "confidence": .95, "sequence_start_seconds": .5, "sequence_duration_seconds": 1.5},
            {"id": "after", "kind": "difference", "low_range": {"start_frame": 20, "end_frame": 30}, "reference_range": {"start_frame": 20, "end_frame": 30}, "origin": "automatic", "status": None, "confidence": None, "sequence_start_seconds": 2, "sequence_duration_seconds": 1},
        ],
        "media": {"low": probe_dict(low), "reference": probe_dict(reference)},
        "summary": {"proposed_blocks": 1, "confirmed_blocks": 0, "matched_seconds": 0},
    }
    jobs.update(job["id"], state="awaiting_match_review", stage="awaiting_match_review", phase="review", review_data=review, review_revision=1)
    monkeypatch.setattr(endpoints, "store", jobs)
    monkeypatch.setattr(endpoints.worker, "notify", lambda: None)

    automatic_draft = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": 1, "span_id": "shared", "operation": "set_match_draft",
        "draft": {
            "low_in": 6, "reference_in": 6, "in_time": .6,
            "low_out": None, "reference_out": None, "out_time": None,
        },
    })
    automatic_span = next(span for span in automatic_draft.json()["spans"] if span["id"] == "shared")
    assert automatic_span["origin"] == "automatic" and automatic_span["status"] == "proposed"
    assert automatic_span["adjustment_baseline"]["low_in"] == 5
    cleared_automatic = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": automatic_draft.json()["revision"], "span_id": "shared",
        "operation": "set_match_draft", "draft": None,
    })
    cleared_automatic_span = next(span for span in cleared_automatic.json()["spans"] if span["id"] == "shared")
    assert "adjustment_baseline" not in cleared_automatic_span

    invalid = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": cleared_automatic.json()["revision"], "span_id": "shared", "operation": "move_boundary", "boundary": "start",
        "low_frame": 20, "reference_frame": 20,
    })
    assert invalid.status_code == 422

    moved = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": cleared_automatic.json()["revision"], "span_id": "shared", "operation": "move_boundary", "boundary": "start",
        "low_frame": 6, "reference_frame": 6,
    })
    assert moved.status_code == 200
    assert moved.json()["spans"][0]["low_range"]["end_frame"] == 6
    assert moved.json()["spans"][1]["low_range"]["start_frame"] == 6

    saved_draft = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": moved.json()["revision"], "span_id": "shared", "operation": "set_match_draft",
        "draft": {
            "low_in": 7, "reference_in": 7, "in_time": .7,
            "low_out": None, "reference_out": None, "out_time": None,
        },
    })
    assert saved_draft.status_code == 200
    selected_draft = next(span for span in saved_draft.json()["spans"] if span["id"] == "shared")
    assert selected_draft["match_draft"]["low_in"] == 7
    assert selected_draft["origin"] == "manual" and selected_draft["status"] == "proposed"
    assert selected_draft["adjustment_baseline"] == {
        "low_in": 5, "low_out": 19, "reference_in": 5, "reference_out": 19,
    }
    persisted = client.get(f"/api/v1/jobs/{job['id']}/match-review")
    assert next(span for span in persisted.json()["spans"] if span["id"] == "shared")["match_draft"] == selected_draft["match_draft"]

    cleared_draft = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": saved_draft.json()["revision"], "span_id": "shared",
        "operation": "set_match_draft", "draft": None,
    })
    assert cleared_draft.status_code == 200
    cleared_span = next(span for span in cleared_draft.json()["spans"] if span["id"] == "shared")
    assert "match_draft" not in cleared_span and cleared_span["adjustment_baseline"]["low_in"] == 5

    complete_draft = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": cleared_draft.json()["revision"], "span_id": "shared", "operation": "set_match_draft",
        "draft": {
            "low_in": 7, "reference_in": 7, "in_time": .7,
            "low_out": 18, "reference_out": 18, "out_time": 1.8,
        },
    })
    ranged = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": complete_draft.json()["revision"], "span_id": "shared", "operation": "apply_match_draft",
    })
    assert ranged.status_code == 200
    ranged_spans = ranged.json()["spans"]
    selected = next(span for span in ranged_spans if span["id"] == "shared")
    assert selected["low_range"] == {"start_frame": 7, "end_frame": 19}
    assert selected["reference_range"] == {"start_frame": 7, "end_frame": 19}
    assert selected["origin"] == "manual" and selected["status"] == "proposed"
    assert selected["adjustment_baseline"]["low_in"] == 5
    assert "match_draft" not in selected
    assert ranged_spans[0]["low_range"]["end_frame"] == 7
    assert ranged_spans[-1]["low_range"]["start_frame"] == 19

    mismatched = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": ranged.json()["revision"], "span_id": "shared", "operation": "set_match_range",
        "low_start": 7, "low_end": 18, "reference_start": 7, "reference_end": 19,
    })
    assert mismatched.status_code == 200
    refused = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": mismatched.json()["revision"], "span_id": "shared",
        "operation": "set_status", "status": "confirmed",
    })
    assert refused.status_code == 422
    assert "missing or extra frames" in refused.json()["detail"]

    repaired = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": mismatched.json()["revision"], "span_id": "shared", "operation": "set_match_range",
        "low_start": 7, "low_end": 18, "reference_start": 7, "reference_end": 18,
    })
    confirmed = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": repaired.json()["revision"], "span_id": "shared", "operation": "set_status", "status": "confirmed",
    })
    assert confirmed.status_code == 200

    confirmed_draft = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": confirmed.json()["revision"], "span_id": "shared", "operation": "set_match_draft",
        "draft": {
            "low_in": 7, "reference_in": 7, "in_time": .7,
            "low_out": None, "reference_out": None, "out_time": None,
        },
    })
    confirmed_with_draft = next(span for span in confirmed_draft.json()["spans"] if span["id"] == "shared")
    assert confirmed_with_draft["status"] == "confirmed" and confirmed_with_draft["origin"] == "manual"
    restored_confirmed = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": confirmed_draft.json()["revision"], "span_id": "shared",
        "operation": "set_match_draft", "draft": None,
    })
    restored_span = next(span for span in restored_confirmed.json()["spans"] if span["id"] == "shared")
    assert restored_span["status"] == "confirmed" and "match_draft" not in restored_span

    difference = next(span for span in restored_confirmed.json()["spans"] if span["kind"] == "difference" and span.get("low_range") and span.get("reference_range"))
    difference_draft = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": restored_confirmed.json()["revision"], "span_id": difference["id"], "operation": "set_match_draft",
        "draft": {
            "low_in": difference["low_range"]["start_frame"],
            "reference_in": difference["reference_range"]["start_frame"],
            "in_time": difference["sequence_start_seconds"],
            "low_out": None, "reference_out": None, "out_time": None,
        },
    })
    blocked_by_draft = client.post(f"/api/v1/jobs/{job['id']}/match-review/approve", json={
        "revision": difference_draft.json()["revision"], "mode": "paired",
    })
    assert blocked_by_draft.status_code == 422
    assert "saved frame draft" in blocked_by_draft.json()["detail"]
    cleared_difference = client.patch(f"/api/v1/jobs/{job['id']}/match-review", json={
        "revision": difference_draft.json()["revision"], "span_id": difference["id"],
        "operation": "set_match_draft", "draft": None,
    })
    approved = client.post(f"/api/v1/jobs/{job['id']}/match-review/approve", json={
        "revision": cleared_difference.json()["revision"], "mode": "paired",
    })
    assert approved.status_code == 202
    assert jobs.get(job["id"])["phase"] == "processing"


def test_review_reanalysis_requires_explicit_discard_confirmation(client, tmp_path, monkeypatch):
    jobs = JobStore(tmp_path / "reanalyze.sqlite3")
    job = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "reanalyze-job"), "quick")
    review = {"revision": 4, "segments": [], "summary": {"proposed_segments": 0}}
    jobs.update(job["id"], state="awaiting_match_review", stage="awaiting_match_review", phase="review", review_data=review, review_revision=4)
    monkeypatch.setattr(endpoints, "store", jobs)
    monkeypatch.setattr(endpoints.worker, "notify", lambda: None)

    refused = client.post(f"/api/v1/jobs/{job['id']}/match-review/reanalyze", json={"revision": 4})
    accepted = client.post(f"/api/v1/jobs/{job['id']}/match-review/reanalyze", json={"revision": 4, "confirm_discard": True})

    assert refused.status_code == 422
    assert accepted.status_code == 202
    rebuilt = jobs.get(job["id"])
    assert rebuilt["state"] == "queued" and rebuilt["phase"] == "analysis"
    assert rebuilt["review_data"] is None
