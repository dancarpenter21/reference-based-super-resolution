from io import BytesIO

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
