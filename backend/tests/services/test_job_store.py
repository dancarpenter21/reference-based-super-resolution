import pytest
from app.services.job_store import JobStore


def test_job_lifecycle(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    job_dir = tmp_path / "job-1"
    job = jobs.create("low.mp4", "reference.mp4", str(job_dir), "quick")
    assert job["state"] == "queued"
    assert jobs.next_queued()["id"] == job["id"]
    updated = jobs.update(job["id"], state="training", stage="training", progress=0.4, metrics={"psnr": 28.1})
    assert updated["metrics"] == {"psnr": 28.1}
    assert jobs.cancel(job["id"])["cancel_requested"] is True


def test_active_job_cannot_be_deleted(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    job = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "job-2"), "balanced")
    with pytest.raises(RuntimeError):
        jobs.delete(job["id"])
    jobs.update(job["id"], state="failed")
    assert jobs.delete(job["id"])["id"] == job["id"]
