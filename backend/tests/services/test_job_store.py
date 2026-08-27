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


def test_audio_matching_preference_is_persisted(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    job = jobs.create(
        "low.mp4", "reference.mp4", str(tmp_path / "audio-job"), "quick",
        use_audio_matching=True,
    )

    assert job["use_audio_matching"] is True
    assert JobStore(jobs.path).get(job["id"])["use_audio_matching"] is True


def test_output_resolution_is_persisted(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    job = jobs.create(
        "low.mp4", "reference.mp4", str(tmp_path / "resolution-job"), "quick",
        output_resolution="1080p",
    )
    assert job["output_resolution"] == "1080p"
    assert JobStore(jobs.path).get(job["id"])["output_resolution"] == "1080p"
    with pytest.raises(ValueError, match="Unknown output resolution"):
        jobs.create("low.mp4", "reference.mp4", str(tmp_path / "bad-size"), "quick", output_resolution="8k")


def test_active_job_cannot_be_deleted(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    job = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "job-2"), "balanced")
    with pytest.raises(RuntimeError):
        jobs.delete(job["id"])
    jobs.update(job["id"], state="failed")
    assert jobs.delete(job["id"])["id"] == job["id"]


def test_jobs_persist_and_list_active_before_newest_history(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    jobs = JobStore(path)
    oldest_active = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "active-1"), "quick")
    newest_active = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "active-2"), "balanced")
    old_history = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "history-1"), "quick")
    jobs.update(old_history["id"], state="completed", stage="completed")
    new_history = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "history-2"), "quality")
    jobs.update(new_history["id"], state="failed", stage="failed")

    restored = JobStore(path).list_all()

    assert [job["id"] for job in restored[:2]] == [oldest_active["id"], newest_active["id"]]
    assert [job["id"] for job in restored[2:]] == [new_history["id"], old_history["id"]]


def test_cancel_all_cancels_queue_and_requests_running_job(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    queued = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "queued"), "quick")
    running = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "running"), "balanced")
    completed = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "completed"), "quality")
    jobs.update(running["id"], state="training", stage="training")
    jobs.update(completed["id"], state="completed", stage="completed")

    affected, _ = jobs.cancel_all()

    assert affected == 2
    assert jobs.get(queued["id"])["state"] == "cancelled"
    assert jobs.get(running["id"])["cancel_requested"] is True
    assert jobs.get(running["id"])["message"] == "Cancellation requested"
    assert jobs.get(completed["id"])["state"] == "completed"
    assert jobs.cancel_all()[0] == 0


def test_active_counts_separate_queued_and_processing(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    queued = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "queued"), "quick")
    running = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "running"), "balanced")
    completed = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "completed"), "quality")
    jobs.update(running["id"], state="analyzing", stage="analyzing")
    jobs.update(completed["id"], state="completed", stage="completed")

    assert jobs.active_counts() == {"queued": 1, "processing": 1, "needs_review": 0, "outstanding": 2}
    assert jobs.get(queued["id"])["message"] == "Queued for frame analysis"


def test_review_is_persisted_and_does_not_consume_worker_capacity(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    jobs = JobStore(path)
    job = jobs.create("low.mp4", "reference.mp4", str(tmp_path / "review"), "quick")
    review = {"revision": 1, "segments": [], "summary": {"proposed_segments": 0}}
    jobs.update(
        job["id"], state="awaiting_match_review", stage="awaiting_match_review",
        phase="review", review_data=review, review_revision=1,
    )

    restored = JobStore(path).get(job["id"])

    assert restored["state"] == "awaiting_match_review"
    assert restored["review_data"] == review
    assert jobs.active_counts() == {"queued": 0, "processing": 0, "needs_review": 1, "outstanding": 0}
    assert jobs.cancel(job["id"])["state"] == "cancelled"


def test_reference_only_job_skips_analysis_phase(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    job = jobs.create(
        "low.mp4", "reference.mp4", str(tmp_path / "reference-only"), "quick",
        matching_mode="reference_only",
    )

    assert job["matching_mode"] == "reference_only"
    assert job["phase"] == "processing"
    assert job["alignment_mode"] == "unpaired"
    assert job["review_data"]["approved_mode"] == "unpaired"
    assert job["message"] == "Queued for reference-only processing"
