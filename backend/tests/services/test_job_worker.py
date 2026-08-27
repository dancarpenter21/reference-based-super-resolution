import time

from app.services import job_worker as job_worker_module
from app.services.job_store import JobStore
from app.services.job_worker import JobWorker


class EmptyJobs:
    def next_queued(self):
        return None


class BrokenJobs:
    def next_queued(self):
        raise RuntimeError("database unavailable")


def test_worker_reports_idle_busy_and_stopped():
    worker = JobWorker(EmptyJobs())
    assert worker.status()["state"] == "stopped"

    worker.start()
    assert worker.status()["state"] == "idle"
    with worker._status_lock:
        worker.current_job_id = "job-123"
    assert worker.status()["state"] == "busy"
    assert worker.status()["current_job_id"] == "job-123"

    worker.stop()
    assert worker.status()["state"] == "stopped"


def test_worker_reports_fatal_loop_error():
    worker = JobWorker(BrokenJobs())
    worker.start()
    deadline = time.monotonic() + 2
    while worker.thread.is_alive() and time.monotonic() < deadline:
        worker.thread.join(timeout=0.01)

    status = worker.status()
    assert status["state"] == "stopped"
    assert status["fatal_error"] == "RuntimeError: database unavailable"


def test_worker_pauses_for_review_then_dispatches_approved_processing(tmp_path, monkeypatch):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    job = jobs.create(
        "low.mp4", "reference.mp4", str(tmp_path / "job"), "quick",
        use_audio_matching=True, output_resolution="1080p",
    )
    review = {"revision": 1, "segments": [], "summary": {"proposed_segments": 0}}
    analysis_calls = []

    def analyze(*args, **kwargs):
        analysis_calls.append((args, kwargs))
        return review

    monkeypatch.setattr(job_worker_module, "analyze_pipeline", analyze)
    processed = []
    monkeypatch.setattr(job_worker_module, "process_pipeline", lambda *args, **kwargs: processed.append((args, kwargs)))
    worker = JobWorker(jobs)

    worker._run(job)
    waiting = jobs.get(job["id"])

    assert waiting["state"] == "awaiting_match_review"
    assert waiting["phase"] == "review"
    assert analysis_calls[0][1]["use_audio_matching"] is True
    assert analysis_calls[0][1]["output_resolution"] == "1080p"
    approved = jobs.approve_review(job["id"], "unpaired", 1)
    worker._run(approved)
    assert processed
    assert processed[0][1]["output_resolution"] == "1080p"
    assert jobs.get(job["id"])["state"] == "completed"


def test_reference_only_worker_never_runs_match_analysis(tmp_path, monkeypatch):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    job = jobs.create(
        "low.mp4", "reference.mp4", str(tmp_path / "job"), "quick",
        matching_mode="reference_only",
    )
    monkeypatch.setattr(
        job_worker_module, "analyze_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analysis should be skipped")),
    )
    processed = []
    monkeypatch.setattr(job_worker_module, "process_pipeline", lambda *args, **kwargs: processed.append(args))

    JobWorker(jobs)._run(job)

    assert processed
    assert jobs.get(job["id"])["state"] == "completed"
