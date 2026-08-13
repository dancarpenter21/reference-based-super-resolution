import time

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
