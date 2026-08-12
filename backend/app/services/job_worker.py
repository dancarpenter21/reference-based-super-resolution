from __future__ import annotations

import logging
import threading

from ml_engine.pipeline import Cancelled, run_pipeline

from .job_store import JobStore, store

logger = logging.getLogger(__name__)


class JobWorker:
    def __init__(self, jobs: JobStore = store):
        self.jobs = jobs
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="refsr-gpu-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def notify(self) -> None:
        self.wake_event.set()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            job = self.jobs.next_queued()
            if job is None:
                self.wake_event.wait(2)
                self.wake_event.clear()
                continue
            self._run(job)

    def _run(self, job: dict) -> None:
        job_id = job["id"]
        if job["cancel_requested"]:
            self.jobs.update(job_id, state="cancelled", stage="cancelled", message="Job cancelled")
            return
        self.jobs.update(job_id, state="analyzing", stage="analyzing", progress=0.01, message="Starting analysis")

        def cancelled() -> bool:
            return self.stop_event.is_set() or self.jobs.get(job_id)["cancel_requested"]

        def update(stage: str, progress: float, message: str, metrics: dict | None) -> None:
            fields = {"state": stage, "stage": stage, "progress": progress, "message": message}
            if metrics:
                fields["metrics"] = metrics
                warning = metrics.get("warning")
                if warning:
                    fields["warning"] = warning
            self.jobs.update(job_id, **fields)

        try:
            run_pipeline(
                job["low_path"], job["reference_path"], job["job_dir"], job["preset"], update, cancelled
            )
            self.jobs.update(
                job_id, state="completed", stage="completed", progress=1.0,
                message="Upscaled video is ready",
            )
        except (Cancelled, InterruptedError):
            if self.stop_event.is_set() and not self.jobs.get(job_id)["cancel_requested"]:
                self.jobs.update(
                    job_id, state="queued", stage="queued", message="Paused for backend shutdown; will resume"
                )
            else:
                self.jobs.update(job_id, state="cancelled", stage="cancelled", message="Job cancelled")
        except Exception as error:
            logger.exception("Job %s failed", job_id)
            self.jobs.update(
                job_id, state="failed", stage="failed", message="Job failed",
                error=f"{type(error).__name__}: {error}",
            )


worker = JobWorker()
