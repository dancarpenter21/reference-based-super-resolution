from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from ml_engine.pipeline import Cancelled, analyze_pipeline, process_pipeline

from .job_store import JobStore, store

logger = logging.getLogger(__name__)


class JobWorker:
    def __init__(self, jobs: JobStore = store):
        self.jobs = jobs
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at: str | None = None
        self.current_job_id: str | None = None
        self.fatal_error: str | None = None
        self._status_lock = threading.Lock()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        with self._status_lock:
            self.started_at = datetime.now(UTC).isoformat()
            self.current_job_id = None
            self.fatal_error = None
        self.thread = threading.Thread(target=self._loop, name="refsr-gpu-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def notify(self) -> None:
        self.wake_event.set()

    def status(self) -> dict:
        alive = bool(self.thread and self.thread.is_alive())
        with self._status_lock:
            current_job_id = self.current_job_id
            return {
                "state": "busy" if alive and current_job_id else "idle" if alive else "stopped",
                "current_job_id": current_job_id,
                "started_at": self.started_at,
                "fatal_error": self.fatal_error,
            }

    def _loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                job = self.jobs.next_queued()
                if job is None:
                    self.wake_event.wait(2)
                    self.wake_event.clear()
                    continue
                with self._status_lock:
                    self.current_job_id = job["id"]
                try:
                    self._run(job)
                finally:
                    with self._status_lock:
                        self.current_job_id = None
        except Exception as error:
            logger.exception("GPU worker stopped unexpectedly")
            with self._status_lock:
                self.fatal_error = f"{type(error).__name__}: {error}"

    def _run(self, job: dict) -> None:
        job_id = job["id"]
        if job["cancel_requested"]:
            self.jobs.update(job_id, state="cancelled", stage="cancelled", message="Job cancelled")
            return
        phase = job.get("phase", "analysis")
        first_state = "analyzing" if phase == "analysis" else "training"
        processing_message = (
            "Starting reference-only processing"
            if job.get("matching_mode") == "reference_only"
            else "Starting approved processing"
        )
        self.jobs.update(
            job_id, state=first_state, stage=first_state,
            progress=0.01 if phase == "analysis" else max(0.07, job["progress"]),
            message="Starting analysis" if phase == "analysis" else processing_message,
        )

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
            if phase == "analysis":
                review = analyze_pipeline(
                    job["low_path"], job["reference_path"], job["job_dir"], update, cancelled,
                    use_audio_matching=job.get("use_audio_matching", False),
                )
                self.jobs.update(
                    job_id, state="awaiting_match_review", stage="awaiting_match_review", progress=0.07,
                    message="Review proposed frame matches", review_data=review,
                    review_revision=int(review.get("revision", 1)), phase="review",
                    warning=review.get("summary", {}).get("warning"),
                )
                return
            current = self.jobs.get(job_id)
            process_pipeline(
                job["low_path"], job["reference_path"], job["job_dir"], job["preset"],
                current["review_data"] or {}, update, cancelled,
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
