from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ml_engine.config import DATA_ROOT, JOBS_ROOT

DB_PATH = DATA_ROOT / "jobs.sqlite3"
TERMINAL_STATES = {"completed", "failed", "cancelled"}


def now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.initialize()

    def connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection = connection
        return connection

    def initialize(self) -> None:
        db = self.connection()
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              state TEXT NOT NULL,
              stage TEXT NOT NULL,
              progress REAL NOT NULL,
              message TEXT NOT NULL,
              preset TEXT NOT NULL,
              low_path TEXT NOT NULL,
              reference_path TEXT NOT NULL,
              job_dir TEXT NOT NULL,
              metrics TEXT,
              warning TEXT,
              error TEXT,
              cancel_requested INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        # Interrupted work is safe to restart. Training writes only versioned/best checkpoints.
        db.execute(
            "UPDATE jobs SET state='queued', stage='queued', message='Recovered after restart' "
            "WHERE state NOT IN ('completed','failed','cancelled')"
        )
        db.commit()

    def create(self, low_path: str, reference_path: str, job_dir: str, preset: str) -> dict:
        job_id = Path(job_dir).name
        timestamp = now()
        self.connection().execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, "queued", "queued", 0.0, "Queued for processing", preset, low_path,
             reference_path, job_dir, None, None, None, 0, timestamp, timestamp),
        )
        self.connection().commit()
        return self.get(job_id)

    def new_paths(self, low_suffix: str, reference_suffix: str) -> tuple[str, Path, Path, Path]:
        job_id = str(uuid.uuid4())
        job_dir = JOBS_ROOT / job_id
        upload_dir = job_dir / "uploads"
        upload_dir.mkdir(parents=True)
        return job_id, job_dir, upload_dir / f"low{low_suffix}", upload_dir / f"reference{reference_suffix}"

    def get(self, job_id: str) -> dict:
        row = self.connection().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._deserialize(row)

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> dict:
        value = dict(row)
        value["cancel_requested"] = bool(value["cancel_requested"])
        value["metrics"] = json.loads(value["metrics"]) if value["metrics"] else None
        return value

    def list_all(self) -> list[dict]:
        rows = self.connection().execute(
            """
            SELECT * FROM jobs
            ORDER BY
              CASE WHEN state IN ('completed','failed','cancelled') THEN 1 ELSE 0 END,
              CASE WHEN state NOT IN ('completed','failed','cancelled') THEN created_at END ASC,
              CASE WHEN state IN ('completed','failed','cancelled') THEN created_at END DESC,
              id ASC
            """
        ).fetchall()
        return [self._deserialize(row) for row in rows]

    def next_queued(self) -> dict | None:
        row = self.connection().execute(
            "SELECT id FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        return self.get(row["id"]) if row else None

    def active_counts(self) -> dict[str, int]:
        rows = self.connection().execute(
            "SELECT state, COUNT(*) count FROM jobs "
            "WHERE state NOT IN ('completed','failed','cancelled') GROUP BY state"
        ).fetchall()
        queued = sum(row["count"] for row in rows if row["state"] == "queued")
        processing = sum(row["count"] for row in rows if row["state"] != "queued")
        return {"queued": queued, "processing": processing, "outstanding": queued + processing}

    def update(self, job_id: str, **fields) -> dict:
        allowed = {"state", "stage", "progress", "message", "metrics", "warning", "error", "cancel_requested"}
        fields = {key: value for key, value in fields.items() if key in allowed}
        if "metrics" in fields and fields["metrics"] is not None:
            fields["metrics"] = json.dumps(fields["metrics"])
        fields["updated_at"] = now()
        sql = ", ".join(f"{key}=?" for key in fields)
        self.connection().execute(f"UPDATE jobs SET {sql} WHERE id=?", (*fields.values(), job_id))
        self.connection().commit()
        return self.get(job_id)

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job["state"] in TERMINAL_STATES:
            return job
        if job["state"] == "queued":
            return self.update(job_id, state="cancelled", stage="cancelled", message="Job cancelled", progress=job["progress"])
        return self.update(job_id, cancel_requested=1, message="Cancellation requested")

    def cancel_all(self) -> tuple[int, list[dict]]:
        db = self.connection()
        timestamp = now()
        with db:
            queued = db.execute("SELECT id FROM jobs WHERE state='queued'").fetchall()
            running = db.execute(
                "SELECT id FROM jobs WHERE state NOT IN ('queued','completed','failed','cancelled') "
                "AND cancel_requested=0"
            ).fetchall()
            db.execute(
                "UPDATE jobs SET state='cancelled', stage='cancelled', message='Job cancelled', "
                "updated_at=? WHERE state='queued'",
                (timestamp,),
            )
            db.execute(
                "UPDATE jobs SET cancel_requested=1, message='Cancellation requested', updated_at=? "
                "WHERE state NOT IN ('queued','completed','failed','cancelled') AND cancel_requested=0",
                (timestamp,),
            )
        return len(queued) + len(running), self.list_all()

    def delete(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job["state"] not in TERMINAL_STATES:
            raise RuntimeError("Cancel the active job before deleting it")
        self.connection().execute("DELETE FROM jobs WHERE id=?", (job_id,))
        self.connection().commit()
        return job


store = JobStore()
