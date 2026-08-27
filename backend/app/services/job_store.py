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
              updated_at TEXT NOT NULL,
              phase TEXT NOT NULL DEFAULT 'analysis',
              review_data TEXT,
              review_revision INTEGER NOT NULL DEFAULT 0,
              alignment_mode TEXT,
              matching_mode TEXT NOT NULL DEFAULT 'guided',
              use_audio_matching INTEGER NOT NULL DEFAULT 0,
              output_resolution TEXT NOT NULL DEFAULT 'legacy_640x480'
            )
            """
        )
        columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
        migrations = {
            "phase": "ALTER TABLE jobs ADD COLUMN phase TEXT NOT NULL DEFAULT 'analysis'",
            "review_data": "ALTER TABLE jobs ADD COLUMN review_data TEXT",
            "review_revision": "ALTER TABLE jobs ADD COLUMN review_revision INTEGER NOT NULL DEFAULT 0",
            "alignment_mode": "ALTER TABLE jobs ADD COLUMN alignment_mode TEXT",
            "matching_mode": "ALTER TABLE jobs ADD COLUMN matching_mode TEXT NOT NULL DEFAULT 'guided'",
            "use_audio_matching": "ALTER TABLE jobs ADD COLUMN use_audio_matching INTEGER NOT NULL DEFAULT 0",
            "output_resolution": "ALTER TABLE jobs ADD COLUMN output_resolution TEXT NOT NULL DEFAULT 'legacy_640x480'",
        }
        for name, sql in migrations.items():
            if name not in columns:
                db.execute(sql)
        # Review is durable and must never be bypassed after a restart.
        db.execute(
            "UPDATE jobs SET state='queued', stage='queued', message='Recovered after restart' "
            "WHERE state IN ('analyzing','training','upscaling','muxing')"
        )
        db.commit()

    def create(
        self, low_path: str, reference_path: str, job_dir: str, preset: str,
        matching_mode: str = "guided", use_audio_matching: bool = False,
        output_resolution: str = "reference",
    ) -> dict:
        if matching_mode not in {"guided", "reference_only"}:
            raise ValueError(f"Unknown matching mode: {matching_mode}")
        if output_resolution not in {
            "reference", "480p", "720p", "1080p", "2160p", "legacy_640x480",
        }:
            raise ValueError(f"Unknown output resolution: {output_resolution}")
        job_id = Path(job_dir).name
        timestamp = now()
        reference_only = matching_mode == "reference_only"
        if reference_only:
            output_resolution = "legacy_640x480"
        phase = "processing" if reference_only else "analysis"
        review = {
            "revision": 0, "approved_mode": "unpaired", "matching_mode": "reference_only",
            "segments": [], "summary": {"proposed_segments": 0, "matched_seconds": 0.0},
        } if reference_only else None
        message = "Queued for reference-only processing" if reference_only else "Queued for frame analysis"
        self.connection().execute(
            """INSERT INTO jobs (
                id,state,stage,progress,message,preset,low_path,reference_path,job_dir,metrics,
                warning,error,cancel_requested,created_at,updated_at,phase,review_data,review_revision,
                alignment_mode,matching_mode,use_audio_matching,output_resolution
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (job_id, "queued", "queued", 0.0, message, preset, low_path,
             reference_path, job_dir, None, None, None, 0, timestamp, timestamp,
             phase, json.dumps(review) if review else None, 0,
             "unpaired" if reference_only else None, matching_mode, int(use_audio_matching),
             output_resolution),
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
        value["use_audio_matching"] = bool(value.get("use_audio_matching", False))
        value["metrics"] = json.loads(value["metrics"]) if value["metrics"] else None
        value["review_data"] = json.loads(value["review_data"]) if value.get("review_data") else None
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
            "WHERE state NOT IN ('completed','failed','cancelled','awaiting_match_review') GROUP BY state"
        ).fetchall()
        queued = sum(row["count"] for row in rows if row["state"] == "queued")
        processing = sum(row["count"] for row in rows if row["state"] != "queued")
        review = self.connection().execute(
            "SELECT COUNT(*) count FROM jobs WHERE state='awaiting_match_review'"
        ).fetchone()["count"]
        return {"queued": queued, "processing": processing, "needs_review": review, "outstanding": queued + processing}

    def update(self, job_id: str, **fields) -> dict:
        allowed = {
            "state", "stage", "progress", "message", "metrics", "warning", "error",
            "cancel_requested", "phase", "review_data", "review_revision", "alignment_mode",
        }
        fields = {key: value for key, value in fields.items() if key in allowed}
        if "metrics" in fields and fields["metrics"] is not None:
            fields["metrics"] = json.dumps(fields["metrics"])
        if "review_data" in fields and fields["review_data"] is not None:
            fields["review_data"] = json.dumps(fields["review_data"])
        fields["updated_at"] = now()
        sql = ", ".join(f"{key}=?" for key in fields)
        self.connection().execute(f"UPDATE jobs SET {sql} WHERE id=?", (*fields.values(), job_id))
        self.connection().commit()
        return self.get(job_id)

    def save_review(self, job_id: str, review: dict, expected_revision: int | None = None) -> dict:
        job = self.get(job_id)
        if job["state"] != "awaiting_match_review":
            raise RuntimeError("Frame matches can only be edited while review is awaiting input")
        if expected_revision is not None and expected_revision != job["review_revision"]:
            raise RuntimeError("Frame-match review changed; reload before saving")
        revision = job["review_revision"] + 1
        review = {**review, "revision": revision}
        return self.update(job_id, review_data=review, review_revision=revision)

    def approve_review(self, job_id: str, mode: str, expected_revision: int) -> dict:
        job = self.get(job_id)
        if job["state"] != "awaiting_match_review":
            raise RuntimeError("This job is not awaiting frame-match review")
        if expected_revision != job["review_revision"]:
            raise RuntimeError("Frame-match review changed; reload before approving")
        review = {**(job["review_data"] or {}), "approved_mode": mode}
        return self.update(
            job_id, review_data=review, phase="processing", state="queued", stage="queued",
            message="Frame matches approved; queued for GPU processing", alignment_mode=mode,
            warning=None, progress=0.07,
        )

    def reanalyze_review(self, job_id: str, expected_revision: int) -> dict:
        job = self.get(job_id)
        if job["state"] != "awaiting_match_review":
            raise RuntimeError("Only a job awaiting frame-match review can be reanalyzed")
        if expected_revision != job["review_revision"]:
            raise RuntimeError("Frame-match review changed; reload before rebuilding it")
        return self.update(
            job_id, state="queued", stage="queued", phase="analysis", progress=0.0,
            message="Queued to rebuild frame alignment", review_data=None,
            review_revision=0, alignment_mode=None, warning=None, error=None,
        )

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job["state"] in TERMINAL_STATES:
            return job
        if job["state"] in {"queued", "awaiting_match_review"}:
            return self.update(job_id, state="cancelled", stage="cancelled", message="Job cancelled", progress=job["progress"])
        return self.update(job_id, cancel_requested=1, message="Cancellation requested")

    def cancel_all(self) -> tuple[int, list[dict]]:
        db = self.connection()
        timestamp = now()
        with db:
            queued = db.execute("SELECT id FROM jobs WHERE state IN ('queued','awaiting_match_review')").fetchall()
            running = db.execute(
                "SELECT id FROM jobs WHERE state NOT IN ('queued','awaiting_match_review','completed','failed','cancelled') "
                "AND cancel_requested=0"
            ).fetchall()
            db.execute(
                "UPDATE jobs SET state='cancelled', stage='cancelled', message='Job cancelled', "
                "updated_at=? WHERE state IN ('queued','awaiting_match_review')",
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
