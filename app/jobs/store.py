from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.job import JobOut
from app.settings import settings

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"running", "failed", "cancelled"},
    "running": {"succeeded", "failed", "cancelled"},
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
}


class IllegalJobTransition(ValueError):
    """queued→running→succeeded/failed; terminal states cannot move backward."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@contextmanager
def connect():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.jobs_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                modality TEXT NOT NULL,
                style_id TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                kind TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spend (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                amount REAL NOT NULL,
                modality TEXT NOT NULL,
                day TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def register_asset(path: Path, kind: str = "upload", asset_id: str | None = None) -> str:
    init_db()
    asset_id = asset_id or new_id("a")
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets(asset_id, path, kind, created_at) VALUES (?,?,?,?)",
            (asset_id, str(path), kind, _now()),
        )
    return asset_id


def asset_path(asset_id: str) -> Path:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT path FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
    if not row:
        raise KeyError(asset_id)
    return Path(row["path"])


def save_job(job: JobOut) -> None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job.job_id,)).fetchone()
        if row:
            old = row["status"]
            if job.status != old and job.status not in ALLOWED_TRANSITIONS.get(old, set()):
                raise IllegalJobTransition(f"{job.job_id}: {old} -> {job.status}")
        conn.execute(
            """
            INSERT INTO jobs(job_id, status, modality, style_id, payload, created_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET
              status=excluded.status,
              payload=excluded.payload
            """,
            (
                job.job_id,
                job.status,
                job.modality,
                job.trace.style_id,
                job.model_dump_json(),
                _now(),
            ),
        )


def get_job(job_id: str) -> JobOut | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT payload FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row:
        return None
    return JobOut.model_validate_json(row["payload"])


def dump_output(bgr_bytes: bytes, style_id: str, kind: str | None = None) -> str:
    init_db()
    asset_id = new_id("a")
    path = settings.assets_dir / f"{asset_id}.jpg"
    path.write_bytes(bgr_bytes)
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets(asset_id, path, kind, created_at) VALUES (?,?,?,?)",
            (asset_id, str(path), kind or f"look:{style_id}", _now()),
        )
    return asset_id
