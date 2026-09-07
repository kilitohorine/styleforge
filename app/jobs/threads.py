from __future__ import annotations

import json
from typing import Any

from app.jobs.store import connect, init_db, _now


def init_threads() -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                style_id TEXT,
                asset_id TEXT,
                params TEXT NOT NULL,
                last_job_id TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )


def load_thread(thread_id: str) -> dict[str, Any] | None:
    init_threads()
    with connect() as conn:
        row = conn.execute(
            "SELECT thread_id, style_id, asset_id, params, last_job_id FROM threads WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "thread_id": row["thread_id"],
        "style_id": row["style_id"],
        "asset_id": row["asset_id"],
        "params": json.loads(row["params"] or "{}"),
        "last_job_id": row["last_job_id"],
    }


def save_thread(
    thread_id: str,
    *,
    style_id: str | None,
    asset_id: str | None,
    params: dict,
    last_job_id: str | None,
) -> None:
    init_threads()
    payload = json.dumps(params, ensure_ascii=False)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO threads(thread_id, style_id, asset_id, params, last_job_id, updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(thread_id) DO UPDATE SET
              style_id=excluded.style_id,
              asset_id=excluded.asset_id,
              params=excluded.params,
              last_job_id=excluded.last_job_id,
              updated_at=excluded.updated_at
            """,
            (thread_id, style_id, asset_id, payload, last_job_id, _now()),
        )
