"""Daily / project spend ledger. Look jobs never consume budget."""

from __future__ import annotations

from datetime import datetime, timezone

from app.jobs.store import connect, init_db
from app.settings import settings


def _day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def init_spend() -> None:
    init_db()
    with connect() as conn:
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


def spent_on(day: str | None = None) -> float:
    init_spend()
    day = day or _day()
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS t FROM spend WHERE day=?",
            (day,),
        ).fetchone()
    return float(row["t"] if row else 0.0)


def spent_project() -> float:
    init_spend()
    with connect() as conn:
        row = conn.execute("SELECT COALESCE(SUM(amount), 0) AS t FROM spend").fetchone()
    return float(row["t"] if row else 0.0)


def snapshot() -> dict:
    return {
        "daily_spent_cny": round(spent_on(), 4),
        "daily_budget_cny": settings.daily_budget_cny,
        "project_spent_cny": round(spent_project(), 4),
        "project_budget_cny": settings.project_budget_cny,
    }


class BudgetExceeded(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.code = "BUDGET_EXCEEDED"


def assert_can_spend(estimated_cny: float, job_max_cny: float) -> None:
    """Raise BudgetExceeded before any paid provider HTTP. Cost-0 Look should not call this."""
    if estimated_cny <= 0:
        return
    if estimated_cny > job_max_cny:
        raise BudgetExceeded(
            f"estimated {estimated_cny} CNY exceeds job budget_cny_max {job_max_cny}"
        )
    daily = spent_on()
    if daily + estimated_cny > settings.daily_budget_cny:
        raise BudgetExceeded(
            f"daily spend {daily:.4f}+{estimated_cny} exceeds DAILY_BUDGET_CNY={settings.daily_budget_cny}"
        )
    proj = spent_project()
    if proj + estimated_cny > settings.project_budget_cny:
        raise BudgetExceeded(
            f"project spend {proj:.4f}+{estimated_cny} exceeds PROJECT_BUDGET_CNY={settings.project_budget_cny}"
        )


def record_spend(job_id: str, amount: float, modality: str) -> None:
    if amount <= 0:
        return
    init_spend()
    from app.jobs.store import _now

    with connect() as conn:
        conn.execute(
            "INSERT INTO spend(job_id, amount, modality, day, created_at) VALUES (?,?,?,?,?)",
            (job_id, float(amount), modality, _day(), _now()),
        )
