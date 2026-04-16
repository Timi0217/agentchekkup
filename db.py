"""SQLite persistence for AgentChekkup evaluation results.

Stores every evaluation as a JSON blob in a single table so results
survive restarts and users can browse history.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "results.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), timeout=10)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn


def init_db():
    """Create the evaluations table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            eval_id       TEXT PRIMARY KEY,
            agent_url     TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'running',
            overall_score INTEGER,
            badge         TEXT,
            total_passed  INTEGER,
            total_failed  INTEGER,
            total_tests   INTEGER,
            duration_secs REAL,
            started_at    REAL NOT NULL,
            data          TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_evaluations_started
        ON evaluations (started_at DESC)
    """)
    conn.commit()


def save_evaluation(evaluation: dict):
    """Insert or replace a full evaluation dict into SQLite."""
    conn = _get_conn()
    sc = evaluation.get("scorecard", {})
    conn.execute(
        """
        INSERT OR REPLACE INTO evaluations
            (eval_id, agent_url, status, overall_score, badge,
             total_passed, total_failed, total_tests,
             duration_secs, started_at, data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evaluation["eval_id"],
            evaluation.get("agent_url", ""),
            evaluation.get("status", "running"),
            sc.get("overall_score"),
            sc.get("badge"),
            sc.get("total_passed"),
            sc.get("total_failed"),
            sc.get("total_tests"),
            evaluation.get("duration_seconds"),
            evaluation.get("started_at", time.time()),
            json.dumps(evaluation),
        ),
    )
    conn.commit()


def load_evaluation(eval_id: str) -> Optional[dict]:
    """Load a single evaluation by ID. Returns the full dict or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT data FROM evaluations WHERE eval_id = ?", (eval_id,)
    ).fetchone()
    if row:
        return json.loads(row["data"])
    return None


def list_evaluations(limit: int = 50, offset: int = 0) -> list[dict]:
    """List recent evaluations (summary only, not the full data blob)."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT eval_id, agent_url, status, overall_score, badge,
               total_passed, total_failed, total_tests,
               duration_secs, started_at, data
        FROM evaluations
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()

    results = []
    for r in rows:
        entry = {
            "eval_id": r["eval_id"],
            "agent_url": r["agent_url"],
            "status": r["status"],
            "overall_score": r["overall_score"],
            "badge": r["badge"],
            "total_passed": r["total_passed"],
            "total_failed": r["total_failed"],
            "total_tests": r["total_tests"],
            "duration_seconds": r["duration_secs"],
            "started_at": r["started_at"],
        }
        # Check if retest data exists — if so, surface the after-fix score
        try:
            data = json.loads(r["data"])
            retest = data.get("retest")
            if retest:
                summary = retest.get("summary", {})
                entry["has_retest"] = True
                entry["before_score"] = summary.get("before_score", entry["overall_score"])
                entry["after_score"] = summary.get("after_score")
                entry["tests_fixed"] = summary.get("tests_fixed", 0)
        except (json.JSONDecodeError, TypeError):
            pass
        results.append(entry)
    return results


def update_evaluation_fixes(eval_id: str, remediation: dict):
    """Update just the remediation/deployed_fixes portion of an evaluation."""
    existing = load_evaluation(eval_id)
    if not existing:
        return
    existing["remediation"] = remediation
    save_evaluation(existing)
