"""
P09 · Score History Database
SQLite-based score tracking for regression detection.
Zero infra — single file, works everywhere.
"""

import os
import sqlite3
from datetime import datetime, timezone


DB_PATH = os.environ.get("EVAL_DB_PATH", "data/processed/eval_history.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH):
    """Create tables if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL UNIQUE,
            timestamp   TEXT NOT NULL,
            model_name  TEXT,
            dataset     TEXT,
            total       INTEGER,
            passed      INTEGER,
            pass_rate   REAL,
            avg_composite REAL,
            avg_rouge1  REAL,
            avg_rougeL  REAL,
            avg_judge   REAL,
            avg_rubric  REAL,
            git_sha     TEXT,
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            test_case_id    TEXT NOT NULL,
            question        TEXT,
            category        TEXT,
            rouge1          REAL,
            rouge2          REAL,
            rougeL          REAL,
            llm_judge_score REAL,
            llm_judge_reason TEXT,
            rubric_score    REAL,
            composite_score REAL,
            passed          INTEGER,
            latency_ms      INTEGER,
            timestamp       TEXT,
            FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON eval_runs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_results_run ON eval_results(run_id);
    """)
    conn.commit()
    conn.close()


def save_run(run_id: str, summary: dict, results: list, db_path: str = DB_PATH):
    """Save a complete eval run to the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        conn.execute("""
            INSERT OR REPLACE INTO eval_runs
            (run_id, timestamp, model_name, dataset, total, passed, pass_rate,
             avg_composite, avg_rouge1, avg_rougeL, avg_judge, avg_rubric, git_sha, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            timestamp,
            summary.get("model_name", "unknown"),
            summary.get("dataset", "unknown"),
            summary.get("total", 0),
            summary.get("passed", 0),
            summary.get("pass_rate", 0),
            summary.get("avg_composite", 0),
            summary.get("avg_rouge1", 0),
            summary.get("avg_rougeL", 0),
            summary.get("avg_llm_judge", 0),
            summary.get("avg_rubric", 0),
            summary.get("git_sha", ""),
            summary.get("notes", ""),
        ))

        for r in results:
            conn.execute("""
                INSERT INTO eval_results
                (run_id, test_case_id, question, category, rouge1, rouge2, rougeL,
                 llm_judge_score, llm_judge_reason, rubric_score, composite_score,
                 passed, latency_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                r.get("test_case_id", ""),
                r.get("question", "")[:200],
                r.get("category", "general"),
                r.get("rouge1", 0),
                r.get("rouge2", 0),
                r.get("rougeL", 0),
                r.get("llm_judge_score", 0),
                r.get("llm_judge_reasoning", "")[:200],
                r.get("rubric_score", 0),
                r.get("composite_score", 0),
                1 if r.get("passed") else 0,
                r.get("latency_ms", 0),
                r.get("timestamp", timestamp),
            ))

        conn.commit()
    finally:
        conn.close()


def get_recent_runs(n: int = 10, db_path: str = DB_PATH) -> list[dict]:
    """Get the N most recent eval runs."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT * FROM eval_runs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (n,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def check_regression(
    current_score: float,
    db_path: str = DB_PATH,
    threshold_drop_pct: float = 5.0,
) -> dict:
    """
    Compare current score against the last run.
    Returns regression info — used as CI gate.
    """
    runs = get_recent_runs(2, db_path)

    if len(runs) < 2:
        return {
            "has_regression": False,
            "reason": "Not enough history to compare",
            "current": current_score,
            "previous": None,
        }

    previous_score = runs[1]["avg_composite"]  # second most recent
    drop_pct = ((previous_score - current_score) / max(previous_score, 0.01)) * 100

    return {
        "has_regression": drop_pct > threshold_drop_pct,
        "current": round(current_score, 2),
        "previous": round(previous_score, 2),
        "drop_pct": round(drop_pct, 1),
        "threshold_pct": threshold_drop_pct,
        "reason": (
            f"Score dropped {drop_pct:.1f}% (threshold: {threshold_drop_pct}%)"
            if drop_pct > threshold_drop_pct
            else f"Score OK — dropped {drop_pct:.1f}% (within {threshold_drop_pct}% threshold)"
        ),
    }


def get_score_trend(db_path: str = DB_PATH, n: int = 20) -> list[dict]:
    """Get score trend for the last N runs — used for dashboard."""
    runs = get_recent_runs(n, db_path)
    return [
        {
            "run_id": r["run_id"],
            "timestamp": r["timestamp"][:10],
            "avg_composite": r["avg_composite"],
            "pass_rate": r["pass_rate"],
        }
        for r in reversed(runs)
    ]
