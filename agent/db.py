"""SQLite storage: orders (seeded, fictional), tickets and conversation logs."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, customer_name TEXT, items TEXT, status TEXT,
    order_date TEXT, delivery_date TEXT, delivery_note TEXT, amount REAL
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, thread_id TEXT, order_id TEXT,
    category TEXT, priority TEXT, summary TEXT, status TEXT DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, thread_id TEXT, user_message TEXT,
    intent TEXT, tool TEXT, tool_status TEXT, response TEXT, response_source TEXT,
    latency_ms REAL, llm_calls INTEGER
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None, reset: bool = False) -> Path:
    """Create tables and seed the 30 fictional orders if the orders table is empty."""
    path = Path(db_path or config.DB_PATH)
    if reset and path.exists():
        path.unlink()
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        if count == 0:
            seed = json.loads((config.DATA_DIR / "orders_seed.json").read_text(encoding="utf-8"))
            conn.executemany(
                "INSERT INTO orders VALUES (:order_id,:customer_name,:items,:status,"
                ":order_date,:delivery_date,:delivery_note,:amount)",
                seed,
            )
        conn.commit()
    finally:
        conn.close()
    return path


def get_order(order_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def insert_ticket(category: str, priority: str, summary: str, order_id: Optional[str],
                  thread_id: Optional[str], db_path: Optional[Path] = None) -> int:
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO tickets (created_at, thread_id, order_id, category, priority, summary)"
            " VALUES (?,?,?,?,?,?)",
            (_now(), thread_id, order_id, category, priority, summary),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def log_turn(thread_id, user_message, intent, tool, tool_status, response, response_source,
             latency_ms, llm_calls, db_path: Optional[Path] = None) -> None:
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO logs (ts, thread_id, user_message, intent, tool, tool_status, response,"
            " response_source, latency_ms, llm_calls) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (_now(), thread_id, user_message, intent, tool, tool_status, response,
             response_source, latency_ms, llm_calls),
        )
        conn.commit()
    finally:
        conn.close()


def list_tickets(db_path: Optional[Path] = None) -> list[dict]:
    conn = connect(db_path)
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM tickets ORDER BY id DESC")]
    finally:
        conn.close()


def list_logs(db_path: Optional[Path] = None, limit: int = 500) -> list[dict]:
    conn = connect(db_path)
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,))]
    finally:
        conn.close()
