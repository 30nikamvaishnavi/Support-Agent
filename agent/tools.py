"""Deterministic tools. The LLM never touches the database: it only decides WHICH tool to use."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

import config
from . import db

PUBLIC_ORDER_FIELDS = ("order_id", "items", "status", "order_date", "delivery_date",
                       "delivery_note", "amount")


def normalize_name(name: str) -> frozenset:
    """Lower-case word set, so 'patil rohan' matches 'Rohan Patil' but 'Rohan' alone does not."""
    return frozenset(re.findall(r"[a-z]+", (name or "").lower()))


def names_match(provided: Optional[str], actual: str) -> bool:
    tokens = normalize_name(provided or "")
    return bool(tokens) and tokens == normalize_name(actual)


def _public(order: dict) -> dict:
    return {k: order[k] for k in PUBLIC_ORDER_FIELDS}


def order_lookup(order_id: Optional[str], customer_name: Optional[str],
                 db_path: Optional[Path] = None) -> dict:
    """Return order details ONLY if the order exists AND the name matches.

    A missing order and a wrong name give the same 'no_match' status, so nobody can use
    this tool to find out which order IDs exist. The internal reason is kept for the logs.
    """
    if not order_id or not customer_name:
        return {"status": "missing_info"}
    order = db.get_order(order_id.upper(), db_path)
    if order is None:
        return {"status": "no_match", "reason": "order_not_found"}
    if not names_match(customer_name, order["customer_name"]):
        return {"status": "no_match", "reason": "name_mismatch"}
    return {"status": "order_found", "order": _public(order)}


def check_refund_eligibility(order: dict, today: date) -> dict:
    """Policy check: delivered, and within the return window."""
    if order["status"] == "Cancelled":
        return {"eligible": False, "reason": "cancelled"}
    if order["status"] != "Delivered":
        return {"eligible": False, "reason": "not_delivered"}
    delivered = date.fromisoformat(order["delivery_date"])
    days = (today - delivered).days
    if days > config.RETURN_WINDOW_DAYS:
        return {"eligible": False, "reason": "outside_window", "days_since_delivery": days}
    return {"eligible": True, "reason": "ok", "days_since_delivery": days}


def create_ticket(summary: str, priority: str, category: str, order_id: Optional[str] = None,
                  thread_id: Optional[str] = None, db_path: Optional[Path] = None) -> dict:
    ticket_id = db.insert_ticket(category, priority, (summary or "")[:300], order_id, thread_id, db_path)
    return {
        "status": "ticket_created", "ticket_id": ticket_id, "category": category,
        "priority": priority, "order_id": order_id,
        "contact_within_hours": config.CONTACT_WITHIN_HOURS,
    }


def human_handoff(reason: str, summary: str = "", order_id: Optional[str] = None,
                  thread_id: Optional[str] = None, db_path: Optional[Path] = None) -> dict:
    """Escalate: creates an urgent ticket that a human agent will pick up."""
    ticket_id = db.insert_ticket("handoff", "urgent", f"[{reason}] {summary}".strip()[:300],
                                 order_id, thread_id, db_path)
    return {
        "status": "handoff", "ticket_id": ticket_id, "reason": reason, "order_id": order_id,
        "contact_within_hours": config.CONTACT_WITHIN_HOURS,
    }
