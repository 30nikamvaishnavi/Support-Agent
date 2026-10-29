"""Deterministic safety checks that run in addition to (never instead of) the LLM."""
from __future__ import annotations

import re
from typing import Optional

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|your\s+|the\s+|any\s+)?(previous\s+|prior\s+)?(rules|instructions|guidelines)",
    r"disregard\s+(all\s+|your\s+|the\s+)?(previous\s+|prior\s+)?(rules|instructions)",
    r"\bsystem\s*:",
    r"\b(admin|developer|debug|god)\s+mode\b",
    r"(show|list|print|dump|give)\s+(me\s+)?(all|every)\s+(the\s+)?(orders|customers|tickets|records)",
    r"(reveal|show|print)\s+.{0,30}(system\s+prompt|your\s+instructions|your\s+prompt)",
    r"\bjailbreak\b",
]
ABUSE_PATTERNS = [
    r"\b(you|your)\b.{0,25}\b(stupid|idiot|moron|useless|dumb|trash)\b",
    r"\bshut\s+up\b",
    r"\b(idiot|moron)\b",
]


def check(message: str) -> Optional[str]:
    """Return 'injection', 'abusive' or None."""
    text = message.lower()
    if any(re.search(p, text) for p in INJECTION_PATTERNS):
        return "injection"
    if any(re.search(p, text) for p in ABUSE_PATTERNS):
        return "abusive"
    return None


_ORDER_ID = re.compile(r"\bORD\d{4,}\b", re.I)
_TICKET = re.compile(r"#\d+")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_BIG_NUMBER = re.compile(r"\b\d[\d,]{2,}(?:\.\d+)?\b")  # 3+ digits: amounts, ids


def _norm(token: str) -> str:
    return token.replace(",", "").lower()


def grounding_ok(response: str, allowed_text: str) -> tuple[bool, list[str]]:
    """Every order ID, ticket number, date and 3+ digit number in the reply must appear in the
    facts the agent actually had (tool result + customer message). Catches invented details."""
    allowed = _norm(allowed_text)
    allowed_numbers = {_norm(t) for t in _BIG_NUMBER.findall(allowed_text)}
    unsupported = []
    for pattern in (_ORDER_ID, _TICKET, _DATE):
        for token in pattern.findall(response):
            if _norm(token) not in allowed:
                unsupported.append(token)
    for token in _BIG_NUMBER.findall(response):
        if _norm(token) not in allowed_numbers and _norm(token) not in allowed:
            unsupported.append(token)
    return (not unsupported, unsupported)
