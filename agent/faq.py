"""FAQ retrieval over data/policies.md using BM25 (pure Python, free, no model download).

Swap this module for an embedding-based retriever (e.g. Gemini embeddings + Chroma) if you
want to extend the project; the rest of the agent only uses `search()`.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import config

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "i", "my", "me", "you",
    "your", "we", "our", "to", "of", "for", "in", "on", "at", "it", "this", "that", "can", "how",
    "what", "which", "will", "be", "or", "and", "if", "with", "have", "has", "get", "there",
    "any", "much", "long", "take", "please", "tell", "about",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


@dataclass
class Hit:
    section: str
    text: str
    score: float


class BM25Index:
    def __init__(self, sections: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.sections = sections
        self.k1, self.b = k1, b
        # the heading is counted twice so it weighs more than a single body word
        self.docs = [tokenize(f"{title} {title} {body}") for title, body in sections]
        self.avg_len = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        n = len(self.docs)
        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def score(self, query_tokens: list[str], i: int) -> float:
        doc_len = len(self.docs[i])
        total = 0.0
        for t in set(query_tokens):
            f = self.tf[i].get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
            total += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return total

    def search(self, query: str, top_k: int = 2, min_score: Optional[float] = None) -> list[Hit]:
        min_score = config.FAQ_MIN_SCORE if min_score is None else min_score
        q = tokenize(query)
        scored = sorted(((self.score(q, i), i) for i in range(len(self.docs))), reverse=True)
        hits = []
        for s, i in scored[:top_k]:
            if s >= min_score:
                title, body = self.sections[i]
                hits.append(Hit(section=title, text=body, score=round(s, 3)))
        return hits


def load_sections(path: Optional[Path] = None) -> list[tuple[str, str]]:
    text = Path(path or config.DATA_DIR / "policies.md").read_text(encoding="utf-8")
    sections, title, body = [], None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if title:
                sections.append((title, " ".join(body).strip()))
            title, body = line[3:].strip(), []
        elif title and line.strip():
            body.append(line.strip())
    if title:
        sections.append((title, " ".join(body).strip()))
    return sections


@lru_cache(maxsize=1)
def _index() -> BM25Index:
    return BM25Index(load_sections())


def search(query: str, top_k: int = 2) -> list[Hit]:
    return _index().search(query, top_k=top_k)
