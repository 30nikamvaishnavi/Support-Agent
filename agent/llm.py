"""Gemini wrapper with rate limiting, retries with backoff and an on-disk cache.

There is deliberately NO fake/offline model here: if the API fails, callers get an LLMError
and must handle it honestly (the agent hands the conversation to a human).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import config


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a usable answer."""


class BaseLLM:
    """Interface the agent depends on. Tests provide their own implementations."""

    api_calls: int = 0
    cache_hits: int = 0

    def generate_json(self, prompt: str) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_text(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class RateLimiter:
    """Allows at most `rpm` requests per minute (simple spacing)."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(1, rpm)
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class DiskCache:
    """Tiny SQLite key-value cache so re-runs do not repeat paid/limited API calls."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, key: str) -> Optional[str]:
        with sqlite3.connect(str(self.path)) as conn:
            row = conn.execute("SELECT v FROM cache WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (key, value))


def _retryable(err: Exception) -> bool:
    code = getattr(err, "code", None)
    text = str(err).upper()
    return code in (429, 500, 503) or any(
        s in text for s in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "OVERLOADED")
    )


def parse_json_text(text: str) -> dict:
    """Parse model output as a JSON object, tolerating ```json fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("model JSON was not an object")
    return data


class GeminiClient(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 rpm: Optional[int] = None, cache_path: Optional[Path] = None,
                 max_retries: Optional[int] = None):
        api_key = api_key or config.GEMINI_API_KEY
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")
        from google import genai  # imported lazily so tests do not need the package
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model or config.GEMINI_MODEL
        self.limiter = RateLimiter(rpm or config.REQUESTS_PER_MINUTE)
        self.cache = DiskCache(cache_path or config.CACHE_PATH)
        self.max_retries = config.MAX_RETRIES if max_retries is None else max_retries
        self.api_calls = 0
        self.cache_hits = 0

    def _call(self, prompt: str, json_mode: bool) -> str:
        key = hashlib.sha256(f"{self.model}|{int(json_mode)}|{prompt}".encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        cfg = self._types.GenerateContentConfig(
            temperature=0.0 if json_mode else 0.2,
            response_mime_type="application/json" if json_mode else None,
            max_output_tokens=400 if not json_mode else None,
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                self.api_calls += 1
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                text = (resp.text or "").strip()
                if not text:
                    raise LLMError("empty response")
                self.cache.set(key, text)
                return text
            except Exception as exc:  # noqa: BLE001 - we re-raise as LLMError below
                last_err = exc
                if not _retryable(exc) or attempt == self.max_retries:
                    break
                time.sleep(min(60, 2 ** (attempt + 1)))
        raise LLMError(f"Gemini call failed: {last_err}")

    def generate_json(self, prompt: str) -> dict:
        return parse_json_text(self._call(prompt, json_mode=True))

    def generate_text(self, prompt: str) -> str:
        return self._call(prompt, json_mode=False)
