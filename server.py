"""Web server for the Support Agent: a JSON API over the real Python agent + the web/ front end.

Standard library only (no Flask/FastAPI needed). The browser UI in web/ is a thin client: it never
re-implements agent logic and only shows what this API returns.

    python server.py --demo      # no API key: keyword classifier (NOT AI), clearly labelled
    python server.py             # real agent, needs GEMINI_API_KEY
    -> open http://127.0.0.1:8000

Environment: HOST (default 127.0.0.1), PORT (default 8000), ADMIN_VIEW=1 (show all tickets, logs
and the orders table), DEMO_MODE=1 (same as --demo).
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import config
from agent import db, faq

WEB_DIR = config.ROOT / "web"
MAX_BODY_BYTES = 8_000
MAX_MESSAGE_CHARS = 1_000
CHAT_LIMIT_PER_MINUTE = int(os.getenv("CHAT_LIMIT_PER_MINUTE", "30"))


class RateLimiter:
    """Per-client sliding window, to protect the API quota when the demo is public."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


@dataclass
class EvalJob:
    running: bool = False
    demo: bool = False
    lines: list = field(default_factory=list)
    returncode: Optional[int] = None
    started_at: Optional[float] = None


@dataclass
class AppState:
    agent: object
    db_path: Path
    demo: bool
    admin_view: bool
    host: str
    startup_error: str = ""
    model: str = ""
    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(CHAT_LIMIT_PER_MINUTE))
    chat_lock: threading.Lock = field(default_factory=threading.Lock)
    job: EvalJob = field(default_factory=EvalJob)
    job_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def can_run_eval(self) -> bool:
        return self.admin_view or self.host in ("127.0.0.1", "localhost")


def build_state(demo: bool, host: str, db_path: Optional[Path] = None,
                admin_view: Optional[bool] = None) -> AppState:
    db_path = Path(db_path or config.DB_PATH)
    db.init_db(db_path)
    admin = os.getenv("ADMIN_VIEW", "0") == "1" if admin_view is None else admin_view
    agent, error, model = None, "", ""
    try:
        from agent.runner import Agent

        if demo:
            from agent.demo_llm import RuleBasedDemoLLM

            llm = RuleBasedDemoLLM()
            agent = Agent(llm, db_path=db_path, llm_responses=False)
            model = llm.model
        elif not config.GEMINI_API_KEY:
            error = "GEMINI_API_KEY is not set. Add it to .env, or start with --demo to try without a key."
        else:
            from agent.llm import GeminiClient

            llm = GeminiClient()
            agent = Agent(llm, db_path=db_path)
            model = llm.model
    except Exception as exc:  # noqa: BLE001 - shown to the user in the UI
        error = f"Could not start the agent: {type(exc).__name__}: {exc}"
    return AppState(agent=agent, db_path=db_path, demo=demo, admin_view=admin, host=host,
                    startup_error=error, model=model)


# ---------------------------------------------------------------- evaluation results helpers
def _results_dir(mode: str) -> Path:
    return config.RESULTS_DIR / "demo" if mode == "demo" else config.RESULTS_DIR


def read_eval_summary(mode: str) -> Optional[dict]:
    path = _results_dir(mode) / "eval_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_eval_cases(mode: str) -> Optional[list]:
    path = _results_dir(mode) / "eval_cases_results.csv"
    if not path.exists():
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("adversarial", "tool_ok", "expect_ok", "leak_ok", "passed", "grounding_failed"):
                row[key] = row.get(key) == "True"
            row["intent_ok"] = {"True": True, "False": False}.get(row.get("intent_ok"))
            rows.append(row)
    return rows


def start_eval(state: AppState, demo: bool) -> tuple[bool, str]:
    with state.job_lock:
        if state.job.running:
            return False, "An evaluation is already running."
        if not demo and not config.GEMINI_API_KEY:
            return False, "GEMINI_API_KEY is not set, so a real evaluation cannot run. Use the demo run to test the harness."
        state.job = EvalJob(running=True, demo=demo, started_at=time.time())
    cmd = [sys.executable, "-m", "evaluation.run_eval"] + (["--demo"] if demo else [])

    def work():
        try:
            proc = subprocess.Popen(cmd, cwd=str(config.ROOT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                state.job.lines.append(line.rstrip())
            state.job.returncode = proc.wait()
        except Exception as exc:  # noqa: BLE001
            state.job.lines.append(f"Could not start the evaluation: {exc}")
            state.job.returncode = 1
        finally:
            state.job.running = False

    threading.Thread(target=work, daemon=True).start()
    return True, "started"


# ---------------------------------------------------------------- HTTP handler
def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SupportAgent/1.0"

        def log_message(self, fmt, *args):  # quiet; never log message bodies
            pass

        # ---- helpers
        def _send(self, code: int, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, message: str):
            self._send(code, {"error": message})

        def _read_json(self) -> Optional[dict]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._error(413 if length > MAX_BODY_BYTES else 400, "Request body missing or too large.")
                return None
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._error(400, "Body must be valid JSON.")
                return None
            if not isinstance(data, dict):
                self._error(400, "Body must be a JSON object.")
                return None
            return data

        def _client(self) -> str:
            return self.client_address[0]

        # ---- routes
        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            qs = parse_qs(url.query)
            path = url.path
            if path.startswith("/api/"):
                return self._api_get(path, qs)
            return self._static(path)

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/chat":
                return self._chat()
            if path == "/api/eval/run":
                return self._eval_run()
            self._error(404, "Not found.")

        def _api_get(self, path, qs):
            if path == "/api/health":
                return self._send(200, {
                    "ready": state.agent is not None, "error": state.startup_error,
                    "engine": getattr(state.agent, "engine", None), "model": state.model,
                    "demo": state.demo, "admin_view": state.admin_view,
                    "can_run_eval": state.can_run_eval,
                })
            if path == "/api/policies":
                return self._send(200, {"sections": [{"section": t, "text": b} for t, b in faq.load_sections()]})
            if path == "/api/faq/search":
                q = (qs.get("q", [""])[0])[:300]
                hits = faq.search(q, top_k=3) if q.strip() else []
                return self._send(200, {"query": q, "hits": [
                    {"section": h.section, "text": h.text, "score": h.score} for h in hits]})
            if path == "/api/tickets":
                thread = qs.get("thread_id", [""])[0]
                tickets = db.list_tickets(state.db_path)
                if not state.admin_view:  # visitors only see tickets from their own conversation
                    tickets = [t for t in tickets if thread and t["thread_id"] == thread]
                return self._send(200, {"tickets": tickets, "scope": "all" if state.admin_view else "conversation"})
            if path in ("/api/logs", "/api/orders"):
                if not state.admin_view:
                    return self._error(403, "Admin view is off (set ADMIN_VIEW=1 to enable).")
                if path == "/api/logs":
                    return self._send(200, {"logs": db.list_logs(state.db_path, 200)})
                conn = db.connect(state.db_path)
                try:
                    rows = [dict(r) for r in conn.execute("SELECT * FROM orders ORDER BY order_id")]
                finally:
                    conn.close()
                return self._send(200, {"orders": rows})
            if path in ("/api/eval/summary", "/api/eval/cases"):
                mode = "demo" if qs.get("mode", ["real"])[0] == "demo" else "real"
                data = read_eval_summary(mode) if path.endswith("summary") else read_eval_cases(mode)
                if data is None:
                    return self._error(404, "No evaluation results yet. Run: python -m evaluation.run_eval"
                                       + (" --demo" if mode == "demo" else ""))
                key = "summary" if path.endswith("summary") else "cases"
                return self._send(200, {key: data, "mode": mode})
            if path == "/api/eval/status":
                job = state.job
                return self._send(200, {"running": job.running, "demo": job.demo,
                                        "returncode": job.returncode, "lines": job.lines[-60:]})
            self._error(404, "Not found.")

        def _chat(self):
            if state.agent is None:
                return self._error(503, state.startup_error or "Agent is not ready.")
            if not state.limiter.allow(self._client()):
                return self._error(429, "Too many messages. Please wait a minute and try again.")
            data = self._read_json()
            if data is None:
                return
            thread_id, message = str(data.get("thread_id", ""))[:64], str(data.get("message", "")).strip()
            if not thread_id or not message:
                return self._error(400, "thread_id and message are required.")
            if len(message) > MAX_MESSAGE_CHARS:
                return self._error(400, f"Message is too long (max {MAX_MESSAGE_CHARS} characters).")
            with state.chat_lock:
                s = state.agent.chat(thread_id, message)
            result = s.get("tool_result") or {}
            self._send(200, {
                "response": s.get("response"), "intent": s.get("intent"),
                "sentiment": s.get("sentiment"), "confidence": s.get("confidence"),
                "tool_called": s.get("tool_called"), "tool_status": result.get("status"),
                "ticket_id": result.get("ticket_id"), "response_source": s.get("response_source"),
                "grounding_failed": bool(s.get("grounding_failed")), "latency_ms": s.get("latency_ms"),
                "api_calls": s.get("api_calls"), "engine": state.agent.engine,
            })

        def _eval_run(self):
            if not state.can_run_eval:
                return self._error(403, "Running evaluations is disabled on this server.")
            data = self._read_json()
            if data is None:
                return
            ok, msg = start_eval(state, bool(data.get("demo")))
            self._send(202 if ok else 409, {"started": ok, "message": msg})

        def _static(self, path: str):
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (WEB_DIR / rel).resolve()
            if WEB_DIR.resolve() not in target.parents or not target.is_file():
                return self._error(404, "Not found.")
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

    return Handler


def create_server(state: AppState, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(state))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="keyword classifier, no API key (NOT AI)")
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = ap.parse_args(argv)
    state = build_state(args.demo or config.DEMO_MODE, args.host)
    if state.startup_error:
        print(f"Warning: {state.startup_error}")
    server = create_server(state, args.host, args.port)
    mode = "DEMO (keyword classifier, not AI)" if state.demo else f"model {state.model or 'not ready'}"
    print(f"Support Agent running at http://{args.host}:{args.port}  [{mode}, engine {getattr(state.agent, 'engine', '-')}]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
