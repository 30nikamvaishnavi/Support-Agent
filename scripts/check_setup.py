"""Checks your setup step by step so you know exactly what works.   python -m scripts.check_setup"""
import importlib
import sys
import tempfile
import time
from pathlib import Path

import config

results = []


def step(name, fn):
    try:
        detail = fn()
        print(f"[PASS] {name}" + (f": {detail}" if detail else ""))
        results.append(True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        results.append(False)
        return False


def check_python():
    assert sys.version_info >= (3, 10), f"Python 3.10+ needed, found {sys.version.split()[0]}"
    return sys.version.split()[0]


def check_import(module):
    def run():
        m = importlib.import_module(module)
        return getattr(m, "__version__", "")
    return run


def check_db_and_faq():
    from agent import db, faq, tools

    with tempfile.TemporaryDirectory() as tmp:
        path = db.init_db(Path(tmp) / "t.db")
        res = tools.order_lookup("ORD1001", "Aarav Mehta", path)
        assert res["status"] == "order_found", res
    hits = faq.search("What is your return policy?")
    assert hits and hits[0].section == "Returns", hits
    return "orders + FAQ search work"


def check_langgraph_engine():
    from agent.runner import Agent
    from agent.demo_llm import RuleBasedDemoLLM

    with tempfile.TemporaryDirectory() as tmp:
        from agent import db

        path = db.init_db(Path(tmp) / "g.db")
        agent = Agent(RuleBasedDemoLLM(), db_path=path, llm_responses=False)
        if agent.engine != "langgraph":
            raise RuntimeError("LangGraph is not installed or failed to build (run: pip install -r requirements.txt)")
        s1 = agent.chat("setup-check", "Hi, I'm Aarav Mehta, where is my order?")
        s2 = agent.chat("setup-check", "It is ORD1001")  # needs conversation memory
        assert s2["tool_result"]["status"] == "order_found", s2["tool_result"]
    return "LangGraph graph built and conversation memory works"


def check_gemini():
    from agent.llm import GeminiClient

    llm = GeminiClient()
    t0 = time.time()
    out = llm.generate_json(
        "Return a JSON object with keys intent, customer_name, sentiment, confidence, summary for this "
        "customer message: <customer_message>Where is my order ORD1001? I'm Aarav Mehta</customer_message>")
    assert "intent" in out, out
    return f"model {llm.model} answered in {time.time() - t0:.1f}s, intent={out.get('intent')}"


def check_end_to_end():
    from agent import db
    from agent.llm import GeminiClient
    from agent.runner import Agent

    with tempfile.TemporaryDirectory() as tmp:
        path = db.init_db(Path(tmp) / "e.db")
        agent = Agent(GeminiClient(), db_path=path)
        s = agent.chat("e2e", "I'm Aarav Mehta. Where is my order ORD1001?")
        assert s["tool_called"] == "order_lookup", s
        return f"engine={agent.engine}, reply source={s['response_source']}, reply: {s['response']}"


if __name__ == "__main__":
    step("Python version", check_python)
    for mod in ("langgraph", "google.genai", "streamlit", "pandas"):
        step(f"import {mod}", check_import(mod))
    step("database, tools and FAQ search", check_db_and_faq)
    step("LangGraph engine + memory (keyword classifier, no API)", check_langgraph_engine)
    if config.GEMINI_API_KEY:
        step("Gemini API call", check_gemini)
        step("end-to-end turn with Gemini", check_end_to_end)
    else:
        print("[SKIP] Gemini checks: GEMINI_API_KEY is not set (add it to .env to test the real agent)")
    print("\nAll checks passed." if all(results) else "\nSome checks failed: fix the first [FAIL] above and re-run.")
    sys.exit(0 if all(results) else 1)
