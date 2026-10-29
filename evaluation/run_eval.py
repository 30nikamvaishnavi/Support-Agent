"""Evaluate the agent on data/eval_cases.json.

Usage:
    python -m evaluation.run_eval --dry-run          # validate cases, show how many API calls
    python -m evaluation.run_eval                    # real run (needs GEMINI_API_KEY)
    python -m evaluation.run_eval --no-llm-response  # fewer API calls: templates for replies
    python -m evaluation.run_eval --demo             # keyword classifier, NO AI, no key needed;
                                                     # saved under results/demo/ and labelled as demo

Results are written to results/eval_cases_results.csv and results/eval_summary.json
(demo runs: results/demo/).
Nothing here invents numbers: every metric is computed from the run that just happened.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import config
from agent import db
from agent.nodes import make_nodes, run_sequential

CASES_PATH = config.DATA_DIR / "eval_cases.json"


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in eval_cases.json")
    for c in cases:
        for key in ("id", "message", "expected_tool", "expect", "must_not_contain", "adversarial"):
            if key not in c:
                raise ValueError(f"case {c.get('id')} is missing '{key}'")
    return cases


def score_case(case: dict, state: dict) -> dict:
    response = (state.get("response") or "").lower()
    intent_ok = None
    if case.get("expected_intent"):
        intent_ok = state.get("intent") == case["expected_intent"]
    tool_ok = state.get("tool_called") == case["expected_tool"]
    expect_ok = all(any(p.lower() in response for p in group) for group in case["expect"])
    leak_ok = not any(bad.lower() in response for bad in case["must_not_contain"])
    passed = tool_ok and expect_ok and leak_ok and intent_ok is not False
    return {
        "id": case["id"], "adversarial": case["adversarial"], "message": case["message"],
        "expected_intent": case.get("expected_intent"), "intent": state.get("intent"),
        "intent_ok": intent_ok, "expected_tool": case["expected_tool"],
        "tool": state.get("tool_called"), "tool_ok": tool_ok, "expect_ok": expect_ok,
        "leak_ok": leak_ok, "passed": passed, "response_source": state.get("response_source"),
        "grounding_failed": bool(state.get("grounding_failed")),
        "latency_ms": state.get("latency_ms"), "api_calls": state.get("api_calls"),
        "response": state.get("response"),
    }


def summarize(rows: list[dict]) -> dict:
    def rate(items):
        return round(sum(items) / len(items), 4) if items else None

    intent = [r["intent_ok"] for r in rows if r["intent_ok"] is not None]
    adv = [r for r in rows if r["adversarial"]]
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    llm_replies = [r for r in rows if r["response_source"] in ("llm", "template_grounding_fallback")]
    return {
        "n_cases": len(rows),
        "overall_pass_rate": rate([r["passed"] for r in rows]),
        "intent_accuracy": rate(intent),
        "tool_selection_accuracy": rate([r["tool_ok"] for r in rows]),
        "fact_accuracy": rate([r["expect_ok"] for r in rows]),
        "adversarial_pass_rate": rate([r["passed"] for r in adv]),
        "adversarial_leaks": sum(not r["leak_ok"] for r in adv),
        "grounding_fallback_rate": rate([r["grounding_failed"] for r in llm_replies]),
        "avg_latency_ms": round(statistics.mean(lat), 1) if lat else None,
        "p95_latency_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
        "total_api_calls": sum(r["api_calls"] or 0 for r in rows),
    }


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(CASES_PATH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-llm-response", action="store_true")
    ap.add_argument("--demo", action="store_true",
                    help="use the keyword-based demo classifier instead of Gemini (not AI)")
    args = ap.parse_args(argv)

    cases = load_cases(Path(args.cases))[: args.limit]
    llm_replies = not args.no_llm_response and not args.demo
    calls = len(cases) * (2 if llm_replies else 1)  # upper bound: classify (+ reply)
    minutes = calls / config.REQUESTS_PER_MINUTE
    if args.demo:
        print(f"{len(cases)} cases, DEMO classifier (no API calls, not an AI model).")
    else:
        print(f"{len(cases)} cases, up to {calls} Gemini calls, ~{minutes:.0f} min at "
              f"{config.REQUESTS_PER_MINUTE} requests/min (cached calls are free on re-runs).")
    if args.dry_run:
        print("Dry run OK: cases are valid.")
        return 0

    if args.demo:
        from agent.demo_llm import RuleBasedDemoLLM

        llm = RuleBasedDemoLLM()
    else:
        from agent.llm import GeminiClient, LLMError

        try:
            llm = GeminiClient()
        except LLMError as exc:
            print(f"Cannot run a real evaluation: {exc}\nTip: --demo runs the harness without a key.")
            return 1

    eval_db = db.init_db(config.EVAL_DB_PATH, reset=True)  # fresh DB: eval never touches real tickets
    nodes = make_nodes(llm, db_path=eval_db, llm_responses=llm_replies)
    rows = []
    for i, case in enumerate(cases, 1):
        state = run_sequential(nodes, {"user_message": case["message"], "thread_id": f"eval-{case['id']}"})
        row = score_case(case, state)
        rows.append(row)
        print(f"[{i}/{len(cases)}] {case['id']}: {'PASS' if row['passed'] else 'FAIL'}")

    summary = summarize(rows)
    summary["metadata"] = {
        "run_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": getattr(llm, "model", "unknown"), "llm_replies": llm_replies, "requests_per_minute": config.REQUESTS_PER_MINUTE,
        "mode": "demo_keyword_classifier_NOT_AI" if args.demo else "gemini",
        "api_calls_made": llm.api_calls, "cache_hits": llm.cache_hits,
        "langgraph": _version("langgraph"), "google-genai": _version("google-genai"),
        "note": "Test cases were written by the project author; small-scale, not a benchmark.",
    }
    out_dir = config.RESULTS_DIR / "demo" if args.demo else config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_cases_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "metadata"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
