"""Graph nodes and routing table.

All business logic lives here as plain functions, so it can be unit tested without LangGraph.
`graph.py` only wires these nodes into a LangGraph StateGraph using the same ROUTING table.

Division of labour:
  * The LLM classifies the message and (optionally) phrases the final reply.
  * Plain Python decides which tool runs, checks the customer's name, applies refund rules,
    and checks that the reply only contains facts the agent really had.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Callable, Optional, TypedDict

import config
from . import db, faq, guardrails, prompts, templates, tools
from .llm import BaseLLM, LLMError

END = "__end__"
ORDER_ID_RE = re.compile(r"\bORD[\s-]?(\d{4})\b", re.I)


class AgentState(TypedDict, total=False):
    thread_id: str
    user_message: str
    history: list
    order_id: Optional[str]
    verified_order_id: Optional[str]
    customer_name: Optional[str]
    intent: Optional[str]
    sentiment: Optional[str]
    confidence: Optional[float]
    summary: Optional[str]
    route: str
    handoff_reason: Optional[str]
    tool_called: str
    tool_result: Optional[dict]
    response: Optional[str]
    response_source: Optional[str]
    grounding_failed: bool
    t0: float
    api_calls0: int
    latency_ms: float
    api_calls: int


# node name -> next node (str) or {route_value: next node}
ROUTING = {
    "start": "guard",
    "guard": {"classify": "classify", "handoff": "handoff", "respond": "respond"},
    "classify": {"faq": "faq", "order": "order", "refund": "refund", "complaint": "complaint",
                 "handoff": "handoff", "respond": "respond"},
    "faq": "respond",
    "order": "respond",
    "refund": "respond",
    "complaint": "respond",
    "handoff": "respond",
    "respond": END,
}
ENTRY = "start"


def _history_text(history: list, limit: int = 6) -> str:
    return "\n".join(f"{h['role']}: {h['text']}" for h in history[-limit:])


def _name_supported(name: str, context: str) -> bool:
    """A name the LLM extracted must literally appear in the conversation (no hallucinated names)."""
    words = re.findall(r"[a-z]+", context.lower())
    return all(tok in words for tok in re.findall(r"[a-z]+", name.lower()))


def public_view(tool_result: dict) -> dict:
    """What the LLM may see: never the internal reason behind a no_match."""
    return {k: v for k, v in (tool_result or {}).items() if k != "reason" or
            tool_result.get("status") != "no_match"}


def make_nodes(llm: BaseLLM, db_path: Optional[Path] = None, today: Optional[date] = None,
               llm_responses: Optional[bool] = None) -> dict[str, Callable[[AgentState], dict]]:
    db_path = Path(db_path or config.DB_PATH)
    today = today or date.fromisoformat(config.DEMO_TODAY)
    use_llm_reply = config.LLM_RESPONSES if llm_responses is None else llm_responses

    def start(state: AgentState) -> dict:
        history = list(state.get("history") or [])
        history.append({"role": "customer", "text": state["user_message"]})
        return {
            "history": history, "route": "classify", "intent": None, "sentiment": None,
            "confidence": None, "summary": None, "handoff_reason": None, "tool_called": "none",
            "tool_result": None, "response": None, "response_source": None,
            "grounding_failed": False, "t0": time.time(), "api_calls0": llm.api_calls,
        }

    def guard(state: AgentState) -> dict:
        flag = guardrails.check(state["user_message"])
        if flag == "injection":
            return {"route": "respond", "intent": "blocked", "tool_result": {"status": "blocked"}}
        if flag == "abusive":
            return {"route": "handoff", "intent": "abusive", "handoff_reason": "abusive_language"}
        return {"route": "classify"}

    def classify(state: AgentState) -> dict:
        message = state["user_message"]
        prompt = prompts.classify_prompt(
            message, _history_text(state["history"][:-1]),
            state.get("order_id"), state.get("customer_name"),
        )
        try:
            out = llm.generate_json(prompt)
        except LLMError:
            return {"route": "handoff", "handoff_reason": "llm_unavailable", "intent": "error"}

        intent = str(out.get("intent", "other")).strip().lower()
        if intent not in prompts.INTENTS:
            intent = "other"
        sentiment = str(out.get("sentiment", "calm")).strip().lower()
        if sentiment not in ("calm", "upset", "very_upset"):
            sentiment = "calm"
        try:
            confidence = max(0.0, min(1.0, float(out.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        update: dict = {"intent": intent, "sentiment": sentiment, "confidence": confidence,
                        "summary": str(out.get("summary") or "")[:200]}

        # Order IDs have a fixed format, so they are extracted with a regex, not by the LLM.
        ids = ORDER_ID_RE.findall(message)
        if ids:
            update["order_id"] = f"ORD{ids[-1]}"
        name = out.get("customer_name")
        context = " ".join(h["text"] for h in state["history"] if h["role"] == "customer")
        if isinstance(name, str) and name.strip() and _name_supported(name, context):
            update["customer_name"] = name.strip()

        if confidence < config.CONFIDENCE_MIN:
            update.update(route="handoff", handoff_reason="low_confidence")
        elif intent == "faq":
            update["route"] = "faq"
        elif intent == "order_status":
            update["route"] = "order"
        elif intent == "refund_request":
            update["route"] = "refund"
        elif intent == "complaint":
            if sentiment == "very_upset":
                update.update(route="handoff", handoff_reason="very_upset")
            else:
                update["route"] = "complaint"
        else:
            update.update(route="respond", tool_result={"status": "out_of_scope"})
        return update

    def faq_node(state: AgentState) -> dict:
        hits = faq.search(state["user_message"])
        if hits:
            result = {"status": "faq_answer",
                      "chunks": [{"section": h.section, "text": h.text} for h in hits]}
        else:
            result = {"status": "no_answer"}
        return {"route": "respond", "tool_called": "faq_search", "tool_result": result}

    def _missing(state: AgentState) -> list[str]:
        return [k for k in ("order_id", "customer_name") if not state.get(k)]

    def order_node(state: AgentState) -> dict:
        missing = _missing(state)
        if missing:
            return {"route": "respond", "tool_called": "none",
                    "tool_result": {"status": "missing_info", "missing": missing}}
        result = tools.order_lookup(state["order_id"], state["customer_name"], db_path)
        update = {"route": "respond", "tool_called": "order_lookup", "tool_result": result}
        if result["status"] == "order_found":
            update["verified_order_id"] = result["order"]["order_id"]
        return update

    def refund_node(state: AgentState) -> dict:
        missing = _missing(state)
        if missing:
            return {"route": "respond", "tool_called": "none",
                    "tool_result": {"status": "missing_info", "missing": missing}}
        lookup = tools.order_lookup(state["order_id"], state["customer_name"], db_path)
        if lookup["status"] != "order_found":
            return {"route": "respond", "tool_called": "order_lookup", "tool_result": lookup}
        order = lookup["order"]
        elig = tools.check_refund_eligibility(order, today)
        if not elig["eligible"]:
            return {"route": "respond", "tool_called": "order_lookup",
                    "verified_order_id": order["order_id"], "tool_result": {
                        "status": "refund_not_eligible", "reason": elig["reason"], "order": order,
                        "return_window_days": config.RETURN_WINDOW_DAYS}}
        summary = state.get("summary") or state["user_message"]
        if order["amount"] > config.REFUND_HANDOFF_LIMIT_INR:
            result = tools.human_handoff("refund_above_limit", summary, order["order_id"],
                                         state.get("thread_id"), db_path)
            return {"route": "respond", "tool_called": "human_handoff",
                    "verified_order_id": order["order_id"], "tool_result": result}
        result = tools.create_ticket(summary, "high", "refund", order["order_id"],
                                     state.get("thread_id"), db_path)
        result["refund_timeline"] = "5-7 business days after the item is received and inspected"
        return {"route": "respond", "tool_called": "create_ticket",
                "verified_order_id": order["order_id"], "tool_result": result}

    def complaint_node(state: AgentState) -> dict:
        priority = "high" if state.get("sentiment") == "upset" else "normal"
        summary = state.get("summary") or state["user_message"]
        # only a VERIFIED order is attached to a ticket, never one the customer merely typed
        result = tools.create_ticket(summary, priority, "complaint", state.get("verified_order_id"),
                                     state.get("thread_id"), db_path)
        return {"route": "respond", "tool_called": "create_ticket", "tool_result": result}

    def handoff_node(state: AgentState) -> dict:
        reason = state.get("handoff_reason") or "unsure"
        summary = state.get("summary") or state["user_message"]
        result = tools.human_handoff(reason, summary, state.get("verified_order_id"),
                                     state.get("thread_id"), db_path)
        return {"route": "respond", "tool_called": "human_handoff", "tool_result": result}

    def respond(state: AgentState) -> dict:
        result = state.get("tool_result") or {"status": "out_of_scope"}
        response = templates.render(result, state.get("handoff_reason"))
        source, grounding_failed = "template", False

        if use_llm_reply and templates.llm_eligible(result):
            try:
                text = llm.generate_text(
                    prompts.respond_prompt(state["user_message"], state.get("intent") or "", public_view(result)))
                allowed = json.dumps(public_view(result)) + " " + " ".join(
                    h["text"] for h in state["history"] if h["role"] == "customer")
                ok, _bad = guardrails.grounding_ok(text, allowed)
                if ok and text.strip():
                    response, source = text.strip(), "llm"
                else:
                    grounding_failed, source = True, "template_grounding_fallback"
            except LLMError:
                source = "template_llm_error"

        latency_ms = round((time.time() - state["t0"]) * 1000, 1)
        api_calls = llm.api_calls - state["api_calls0"]
        status = result.get("status", "")
        if result.get("reason"):
            status = f"{status}:{result['reason']}"
        history = list(state["history"]) + [{"role": "agent", "text": response}]
        db.log_turn(state.get("thread_id"), state["user_message"], state.get("intent"),
                    state.get("tool_called"), status, response, source, latency_ms, api_calls, db_path)
        return {"response": response, "response_source": source, "history": history,
                "grounding_failed": grounding_failed, "latency_ms": latency_ms,
                "api_calls": api_calls}

    return {"start": start, "guard": guard, "classify": classify, "faq": faq_node,
            "order": order_node, "refund": refund_node, "complaint": complaint_node,
            "handoff": handoff_node, "respond": respond}


def run_sequential(nodes: dict, state: AgentState) -> AgentState:
    """Run the same nodes with the same ROUTING table, without LangGraph (used by tests/eval
    helpers). The real app uses graph.build_graph(), which adds conversation memory."""
    state = dict(state)
    current = ENTRY
    while current != END:
        state.update(nodes[current](state))
        target = ROUTING[current]
        current = target if isinstance(target, str) else target[state["route"]]
    return state
