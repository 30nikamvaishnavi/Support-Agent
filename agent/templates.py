"""Deterministic reply templates.

They are used (a) for security-sensitive outcomes where the LLM should not improvise
(blocked, no_match, missing_info), (b) when the LLM is disabled/unavailable, and
(c) as the fallback when an LLM reply fails the grounding check.
"""
from __future__ import annotations

import config

LLM_STATUSES = {"order_found", "faq_answer", "ticket_created", "refund_not_eligible", "handoff"}


def llm_eligible(tool_result: dict) -> bool:
    return tool_result.get("status") in LLM_STATUSES


def _order_line(order: dict) -> str:
    status = order["status"]
    if status == "Delivered":
        when = f"It was delivered on {order['delivery_date']}."
    elif status == "Cancelled":
        when = "This order was cancelled."
    elif status == "Delayed":
        when = f"The new estimated delivery date is {order['delivery_date']}."
    else:
        when = f"Estimated delivery date: {order['delivery_date']}."
    return f"Order {order['order_id']} ({order['items']}) is {status}. {when}"


def render(tool_result: dict, handoff_reason: str | None = None) -> str:
    status = tool_result.get("status")
    hours = config.CONTACT_WITHIN_HOURS

    if status == "blocked":
        return ("I can't help with that request. I can help with order status, refunds and "
                "returns, complaints, and questions about our store policies.")
    if status == "out_of_scope":
        return ("I can only help with order status, refunds and returns, complaints, and "
                "questions about our store policies. Is there something in those areas I can "
                "help you with?")
    if status == "missing_info":
        missing = tool_result.get("missing", [])
        need = []
        if "order_id" in missing:
            need.append("your order ID (for example ORD1234)")
        if "customer_name" in missing:
            need.append("the full name on the order")
        return "To look into this, please share " + " and ".join(need) + "."
    if status == "no_match":
        return ("I couldn't find an order matching that order ID and name. Please double-check "
                "both and try again.")
    if status == "order_found":
        return _order_line(tool_result["order"])
    if status == "faq_answer":
        chunk = tool_result["chunks"][0]
        return f"According to our policy ({chunk['section']}): {chunk['text']}"
    if status == "no_answer":
        return ("I couldn't find that in our store policies. I can pass your question to a "
                "human agent if you'd like.")
    if status == "refund_not_eligible":
        order, reason = tool_result["order"], tool_result["reason"]
        window = tool_result["return_window_days"]
        if reason == "not_delivered":
            return (f"Order {order['order_id']} has not been delivered yet (status: "
                    f"{order['status']}), so a return refund isn't possible yet. Returns are "
                    f"accepted within {window} days of delivery.")
        if reason == "cancelled":
            return (f"Order {order['order_id']} was cancelled, so there is nothing to return. "
                    "If you were charged, please ask about the cancellation refund.")
        return (f"Order {order['order_id']} was delivered on {order['delivery_date']}, which is "
                f"outside our {window}-day return window, so it isn't eligible for a refund.")
    if status == "ticket_created":
        tid = tool_result["ticket_id"]
        if tool_result["category"] == "refund":
            return (f"I've created refund ticket #{tid} for order {tool_result['order_id']}. "
                    f"Our team will review it and contact you within {hours} hours. Approved "
                    "refunds arrive within 5-7 business days after the item is received and "
                    "inspected.")
        return (f"I'm sorry about that. I've logged complaint ticket #{tid} "
                f"(priority: {tool_result['priority']}). Our team will follow up within "
                f"{hours} hours.")
    if status == "handoff":
        tid = tool_result["ticket_id"]
        reason = tool_result.get("reason", handoff_reason)
        if reason == "refund_above_limit":
            return (f"Refunds above INR {int(config.REFUND_HANDOFF_LIMIT_INR)} need review by a "
                    f"human agent. I've created ticket #{tid}, and an agent will contact you "
                    f"within {hours} hours.")
        if reason == "llm_unavailable":
            return (f"I'm having technical trouble right now, so I've passed your message to a "
                    f"human agent (ticket #{tid}). They will contact you within {hours} hours.")
        return (f"I'm sorry this has been frustrating. I've passed your conversation to a human "
                f"agent (ticket #{tid}), who will contact you within {hours} hours.")
    return "Sorry, something went wrong. Please try again or ask for a human agent."
