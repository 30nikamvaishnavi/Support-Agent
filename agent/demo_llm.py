"""Keyword-based stand-in for the classifier, for trying the app WITHOUT an API key.

It is explicit opt-in (DEMO_MODE=1 or --demo), always labelled as demo in the UI and in saved
results, and has no text mode, so replies come from deterministic templates. Its rules are
generic keyword rules; they are a rough baseline, not a model.
"""
from __future__ import annotations

import re

from .llm import BaseLLM, LLMError

_NOT_NAMES = {"where", "hello", "hi", "order", "refund", "status", "this", "please", "can", "your",
              "ignore", "system", "write", "python", "do", "how", "what", "which", "the", "my", "i",
              "ord", "admin", "mode"}
_NAME_PAIR = re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b")
_ORDER_ID = re.compile(r"\bORD[\s-]?\d{4}\b", re.I)


def _find_name(message: str):
    for first, last in _NAME_PAIR.findall(message):
        if first.lower() not in _NOT_NAMES and last.lower() not in _NOT_NAMES:
            return f"{first} {last}"
    return None


class RuleBasedDemoLLM(BaseLLM):
    is_demo = True
    model = "demo-rule-based (not an AI model)"

    def __init__(self):
        self.api_calls = 0
        self.cache_hits = 0

    def generate_json(self, prompt: str) -> dict:
        self.api_calls += 1
        # the customer's text sits between the tags in the classification prompt
        # (the instructions above it also mention the tag, so take the LAST opening tag)
        start = prompt.rfind("<customer_message>")
        end = prompt.rfind("</customer_message>")
        msg = prompt[start + len("<customer_message>"):end].strip() if 0 <= start < end else prompt
        low = msg.lower()
        has_order = bool(_ORDER_ID.search(msg))

        refund_word = re.search(r"\b(refund|money back|reimburse)", low)
        if refund_word and (has_order or "money back" in low):
            intent, conf = "refund_request", 0.9
        elif re.search(r"\b(where is|status|track|arrive|when will|has not come|hasn't come|not arrived)\b", low) \
                or (has_order and not refund_word):
            intent, conf = "order_status", 0.9
        elif re.search(r"\b(rude|unhappy|damaged|complain|unacceptable|furious|terrible|angry|late|awful)\b", low):
            intent, conf = "complaint", 0.9
        elif refund_word or re.search(
                r"\b(policy|return|warranty|shipping|ship|payment|pay|cancel|address|charges|how long)\b", low):
            intent, conf = "faq", 0.8
        else:
            intent, conf = "other", 0.7

        if re.search(r"furious|unacceptable|third time|!!", low):
            sentiment = "very_upset"
        elif re.search(r"unhappy|rude|angry|terrible|awful|damaged", low):
            sentiment = "upset"
        else:
            sentiment = "calm"
        return {"intent": intent, "customer_name": _find_name(msg), "sentiment": sentiment,
                "confidence": conf, "summary": msg[:100]}

    def generate_text(self, prompt: str) -> str:
        raise LLMError("demo mode has no text generation; templates are used")
