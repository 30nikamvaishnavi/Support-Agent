"""Test doubles. These are for unit tests ONLY and are never used by the app or the evaluation."""
import re

import config
from agent.llm import BaseLLM, LLMError


class ScriptedLLM(BaseLLM):
    """Returns a fixed classification dict; text replies are canned."""

    def __init__(self, classification=None, text="ok", fail=False):
        self.classification = classification or {}
        self.text = text
        self.fail = fail
        self.api_calls = 0
        self.cache_hits = 0

    def generate_json(self, prompt):
        self.api_calls += 1
        if self.fail:
            raise LLMError("simulated outage")
        return dict(self.classification)

    def generate_text(self, prompt):
        self.api_calls += 1
        if self.fail:
            raise LLMError("simulated outage")
        return self.text


class OracleLLM(BaseLLM):
    """A 'perfect classifier' that reads the expected intent from the eval case.

    Used to check that tools, routing and templates are consistent with eval_cases.json.
    It says NOTHING about real model quality.
    """

    def __init__(self, case):
        self.case = case
        self.api_calls = 0
        self.cache_hits = 0

    def generate_json(self, prompt):
        self.api_calls += 1
        msg = self.case["message"]
        name = re.search(r"(?:I'm|I am|This is|name|here\.?|for)\s+([A-Z][a-z]+ [A-Z][a-z]+)", msg)
        name = name.group(1) if name else None
        if not name:
            m = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)[,.]?\s+(?:here|please)|,\s*([A-Z][a-z]+ [A-Z][a-z]+)\.", msg)
            if m:
                name = m.group(1) or m.group(2)
        if not name:
            m = re.search(r"\b(Aarav Mehta|Arjun Desai|Kabir Shah|Vikram Singh|Ananya Rao|Dev Kapoor)\b", msg)
            name = m.group(1) if m else None
        very_upset = self.case["id"] == "cmp-03"
        return {
            "intent": self.case.get("expected_intent") or "other",
            "customer_name": name,
            "sentiment": "very_upset" if very_upset else "calm",
            "confidence": 0.95,
            "summary": msg[:100],
        }

    def generate_text(self, prompt):  # never called when llm_responses=False
        raise LLMError("oracle has no text mode")
