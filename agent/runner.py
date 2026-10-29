"""One entry point for the UI, the CLI and the setup check.

Uses LangGraph (with conversation memory) when it is installed. If it is not, it falls back
to the built-in sequential runner, which executes the very same nodes and routing table.
`Agent.engine` says which one is active, and the UI shows it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .llm import BaseLLM
from .nodes import make_nodes, run_sequential


class Agent:
    def __init__(self, llm: BaseLLM, db_path: Optional[Path] = None,
                 llm_responses: Optional[bool] = None, prefer_langgraph: bool = True):
        self.llm = llm
        self.engine = "sequential"
        self._memory: dict[str, dict] = {}
        self._graph = None
        if prefer_langgraph:
            try:
                from .graph import build_graph

                self._graph = build_graph(llm, db_path=db_path, llm_responses=llm_responses)
                self.engine = "langgraph"
            except ImportError:
                pass
        if self._graph is None:
            self._nodes = make_nodes(llm, db_path=db_path, llm_responses=llm_responses)

    def chat(self, thread_id: str, message: str) -> dict:
        """Send one customer message; returns the final state of that turn."""
        if self._graph is not None:
            from .graph import run_turn

            return run_turn(self._graph, thread_id, message)
        state = dict(self._memory.get(thread_id, {}))
        state.update({"user_message": message, "thread_id": thread_id})
        out = run_sequential(self._nodes, state)
        self._memory[thread_id] = {k: out[k] for k in ("history", "order_id", "verified_order_id", "customer_name") if k in out}
        return out
