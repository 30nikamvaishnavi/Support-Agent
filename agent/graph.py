"""LangGraph wiring: nodes + routing from nodes.py, with per-thread conversation memory."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .llm import BaseLLM
from .nodes import ENTRY, ROUTING, AgentState, make_nodes
from .nodes import END as _END


def _route(state: AgentState) -> str:
    return state["route"]


def build_graph(llm: BaseLLM, db_path: Optional[Path] = None, llm_responses: Optional[bool] = None):
    nodes = make_nodes(llm, db_path=db_path, llm_responses=llm_responses)
    builder = StateGraph(AgentState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.set_entry_point(ENTRY)
    for name, target in ROUTING.items():
        if isinstance(target, dict):
            builder.add_conditional_edges(name, _route, target)
        else:
            builder.add_edge(name, END if target == _END else target)
    # MemorySaver keeps history, order_id and customer_name per thread_id within this process.
    return builder.compile(checkpointer=MemorySaver())


def run_turn(graph, thread_id: str, message: str) -> dict:
    """Send one customer message and return the final state for that turn."""
    return graph.invoke(
        {"user_message": message, "thread_id": thread_id},
        config={"configurable": {"thread_id": thread_id}},
    )
