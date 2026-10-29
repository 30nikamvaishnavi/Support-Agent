"""Streamlit UI: Chat, Admin (tickets + logs) and Evaluation pages.

Run:  streamlit run app.py
"""
import json
import uuid

import pandas as pd
import streamlit as st

import config
from agent import db

st.set_page_config(page_title="ShopSphere Support Agent", page_icon="🛒", layout="wide")
db.init_db(config.DB_PATH)

page = st.sidebar.radio("Page", ["Chat", "Admin", "Evaluation"])
st.sidebar.caption("Demo with a fictional store and fictional customers. Do not enter real personal data.")


@st.cache_resource
def get_agent():
    from agent.runner import Agent

    if config.DEMO_MODE:
        from agent.demo_llm import RuleBasedDemoLLM

        return Agent(RuleBasedDemoLLM(), db_path=config.DB_PATH, llm_responses=False)
    from agent.llm import GeminiClient

    return Agent(GeminiClient(), db_path=config.DB_PATH)


if page == "Chat":
    st.title("ShopSphere Support Agent")
    if config.DEMO_MODE:
        st.warning("DEMO MODE: a keyword classifier answers, not an AI model, and replies are "
                   "templates. Set GEMINI_API_KEY and remove DEMO_MODE=1 to use the real agent.")
    elif not config.GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Add it to a .env file and restart, or set DEMO_MODE=1 "
                 "to try the app without a key (keyword classifier, not AI).")
        st.stop()
    try:
        agent = get_agent()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not start the agent: {exc}")
        st.stop()
    st.caption(f"Engine: {agent.engine} · model: "
               f"{'demo (not AI)' if config.DEMO_MODE else config.GEMINI_MODEL} · fictional orders "
               "ORD1001-ORD1030 · try: \"I'm Aarav Mehta, where is my order ORD1001?\"")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.trace = None

    if st.sidebar.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages, st.session_state.trace = [], None
        st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["text"])

    prompt = st.chat_input("Type your message")
    if prompt:
        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                state = agent.chat(st.session_state.thread_id, prompt)
            st.write(state["response"])
        st.session_state.messages.append({"role": "assistant", "text": state["response"]})
        st.session_state.trace = {
            "intent": state.get("intent"), "sentiment": state.get("sentiment"),
            "confidence": state.get("confidence"), "tool_called": state.get("tool_called"),
            "tool_status": (state.get("tool_result") or {}).get("status"),
            "reply_source": state.get("response_source"),
            "latency_ms": state.get("latency_ms"), "api_calls": state.get("api_calls"),
        }

    with st.sidebar:
        st.subheader("Last turn: what the agent did")
        st.json(st.session_state.trace or {})

elif page == "Admin":
    st.title("Admin")
    tickets, logs = db.list_tickets(config.DB_PATH), db.list_logs(config.DB_PATH)
    st.subheader(f"Tickets ({len(tickets)})")
    st.dataframe(pd.DataFrame(tickets), use_container_width=True)
    st.subheader(f"Conversation logs ({len(logs)})")
    if logs:
        df = pd.DataFrame(logs)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Messages per intent")
            st.bar_chart(df["intent"].fillna("none").value_counts())
        with c2:
            st.caption("Average latency (ms) per tool")
            st.bar_chart(df.groupby("tool")["latency_ms"].mean())
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No conversations yet.")

else:
    st.title("Evaluation")
    which = st.radio("Results to show", ["Real run (Gemini)", "Demo run (keyword classifier, NOT AI)"],
                     horizontal=True)
    folder = config.RESULTS_DIR if which.startswith("Real") else config.RESULTS_DIR / "demo"
    summary_path = folder / "eval_summary.json"
    cases_path = folder / "eval_cases_results.csv"
    if not summary_path.exists():
        st.info("No results yet. In a terminal run:  python -m evaluation.run_eval"
                + ("" if which.startswith("Real") else " --demo"))
        st.stop()
    if not which.startswith("Real"):
        st.warning("These numbers come from a keyword classifier written by the project author, who "
                   "had seen the test cases. Treat them as a harness check, not as model quality.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = summary.pop("metadata", {})
    st.caption(f"Run {meta.get('run_date_utc')} · model {meta.get('model')} · "
               f"{meta.get('note', '')}")
    cols = st.columns(4)
    for col, key in zip(cols, ["overall_pass_rate", "intent_accuracy", "tool_selection_accuracy",
                               "adversarial_pass_rate"]):
        value = summary.get(key)
        col.metric(key.replace("_", " "), "n/a" if value is None else f"{value:.1%}")
    st.json(summary)
    if cases_path.exists():
        df = pd.read_csv(cases_path)
        only_failed = st.checkbox("Show only failed cases", value=True)
        st.dataframe(df[~df["passed"]] if only_failed else df, use_container_width=True)
