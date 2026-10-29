"""Chat in the terminal.   python -m agent.cli          (needs GEMINI_API_KEY)
                           python -m agent.cli --demo   (no key; keyword classifier, NOT AI)"""
import argparse
import sys
import uuid

import config
from . import db
from .runner import Agent


def build_llm(demo: bool):
    if demo or config.DEMO_MODE:
        from .demo_llm import RuleBasedDemoLLM

        print("*** DEMO MODE: keyword classifier, not an AI model. Replies are templates. ***")
        return RuleBasedDemoLLM(), False
    from .llm import GeminiClient, LLMError

    try:
        return GeminiClient(), None
    except LLMError as exc:
        print(f"{exc}\nTip: use --demo to try the app without a key.")
        sys.exit(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    db.init_db(config.DB_PATH)
    llm, llm_responses = build_llm(args.demo)
    agent = Agent(llm, db_path=config.DB_PATH, llm_responses=llm_responses)
    thread = str(uuid.uuid4())
    print(f"Engine: {agent.engine}. Fictional orders ORD1001-ORD1030. Type 'quit' to exit.")
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("you> "), None):
        message = line.strip()
        if message.lower() in ("quit", "exit"):
            break
        if not message:
            continue
        state = agent.chat(thread, message)
        print(f"agent> {state['response']}")
        print(f"       [intent={state.get('intent')} tool={state.get('tool_called')} "
              f"source={state.get('response_source')} {state.get('latency_ms')} ms]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
