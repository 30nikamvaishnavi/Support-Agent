"""Central settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DB_PATH = Path(os.getenv("SUPPORT_DB_PATH", DATA_DIR / "support.db"))
EVAL_DB_PATH = Path(os.getenv("EVAL_DB_PATH", DATA_DIR / "eval.db"))
CACHE_PATH = Path(os.getenv("LLM_CACHE_PATH", DATA_DIR / "llm_cache.db"))

# LLM. Model names change over time: if you get "model not found", set GEMINI_MODEL
# to a current Flash model name from Google AI Studio.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
REQUESTS_PER_MINUTE = int(os.getenv("LLM_RPM", "10"))  # free-tier friendly
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
LLM_RESPONSES = os.getenv("LLM_RESPONSES", "1") == "1"  # 0 = deterministic templates only

# Business rules (fictional store, all editable)
REFUND_HANDOFF_LIMIT_INR = float(os.getenv("REFUND_HANDOFF_LIMIT_INR", "5000"))
RETURN_WINDOW_DAYS = int(os.getenv("RETURN_WINDOW_DAYS", "7"))
CONTACT_WITHIN_HOURS = 24
CONFIDENCE_MIN = float(os.getenv("CONFIDENCE_MIN", "0.5"))
FAQ_MIN_SCORE = float(os.getenv("FAQ_MIN_SCORE", "1.0"))

# The seed orders are dated relative to this "today" so results are repeatable.
DEMO_TODAY = os.getenv("DEMO_TODAY", "2026-10-28")

# DEMO_MODE=1 runs the app with a keyword-based classifier instead of Gemini, so you can try
# the whole pipeline without an API key. It is NOT an AI model and is always labelled as demo.
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
