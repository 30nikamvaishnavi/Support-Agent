import tempfile
import unittest
from pathlib import Path

from agent import db
from agent.demo_llm import RuleBasedDemoLLM
from agent.llm import LLMError
from agent.runner import Agent


class DemoAndRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = db.init_db(Path(self.tmp.name) / "t.db")
        self.agent = Agent(RuleBasedDemoLLM(), db_path=self.db, llm_responses=False,
                           prefer_langgraph=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_demo_is_labelled_and_has_no_text_mode(self):
        llm = RuleBasedDemoLLM()
        self.assertTrue(llm.is_demo)
        self.assertIn("not an AI", llm.model)
        with self.assertRaises(LLMError):
            llm.generate_text("hi")

    def test_end_to_end_order_status(self):
        s = self.agent.chat("a", "Hi, I'm Aarav Mehta. Where is my order ORD1001?")
        self.assertEqual(s["tool_called"], "order_lookup")
        self.assertIn("Delivered", s["response"])

    def test_memory_between_turns(self):
        s1 = self.agent.chat("m", "I'm Aarav Mehta, where is my order?")
        self.assertEqual(s1["tool_result"]["status"], "missing_info")
        s2 = self.agent.chat("m", "It is ORD1001")
        self.assertEqual(s2["tool_result"]["status"], "order_found")

    def test_threads_do_not_share_memory(self):
        self.agent.chat("one", "I'm Aarav Mehta, where is ORD1001?")
        s = self.agent.chat("two", "Where is my order?")
        self.assertEqual(s["tool_result"]["status"], "missing_info")

    def test_refund_above_limit_creates_handoff_ticket(self):
        s = self.agent.chat("r", "Refund request for ORD1003, Rohan Patil. The laptop screen is cracked.")
        self.assertEqual(s["tool_called"], "human_handoff")
        self.assertEqual(len(db.list_tickets(self.db)), 1)

    def test_every_turn_is_logged(self):
        self.agent.chat("l", "What is your return policy?")
        self.agent.chat("l", "Hello")
        self.assertEqual(len(db.list_logs(self.db)), 2)


if __name__ == "__main__":
    unittest.main()


class VerifiedOrderTests(unittest.TestCase):
    def test_failed_verification_never_attaches_order_to_a_ticket(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = db.init_db(Path(tmp) / "v.db")
            agent = Agent(RuleBasedDemoLLM(), db_path=path, llm_responses=False, prefer_langgraph=False)
            agent.chat("x", "I'm Rohan Patil, check ORD1002")      # wrong name for ORD1002
            agent.chat("x", "The box was damaged and your delivery guy was rude")
            ticket = db.list_tickets(path)[0]
            self.assertEqual(ticket["category"], "complaint")
            self.assertIsNone(ticket["order_id"])

    def test_verified_order_is_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = db.init_db(Path(tmp) / "v.db")
            agent = Agent(RuleBasedDemoLLM(), db_path=path, llm_responses=False, prefer_langgraph=False)
            agent.chat("y", "I'm Aarav Mehta, where is ORD1001?")
            agent.chat("y", "The packaging was damaged")
            self.assertEqual(db.list_tickets(path)[0]["order_id"], "ORD1001")
