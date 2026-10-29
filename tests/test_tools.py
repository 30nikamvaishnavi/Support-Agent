import tempfile
import unittest
from datetime import date
from pathlib import Path

import config
from agent import db, tools

TODAY = date.fromisoformat("2026-10-28")


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = db.init_db(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_seed_has_30_orders(self):
        conn = db.connect(self.db)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 30)
        conn.close()

    def test_lookup_ok_name_case_and_order_insensitive(self):
        for name in ("Aarav Mehta", "aarav mehta", "MEHTA AARAV"):
            res = tools.order_lookup("ORD1001", name, self.db)
            self.assertEqual(res["status"], "order_found", name)
        self.assertEqual(res["order"]["status"], "Delivered")

    def test_wrong_name_gives_no_order_data(self):
        res = tools.order_lookup("ORD1002", "Rohan Patil", self.db)
        self.assertEqual(res["status"], "no_match")
        self.assertNotIn("order", res)

    def test_missing_order_and_wrong_name_look_identical_to_caller(self):
        a = tools.order_lookup("ORD9999", "Dev Kapoor", self.db)
        b = tools.order_lookup("ORD1002", "Rohan Patil", self.db)
        self.assertEqual(a["status"], b["status"])  # reason differs only internally

    def test_first_name_only_is_not_enough(self):
        self.assertEqual(tools.order_lookup("ORD1001", "Aarav", self.db)["status"], "no_match")

    def test_missing_info(self):
        self.assertEqual(tools.order_lookup(None, "Aarav Mehta", self.db)["status"], "missing_info")
        self.assertEqual(tools.order_lookup("ORD1001", None, self.db)["status"], "missing_info")

    def test_refund_eligibility(self):
        get = lambda oid: db.get_order(oid, self.db)
        self.assertTrue(tools.check_refund_eligibility(get("ORD1001"), TODAY)["eligible"])
        self.assertEqual(tools.check_refund_eligibility(get("ORD1006"), TODAY)["reason"], "outside_window")
        self.assertEqual(tools.check_refund_eligibility(get("ORD1002"), TODAY)["reason"], "not_delivered")
        self.assertEqual(tools.check_refund_eligibility(get("ORD1008"), TODAY)["reason"], "cancelled")

    def test_create_ticket_and_handoff_are_stored(self):
        t = tools.create_ticket("screen cracked", "high", "refund", "ORD1001", "th1", self.db)
        h = tools.human_handoff("very_upset", "angry", None, "th1", self.db)
        tickets = db.list_tickets(self.db)
        self.assertEqual(len(tickets), 2)
        self.assertEqual({x["id"] for x in tickets}, {t["ticket_id"], h["ticket_id"]})
        self.assertEqual(h["status"], "handoff")


if __name__ == "__main__":
    unittest.main()
