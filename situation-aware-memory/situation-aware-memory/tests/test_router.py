import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcore.router import RuleRouter, find_period

R = RuleRouter()


class Periods(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(find_period("in March 2025")[1], datetime(2025, 3, 31, 23, 59, 59))
        self.assertEqual(find_period("on 2025-03-14")[0], datetime(2025, 3, 14))
        self.assertEqual(find_period("on 5 June 2024")[0], datetime(2024, 6, 5))
        self.assertEqual(find_period("Feb 29, 2024")[1].day, 29)
        self.assertEqual(find_period("during 2025"), (datetime(2025, 1, 1), datetime(2025, 12, 31, 23, 59, 59)))
        self.assertIsNone(find_period("what is my city"))


class Routing(unittest.TestCase):
    def t(self, q):
        return R.route(q)

    def test_current(self):
        p = self.t("What is my current city of residence?")
        self.assertEqual((p.query_type, p.policy, p.multi_hop), ("current", "latest", False))

    def test_previous(self):
        self.assertEqual(self.t("What was my previous employer?").policy, "previous")
        self.assertEqual(self.t("Where did I used to live?").policy, "previous")

    def test_as_of(self):
        p = self.t("What was my gym in March 2025?")
        self.assertEqual((p.policy, p.policy_params["at"]), ("as_of", datetime(2025, 3, 31, 23, 59, 59)))

    def test_before_date(self):
        p = self.t("What was my employer before 2025?")
        self.assertEqual(p.policy_params["at"], datetime(2024, 12, 31, 23, 59, 59))

    def test_count_with_and_without_window(self):
        self.assertEqual(self.t("How many different cities have I mentioned?").policy_params, {})
        p = self.t("How many different cities did I mention in 2025?")
        self.assertEqual((p.policy, p.policy_params["start"].year), ("count_distinct", 2025))

    def test_multi_hop(self):
        self.assertTrue(self.t("What is the city of my manager?").multi_hop)
        self.assertTrue(self.t("What is my manager's city?").multi_hop)
        self.assertFalse(self.t("What is Priya's city?").multi_hop)
        self.assertFalse(self.t("What is my city of residence?").multi_hop)

    def test_reasons_present(self):
        self.assertTrue(self.t("What is my current city?").reasons)


if __name__ == "__main__":
    unittest.main()


class RealFactConsolidationRegression(unittest.TestCase):
    """50 real FactConsolidation single-hop questions (from a real run).
    All are plain current-value, single-hop questions; 4 of them were once
    misrouted as multi-hop by the 'of the X' rule."""

    def test_all_route_as_current_single_hop(self):
        path = Path(__file__).parent / "fixtures" / "fc_sh_questions.txt"
        qs = [q for q in path.read_text(encoding="utf-8").splitlines() if q.strip()]
        self.assertEqual(len(qs), 50)
        bad = [(q, p.query_type, p.multi_hop) for q in qs
               for p in [R.route(q)] if p.query_type != "current" or p.multi_hop]
        self.assertEqual(bad, [])

    def test_relation_chain_still_multi_hop(self):
        self.assertTrue(R.route("What is the country of citizenship of the spouse of the CEO of Acme?").multi_hop)
