import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcore import AsOf, Candidate, CountDistinct, Latest, Previous, make_policy


def ts(y, m, d):
    return datetime(y, m, d)


class PaperParity(unittest.TestCase):
    """Latest must reproduce the base paper's max(serial) rule."""

    def test_serial_12_vs_847(self):
        k = [
            Candidate("12", "The CEO of X is Alice.", "Alice", serial=12),
            Candidate("847", "The CEO of X is Bob.", "Bob", serial=847),
        ]
        r = Latest()(k)
        self.assertTrue(r.ok)
        self.assertEqual(r.answer, "Bob")
        self.assertEqual([c.fact_id for c in r.rejected], ["12"])

    def test_order_independent(self):
        a = Candidate("1", "", "A", serial=1)
        b = Candidate("2", "", "B", serial=2)
        self.assertEqual(Latest()([a, b]).answer, Latest()([b, a]).answer)


class Abstention(unittest.TestCase):
    def test_no_candidates(self):
        self.assertTrue(Latest()([]).abstained)

    def test_missing_ordering_metadata(self):
        r = Latest()([Candidate("1", "", "A"), Candidate("2", "", "B")])
        self.assertTrue(r.abstained)

    def test_mixed_metadata_abstains(self):
        r = Latest()([Candidate("1", "", "A", serial=1),
                      Candidate("2", "", "B", timestamp=ts(2026, 1, 1))])
        self.assertTrue(r.abstained)

    def test_tie_with_different_values_abstains(self):
        r = Latest()([Candidate("1", "", "A", serial=5), Candidate("2", "", "B", serial=5)])
        self.assertTrue(r.abstained)

    def test_tie_with_same_value_is_fine(self):
        r = Latest()([Candidate("1", "", "Delhi", serial=5), Candidate("2", "", " delhi ", serial=5)])
        self.assertTrue(r.ok)


class TimestampPolicies(unittest.TestCase):
    def setUp(self):
        self.k = [
            Candidate("a", "I live in Pune", "Pune", timestamp=ts(2024, 1, 10)),
            Candidate("b", "I moved to Chennai", "Chennai", timestamp=ts(2025, 6, 1)),
            Candidate("c", "Now in Bengaluru", "Bengaluru", timestamp=ts(2026, 3, 15)),
        ]

    def test_latest_by_timestamp(self):
        self.assertEqual(Latest()(self.k).answer, "Bengaluru")

    def test_previous(self):
        self.assertEqual(Previous()(self.k).answer, "Chennai")
        self.assertEqual(Previous(n=2)(self.k).answer, "Pune")

    def test_previous_not_enough_history(self):
        self.assertTrue(Previous(n=3)(self.k).abstained)

    def test_previous_collapses_repeats(self):
        k = self.k + [Candidate("d", "Still in Bengaluru", "Bengaluru", timestamp=ts(2026, 5, 1))]
        self.assertEqual(Previous()(k).answer, "Chennai")

    def test_as_of(self):
        self.assertEqual(AsOf(ts(2025, 12, 31))(self.k).answer, "Chennai")
        self.assertEqual(AsOf(ts(2024, 1, 10))(self.k).answer, "Pune")

    def test_as_of_before_any_version(self):
        self.assertTrue(AsOf(ts(2020, 1, 1))(self.k).abstained)

    def test_count_distinct_window(self):
        r = CountDistinct(start=ts(2025, 1, 1), end=ts(2026, 12, 31))(self.k)
        self.assertEqual(r.answer, 2)
        self.assertEqual(CountDistinct()(self.k).answer, 3)


class Registry(unittest.TestCase):
    def test_make_policy(self):
        self.assertEqual(make_policy("previous", n=2).n, 2)
        with self.assertRaises(KeyError):
            make_policy("nope")


if __name__ == "__main__":
    unittest.main()
