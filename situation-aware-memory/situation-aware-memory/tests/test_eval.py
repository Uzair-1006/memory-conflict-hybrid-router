import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.datasets import generate_synthetic
from evaluation.metrics import mcnemar_exact, match, wilson
from evaluation.oracle import OracleLLM
from evaluation.runner import evaluate
from memcore.agent import Agent


class Metrics(unittest.TestCase):
    def test_match(self):
        self.assertTrue(match("The answer is Bob.", ["bob"]))
        self.assertFalse(match(None, ["bob"]))
        self.assertTrue(match(3, ["3"]))
        self.assertFalse(match(10, ["1"]))          # numeric gold needs an exact match

    def test_wilson_bounds(self):
        lo, hi = wilson(50, 100)
        self.assertTrue(0.39 < lo < 0.41 and 0.59 < hi < 0.61)
        self.assertEqual(wilson(0, 0), (0.0, 0.0))

    def test_mcnemar(self):
        r = mcnemar_exact([True] * 10 + [False] * 5, [False] * 10 + [False] * 5)
        self.assertEqual((r["only_a"], r["only_b"]), (10, 0))
        self.assertAlmostEqual(r["p"], 2 / 2 ** 10)
        self.assertEqual(mcnemar_exact([True, False], [True, False])["p"], 1.0)


class Synthetic(unittest.TestCase):
    def test_deterministic(self):
        a, b = generate_synthetic(3), generate_synthetic(3)
        self.assertEqual([q.text for q in a.questions], [q.text for q in b.questions])
        self.assertEqual([r.fact_id for r in a.store.all()], [r.fact_id for r in b.store.all()])

    def test_covers_all_question_types(self):
        types = {q.qtype for q in generate_synthetic(0).questions}
        self.assertEqual(types, {"current", "previous", "as_of", "count", "multi_hop"})

    def test_versions_change_value(self):
        ds = generate_synthetic(1)
        city = sorted([r for r in ds.store.all() if r.fact_id.startswith("city-")], key=lambda r: r.timestamp)
        vals = [ds.meta[r.fact_id]["value"] for r in city]
        self.assertTrue(all(a != b for a, b in zip(vals, vals[1:])))


class EndToEnd(unittest.TestCase):
    def test_routed_beats_fixed_with_perfect_extractor(self):
        ds = [generate_synthetic(s) for s in range(3)]
        res = evaluate(ds, lambda d: OracleLLM(d), ["fixed_bm25_latest", "routed_full"])
        s = res["summary"]["systems"]
        self.assertEqual(s["routed_full"]["accuracy"], 1.0)
        self.assertLess(s["fixed_bm25_latest"]["accuracy"], 0.5)
        self.assertEqual(s["routed_full"]["router_type_accuracy"], 1.0)

    def test_agent_abstains_with_message(self):
        ds = generate_synthetic(0)
        r = Agent(ds.store, OracleLLM(ds)).answer("What is my current favorite planet?")
        self.assertTrue(r.abstained)
        self.assertIn("clarify", r.message)
        self.assertIn("latest", r.explanation)


if __name__ == "__main__":
    unittest.main()


class FailureClassification(unittest.TestCase):
    def test_labels(self):
        from evaluation.runner import classify_failure as cf
        diag = {"retrieved_texts": ["The CEO is Bob."], "candidate_values": ["Alice"]}
        self.assertIsNone(cf(["Bob"], diag, True))
        self.assertEqual(cf(["Bob"], {"retrieved_texts": ["unrelated"], "candidate_values": []}, False), "retrieval_miss")
        self.assertEqual(cf(["Bob"], diag, False), "extraction_miss")
        self.assertEqual(cf(["Bob"], {**diag, "candidate_values": ["Alice", "Bob"]}, False), "policy_or_selection")
        self.assertEqual(cf(["Bob"], {**diag, "candidate_values": None}, False), "model_error")
        self.assertEqual(cf(["3"], diag, False), "n/a")
        self.assertEqual(cf(["Bob"], None, False), "n/a")

    def test_extractor_noise_produces_extraction_misses(self):
        ds = [generate_synthetic(s) for s in range(3)]
        res = evaluate(ds, lambda d: OracleLLM(d, miss_rate=0.5), ["routed_full"])
        f = res["summary"]["systems"]["routed_full"]["failures"]
        self.assertGreater(f.get("extraction_miss", 0), 0)
        self.assertTrue(all("failure" in r for r in res["rows"]["routed_full"]))


class EmptyGuard(unittest.TestCase):
    def test_cli_refuses_zero_questions(self):
        import tempfile
        from evaluation.run import main
        with tempfile.TemporaryDirectory() as d:
            m, q = Path(d) / "m.jsonl", Path(d) / "q.jsonl"
            m.write_text('{"fact_id": "1", "text": "x"}\n'); q.write_text("")
            with self.assertRaises(SystemExit) as cm:
                main(["--dataset", "jsonl", "--memory", str(m), "--questions", str(q), "--llm", "openai"])
            self.assertEqual(cm.exception.code, 2)
