import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.datasets import generate_synthetic
from evaluation.oracle import OracleLLM
from evaluation.runner import build_system, evaluate
from memcore.llm import ScriptedLLM
from memcore.memory import MemoryRecord, MemoryStore
from memcore.pipeline import SingleHopPipeline
from memcore.recall import RecallBoostExtractor, sibling_candidates
from memcore.retrieval import BM25Retriever

# Mirrors the real FactConsolidation failure: the extractor returns only the value
# that matches world knowledge and leaves out the newer, counter-intuitive one.
Q = "What is the official language of Japan?"


def store():
    return MemoryStore([
        MemoryRecord("5", "The official language of Japan is Japanese.", serial=5),
        MemoryRecord("90", "The official language of Japan is Swedish.", serial=90),
        MemoryRecord("7", "The official language of Sweden is Swedish.", serial=7),   # other subject
        MemoryRecord("8", "The capital of Japan is Tokyo.", serial=8),                # other predicate
    ])


def first_pass_only_japanese():
    return json.dumps({"candidates": [{"fact_id": "5", "answer_entity": "Japanese"}]})


class Sibling(unittest.TestCase):
    def test_recovers_missed_newer_version_and_only_that(self):
        hits = BM25Retriever(store()).search(Q, 10)
        base = RecallBoostExtractor(ScriptedLLM([first_pass_only_japanese()]), "sibling")
        r = base.extract(Q, hits)
        self.assertEqual({c.fact_id for c in r.candidates}, {"5", "90"})   # not 7 (subject) or 8 (predicate)
        self.assertEqual({c.value for c in r.candidates}, {"Japanese", "Swedish"})
        self.assertIn("sibling: +1", r.notes)

    def test_pipeline_now_answers_with_newest(self):
        llm = ScriptedLLM([first_pass_only_japanese()])
        p = SingleHopPipeline(BM25Retriever(store()), RecallBoostExtractor(llm, "sibling"))
        self.assertEqual(p.run(Q).answer, "Swedish")

    def test_no_template_no_expansion(self):
        s = MemoryStore([MemoryRecord("a", "I live in Pune."), MemoryRecord("b", "I moved to Chennai last year.")])
        hits = BM25Retriever(s).search("where do I live", 5)
        base = RecallBoostExtractor(ScriptedLLM([json.dumps({"candidates": [{"fact_id": "a", "answer_entity": "Pune"}]})]), "sibling")
        self.assertEqual(len(base.extract("where do I live", hits).candidates), 1)

    def test_value_not_in_text_is_skipped(self):
        hits = BM25Retriever(store()).search(Q, 10)
        r = RecallBoostExtractor(ScriptedLLM([json.dumps({"candidates": [{"fact_id": "5", "answer_entity": "Klingon"}]})]), "sibling")
        self.assertEqual(len(r.extract(Q, hits).candidates), 1)


class LLMSecondPass(unittest.TestCase):
    def test_merges_second_pass_and_drops_invented_ids(self):
        second = json.dumps({"candidates": [{"fact_id": "90", "answer_entity": "Swedish"},
                                            {"fact_id": "999", "answer_entity": "Ghost"}]})
        llm = ScriptedLLM([first_pass_only_japanese(), second])
        r = RecallBoostExtractor(llm, "recall").extract(Q, BM25Retriever(store()).search(Q, 10))
        self.assertEqual({c.fact_id for c in r.candidates}, {"5", "90"})
        self.assertEqual(r.dropped_unknown_ids, 1)
        self.assertIn("<RECALL>", llm.prompts[1])
        self.assertIn("Already found item IDs: 5", llm.prompts[1])

    def test_skipped_when_everything_already_extracted(self):
        s = MemoryStore([MemoryRecord("5", "The official language of Japan is Japanese.", serial=5)])
        llm = ScriptedLLM([first_pass_only_japanese()])        # a 2nd call would raise IndexError
        r = RecallBoostExtractor(llm, "recall").extract(Q, BM25Retriever(s).search(Q, 10))
        self.assertEqual(len(r.candidates), 1)
        self.assertEqual(len(llm.prompts), 1)

    def test_bad_second_pass_keeps_first_results(self):
        llm = ScriptedLLM([first_pass_only_japanese(), "not json"])
        r = RecallBoostExtractor(llm, "recall").extract(Q, BM25Retriever(store()).search(Q, 10))
        self.assertEqual([c.fact_id for c in r.candidates], ["5"])
        self.assertFalse(r.parse_ok)

    def test_both_mode_and_bad_mode(self):
        with self.assertRaises(ValueError):
            RecallBoostExtractor(ScriptedLLM([]), "nope")


class Wiring(unittest.TestCase):
    def test_system_name_suffixes(self):
        ds = generate_synthetic(0)
        for name in ["fixed_bm25_latest+sibling", "routed_full+recall", "routed_llm+both"]:
            self.assertEqual(build_system(name, ds, OracleLLM(ds)).name, name)
        with self.assertRaises(ValueError):
            build_system("routed_full+bogus", ds, OracleLLM(ds))
        with self.assertRaises(ValueError):
            build_system("direct_bm25+recall", ds, OracleLLM(ds))

    def test_recall_improves_under_extractor_noise_mechanism_only(self):
        # The stand-in LLM draws its misses independently on the second pass, so this
        # checks the mechanism (union of two passes), not real-model behaviour.
        ds = [generate_synthetic(s) for s in range(4)]
        res = evaluate(ds, lambda d: OracleLLM(d, miss_rate=0.25), ["routed_full", "routed_full+recall"])
        s = res["summary"]["systems"]
        self.assertGreater(s["routed_full+recall"]["accuracy"], s["routed_full"]["accuracy"])
        self.assertGreater(s["routed_full+recall"]["llm_calls_per_q"], s["routed_full"]["llm_calls_per_q"])


if __name__ == "__main__":
    unittest.main()
