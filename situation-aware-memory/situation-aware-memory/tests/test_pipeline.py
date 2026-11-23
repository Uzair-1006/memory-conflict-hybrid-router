import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcore.extractor import Extractor
from memcore.llm import ScriptedLLM
from memcore.memory import MemoryRecord, MemoryStore
from memcore.pipeline import SingleHopPipeline
from memcore.policies import AsOf, Previous
from memcore.retrieval import BM25Retriever, tokenize


def ceo_store():
    return MemoryStore([
        MemoryRecord("12", "The CEO of Company X is Alice.", serial=12),
        MemoryRecord("847", "The CEO of Company X is Bob.", serial=847),
        MemoryRecord("30", "The CFO of Company X is Carol.", serial=30),
        MemoryRecord("55", "The headquarters of Company Y is Paris.", serial=55),
        MemoryRecord("56", "Dogs are popular pets in many countries.", serial=56),
    ])


def keyword_llm():
    """Offline stand-in for the extractor LLM: returns every item whose text
    contains 'CEO of Company X' (mimics strict subject+predicate matching)."""
    def respond(prompt: str) -> str:
        out = []
        for line in prompt.split("Items:\n")[1].splitlines():
            if line.startswith("[") and "CEO of Company X" in line:
                fid = line[1:line.index("]")]
                out.append({"fact_id": fid, "answer_entity": line.rstrip(".").split(" is ")[-1],
                            "confidence": 0.95})
        return json.dumps({"candidates": out})
    return ScriptedLLM(respond)


class BM25Tests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("The CEO's name, 2026!"), ["the", "ceo", "s", "name", "2026"])

    def test_relevant_ranked_first(self):
        hits = BM25Retriever(ceo_store()).search("Who is the CEO of Company X?", top_k=3)
        self.assertEqual({h.record.fact_id for h in hits[:2]}, {"12", "847"})
        self.assertEqual([h.rank for h in hits], [1, 2, 3])

    def test_empty_store(self):
        self.assertEqual(BM25Retriever(MemoryStore()).search("anything"), [])


class ExtractorTests(unittest.TestCase):
    def hits(self):
        return BM25Retriever(ceo_store()).search("CEO of Company X", top_k=5)

    def test_metadata_comes_from_store_not_llm(self):
        # LLM claims a bogus serial; extractor must ignore it.
        llm = ScriptedLLM([json.dumps({"candidates": [
            {"fact_id": "12", "answer_entity": "Alice", "serial": 99999}]})])
        r = Extractor(llm).extract("CEO of Company X?", self.hits())
        self.assertEqual(r.candidates[0].serial, 12)

    def test_hallucinated_id_dropped(self):
        llm = ScriptedLLM([json.dumps({"candidates": [
            {"fact_id": "12", "answer_entity": "Alice"},
            {"fact_id": "9999", "answer_entity": "Mallory"}]})])
        r = Extractor(llm).extract("q", self.hits())
        self.assertEqual([c.fact_id for c in r.candidates], ["12"])
        self.assertEqual(r.dropped_unknown_ids, 1)

    def test_bad_json_is_reported_not_raised(self):
        r = Extractor(ScriptedLLM(["not json"])).extract("q", self.hits())
        self.assertFalse(r.parse_ok)
        self.assertEqual(r.candidates, [])

    def test_no_hits_skips_llm(self):
        llm = ScriptedLLM([])
        r = Extractor(llm).extract("q", [])
        self.assertEqual(r.candidates, [])
        self.assertEqual(llm.prompts, [])

    def test_confidence_clamped_and_duplicates_ignored(self):
        llm = ScriptedLLM([json.dumps({"candidates": [
            {"fact_id": "12", "answer_entity": "Alice", "confidence": 7},
            {"fact_id": "12", "answer_entity": "Alice"}]})])
        r = Extractor(llm).extract("q", self.hits())
        self.assertEqual(len(r.candidates), 1)
        self.assertEqual(r.candidates[0].confidence, 1.0)


class PipelineTests(unittest.TestCase):
    def pipe(self):
        return SingleHopPipeline(BM25Retriever(ceo_store()), Extractor(keyword_llm()))

    def test_conflict_resolved_to_newest(self):
        r = self.pipe().run("Who is the CEO of Company X?")
        self.assertEqual(r.answer, "Bob")
        self.assertEqual(r.evidence, ["847"])
        self.assertFalse(r.abstained)

    def test_cfo_distractor_not_used(self):
        r = self.pipe().run("Who is the CEO of Company X?")
        self.assertNotIn("30", {c.fact_id for c in r.extraction.candidates})

    def test_abstains_when_nothing_matches(self):
        p = SingleHopPipeline(BM25Retriever(ceo_store()), Extractor(ScriptedLLM(
            lambda _: json.dumps({"candidates": []}))))
        r = p.run("What is the capital of Mars?")
        self.assertTrue(r.abstained)
        self.assertIsNone(r.answer)

    def test_policy_swappable_per_question(self):
        r = self.pipe().run("Who is the CEO of Company X?", policy=Previous())
        self.assertEqual(r.answer, "Alice")


class TimestampPipeline(unittest.TestCase):
    def test_as_of_with_timestamps(self):
        store = MemoryStore([
            MemoryRecord("a", "My city of residence is Pune.", timestamp=datetime(2024, 1, 1)),
            MemoryRecord("b", "My city of residence is Chennai.", timestamp=datetime(2025, 6, 1)),
        ])
        llm = ScriptedLLM(lambda p: json.dumps({"candidates": [
            {"fact_id": "a", "answer_entity": "Pune"},
            {"fact_id": "b", "answer_entity": "Chennai"}]}))
        pipe = SingleHopPipeline(BM25Retriever(store), Extractor(llm))
        r = pipe.run("Where did I live?", policy=AsOf(datetime(2024, 12, 31)))
        self.assertEqual(r.answer, "Pune")


class StoreTests(unittest.TestCase):
    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            MemoryStore([MemoryRecord("1", "a"), MemoryRecord("1", "b")])


if __name__ == "__main__":
    unittest.main()
