import json
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.longmemeval import load_longmemeval, parse_lme_date
from evaluation.runner import evaluate
from memcore.direct import DirectAnswerer
from memcore.llm import ScriptedLLM
from memcore.llm_router import LLMRouter
from memcore.memory import MemoryRecord, MemoryStore
from memcore.retrieval import BM25Retriever


def store():
    return MemoryStore([
        MemoryRecord("12", "The CEO of Company X is Alice.", serial=12),
        MemoryRecord("847", "The CEO of Company X is Bob.", serial=847),
        MemoryRecord("t1", "I live in Pune.", timestamp=datetime(2024, 3, 2)),
    ])


class Direct(unittest.TestCase):
    def test_answer_and_prompt_contains_versions(self):
        llm = ScriptedLLM(['{"answer": " Bob "}'])
        r = DirectAnswerer(BM25Retriever(store()), llm).run("Who is the CEO of Company X?")
        self.assertEqual(r.answer, "Bob")
        self.assertFalse(r.abstained)
        self.assertIn("version 847", llm.prompts[0])

    def test_timestamp_tag(self):
        llm = ScriptedLLM(['{"answer": "Pune"}'])
        DirectAnswerer(BM25Retriever(store()), llm).run("Where do I live?")
        self.assertIn("date 2024-03-02", llm.prompts[0])

    def test_null_and_bad_json_abstain(self):
        for raw in ['{"answer": null}', '{"answer": "  "}', "oops"]:
            r = DirectAnswerer(BM25Retriever(store()), ScriptedLLM([raw])).run("Who is the CEO?")
            self.assertTrue(r.abstained, raw)

    def test_no_hits_skips_llm(self):
        llm = ScriptedLLM([])
        r = DirectAnswerer(BM25Retriever(MemoryStore()), llm).run("q")
        self.assertTrue(r.abstained)
        self.assertEqual(llm.prompts, [])


class LLMRouterTests(unittest.TestCase):
    def route(self, raw, q="What is my current city?"):
        return LLMRouter(ScriptedLLM([raw]), today=date(2026, 11, 21)).route(q)

    def test_valid_as_of(self):
        p = self.route('{"query_type":"as_of","multi_hop":false,"at":"2025-03-31"}', "city in March 2025?")
        self.assertEqual((p.policy, p.policy_params["at"]), ("as_of", datetime(2025, 3, 31, 23, 59, 59)))

    def test_valid_count_window_and_multihop(self):
        p = self.route('{"query_type":"count","multi_hop":true,"start":"2025-01-01","end":"2025-12-31"}')
        self.assertEqual(p.policy, "count_distinct")
        self.assertEqual(p.policy_params["start"], datetime(2025, 1, 1))
        self.assertTrue(p.multi_hop)

    def test_today_in_prompt(self):
        llm = ScriptedLLM(['{"query_type":"current","multi_hop":false}'])
        LLMRouter(llm, today=date(2026, 11, 21)).route("q")
        self.assertIn("2026-11-21", llm.prompts[0])

    def test_invalid_outputs_fall_back_to_rules(self):
        bad = ["not json", '{"query_type":"banana"}', '{"query_type":"as_of"}',
               '{"query_type":"current","multi_hop":"yes"}',
               '{"query_type":"count","start":"31/12/2025"}']
        for raw in bad:
            p = self.route(raw, "What was my previous employer?")
            self.assertEqual(p.policy, "previous", raw)           # rule router's answer
            self.assertTrue(any("used rules" in r for r in p.reasons), raw)


class LongMemEval(unittest.TestCase):
    def item(self, qid="q1", qtype="knowledge-update"):
        return {
            "question_id": qid, "question_type": qtype, "question": "How long is my commute?",
            "answer": "45 minutes", "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/28 (Sun) 18:05"],
            "haystack_sessions": [
                [{"role": "user", "content": "My commute takes 30 minutes."},
                 {"role": "assistant", "content": "Noted."}],
                [{"role": "user", "content": "My commute is now 45 minutes."},
                 {"role": "user", "content": "  "}],
            ],
        }

    def test_date_parse(self):
        self.assertEqual(parse_lme_date("2023/05/20 (Sat) 02:21"), datetime(2023, 5, 20, 2, 21))
        self.assertIsNone(parse_lme_date("garbage"))

    def test_load_filters_and_builds_store(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lme.json"
            p.write_text(json.dumps([self.item("a"), self.item("b", "single-session-user")]))
            ds = load_longmemeval(p)
            self.assertEqual(len(ds), 1)
            recs = ds[0].store.all()
            self.assertEqual([r.fact_id for r in recs], ["s1:0", "s2:0"])   # user turns only, blanks skipped
            self.assertLess(recs[0].timestamp, recs[1].timestamp)
            self.assertEqual(ds[0].questions[0].gold, ["45 minutes"])
            self.assertEqual(len(load_longmemeval(p, qtypes=())), 2)

    def test_end_to_end_with_stub_llm(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lme.json"
            p.write_text(json.dumps([self.item()]))
            ds = load_longmemeval(p)
        def respond(prompt):
            if "<ROUTER>" in prompt or "<TASK>" in prompt:
                raise AssertionError("unexpected prompt")
            return json.dumps({"candidates": [
                {"fact_id": "s1:0", "answer_entity": "30 minutes"},
                {"fact_id": "s2:0", "answer_entity": "45 minutes"}]})
        res = evaluate(ds, lambda _: ScriptedLLM(respond), ["fixed_bm25_latest"])
        self.assertEqual(res["summary"]["systems"]["fixed_bm25_latest"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
