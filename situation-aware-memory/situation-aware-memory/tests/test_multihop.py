import json
import re
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcore.extractor import Extractor
from memcore.llm import ScriptedLLM
from memcore.memory import MemoryRecord, MemoryStore
from memcore.multihop import MultiHopPipeline
from memcore.retrieval import BM25Retriever


def store():
    return MemoryStore([
        MemoryRecord("m1", "My manager is Priya.", timestamp=datetime(2024, 1, 1)),
        MemoryRecord("m2", "My manager is Rahul.", timestamp=datetime(2025, 1, 1)),   # newest, but no city known
        MemoryRecord("c1", "Priya's city is Pune.", timestamp=datetime(2024, 6, 1)),
        MemoryRecord("n1", "I watched a film on Friday.", timestamp=datetime(2025, 2, 1)),
    ])


def llm():
    def respond(prompt):
        if "<TASK>" in prompt:
            return json.dumps({"hops": [{"id": 1, "query": "Who is my manager?"},
                                        {"id": 2, "query": "What is the city of {hop_1_answer}?"}]})
        q = re.search(r"Question: (.*)\n", prompt).group(1)
        out = []
        for line in prompt.split("Items:\n")[1].splitlines():
            if not line.startswith("["):
                continue
            fid = line[1:line.index("]")]
            text = line[line.index("]") + 2:]
            if "my manager" in q and text.startswith("My manager is"):
                out.append({"fact_id": fid, "answer_entity": text.rstrip(".").split(" is ")[-1]})
            m = re.search(r"city of (\w+)\?", q)
            if m and text.startswith(f"{m.group(1)}'s city"):
                out.append({"fact_id": fid, "answer_entity": text.rstrip(".").split(" is ")[-1]})
        return json.dumps({"candidates": out})
    return ScriptedLLM(respond)


class Backtracking(unittest.TestCase):
    def pipe(self, beam):
        l = llm()
        return MultiHopPipeline(BM25Retriever(store()), Extractor(l), l, beam=beam)

    def test_backtracks_to_older_manager(self):
        r = self.pipe(beam=2).run("What is the city of my manager?")
        self.assertFalse(r.abstained)
        self.assertEqual(r.answer, "Pune")
        self.assertTrue(r.backtracked)
        self.assertEqual(r.evidence, ["c1"])

    def test_beam_one_abstains_instead_of_guessing(self):
        r = self.pipe(beam=1).run("What is the city of my manager?")
        self.assertTrue(r.abstained)
        self.assertIsNone(r.answer)

    def test_bad_decomposition_abstains(self):
        l = ScriptedLLM(["not json"])
        r = MultiHopPipeline(BM25Retriever(store()), Extractor(l), l).run("q")
        self.assertTrue(r.abstained)
        self.assertEqual(r.reason, "decomposition failed")


if __name__ == "__main__":
    unittest.main()
