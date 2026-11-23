import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcore.embeddings import DenseRetriever, HashingEmbedder
from memcore.hybrid import HybridRetriever, TimeWindowRetriever, rrf_fuse
from memcore.memory import MemoryRecord, MemoryStore
from memcore.retrieval import BM25Retriever, Hit


def store():
    return MemoryStore([
        MemoryRecord("a", "My city of residence is Pune.", timestamp=datetime(2024, 1, 1)),
        MemoryRecord("b", "My city of residence is Chennai.", timestamp=datetime(2025, 6, 1)),
        MemoryRecord("c", "I enjoy watching cricket on weekends.", timestamp=datetime(2025, 7, 1)),
        MemoryRecord("d", "My employer is Zoho.", timestamp=datetime(2025, 8, 1)),
    ])


class Dense(unittest.TestCase):
    def test_deterministic_and_relevant(self):
        e = HashingEmbedder()
        self.assertEqual(e.embed(["hello world"]), e.embed(["hello world"]))
        hits = DenseRetriever(store(), e).search("city of residence", 2)
        self.assertEqual({h.record.fact_id for h in hits}, {"a", "b"})

    def test_unit_norm(self):
        v = HashingEmbedder().embed(["some text here"])[0]
        self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=6)


class RRF(unittest.TestCase):
    def test_fusion_prefers_agreement(self):
        s = store()
        r1 = [Hit(1, 1, s.get("a")), Hit(2, 1, s.get("b")), Hit(3, 1, s.get("c"))]
        r2 = [Hit(1, 1, s.get("b")), Hit(2, 1, s.get("d")), Hit(3, 1, s.get("a"))]
        fused = rrf_fuse([r1, r2], top_k=4)
        self.assertEqual(fused[0].record.fact_id, "b")       # rank 2 + rank 1
        self.assertEqual({h.record.fact_id for h in fused}, {"a", "b", "c", "d"})
        self.assertEqual([h.rank for h in fused], [1, 2, 3, 4])

    def test_hybrid_runs(self):
        s = store()
        h = HybridRetriever([BM25Retriever(s), DenseRetriever(s, HashingEmbedder())])
        top = h.search("city of residence", 2)
        self.assertEqual({x.record.fact_id for x in top}, {"a", "b"})


class TimeWindow(unittest.TestCase):
    def test_filters_by_window(self):
        s = store()
        t = TimeWindowRetriever(BM25Retriever(s), start=datetime(2025, 1, 1), end=datetime(2025, 12, 31))
        ids = {h.record.fact_id for h in t.search("city of residence", 4)}
        self.assertNotIn("a", ids)
        self.assertIn("b", ids)

    def test_drops_records_without_timestamp(self):
        s = MemoryStore([MemoryRecord("x", "My city is Pune.")])
        self.assertEqual(TimeWindowRetriever(BM25Retriever(s), start=datetime(2000, 1, 1)).search("city"), [])


if __name__ == "__main__":
    unittest.main()
