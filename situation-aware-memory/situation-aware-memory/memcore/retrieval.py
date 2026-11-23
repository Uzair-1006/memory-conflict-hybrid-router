"""Lexical retrieval: Okapi BM25 in pure Python (no external dependency).

Matches the paper's setup (lowercase alphanumeric tokenizer, top-10) and the
`rank_bm25.BM25Okapi` defaults (k1=1.5, b=0.75, epsilon=0.25 IDF floor), so
results should line up with the baseline. Dense and hybrid (RRF) retrieval
will be added behind the same `Retriever` interface.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .memory import MemoryRecord, MemoryStore

TOP_K = 10


def tokenize(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    record: MemoryRecord


class Retriever(Protocol):
    name: str

    def search(self, query: str, top_k: int = TOP_K) -> list[Hit]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: MemoryStore, k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25):
        self.k1, self.b = k1, b
        self._records = store.all()
        self._docs = [Counter(tokenize(r.text)) for r in self._records]
        self._lens = [sum(d.values()) for d in self._docs]
        n = len(self._docs)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        idf: dict[str, float] = {}
        negatives: list[str] = []
        total = 0.0
        for term, f in df.items():
            v = math.log((n - f + 0.5) / (f + 0.5))
            idf[term] = v
            total += v
            if v < 0:
                negatives.append(term)
        floor = epsilon * (total / len(df)) if df else 0.0
        for term in negatives:
            idf[term] = floor
        self._idf = idf

    def _score(self, q_tokens: list[str], i: int) -> float:
        doc, dl = self._docs[i], self._lens[i]
        norm = self.k1 * (1 - self.b + self.b * dl / self._avgdl) if self._avgdl else self.k1
        s = 0.0
        for t in q_tokens:
            f = doc.get(t, 0)
            if f:
                s += self._idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: str, top_k: int = TOP_K) -> list[Hit]:
        q = tokenize(query)
        scored = [(self._score(q, i), i) for i in range(len(self._records))]
        # Stable: ties keep insertion order.
        scored.sort(key=lambda x: -x[0])
        return [Hit(rank=r + 1, score=s, record=self._records[i])
                for r, (s, i) in enumerate(scored[:top_k])]
