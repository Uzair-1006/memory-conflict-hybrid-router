"""Hybrid retrieval: Reciprocal Rank Fusion + a temporal filter.

RRF score of a document = sum over rankings of 1 / (k + rank). It needs only
ranks, so lexical (BM25) and dense scores never have to be put on one scale.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .retrieval import TOP_K, Hit, Retriever


def rrf_fuse(rankings: list[list[Hit]], k: int = 60, top_k: int = TOP_K) -> list[Hit]:
    score: dict[str, float] = {}
    rec = {}
    for ranking in rankings:
        for h in ranking:
            fid = h.record.fact_id
            score[fid] = score.get(fid, 0.0) + 1.0 / (k + h.rank)
            rec[fid] = h.record
    order = sorted(score, key=lambda f: -score[f])[:top_k]
    return [Hit(rank=i + 1, score=score[f], record=rec[f]) for i, f in enumerate(order)]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], pool: int = 50, k: int = 60):
        self.retrievers, self.pool, self.k = retrievers, pool, k

    def search(self, query: str, top_k: int = TOP_K) -> list[Hit]:
        rankings = [r.search(query, self.pool) for r in self.retrievers]
        return rrf_fuse(rankings, self.k, top_k)


class TimeWindowRetriever:
    """Keeps only memories whose timestamp lies in [start, end]. Over-fetches
    from the base retriever so the filter does not starve the result list.
    Records without a timestamp are dropped (they cannot be placed in time)."""

    name = "temporal"

    def __init__(self, base: Retriever, start: Optional[datetime] = None,
                 end: Optional[datetime] = None, overfetch: int = 5):
        self.base, self.start, self.end, self.overfetch = base, start, end, overfetch

    def search(self, query: str, top_k: int = TOP_K) -> list[Hit]:
        out: list[Hit] = []
        for h in self.base.search(query, top_k * self.overfetch):
            t = h.record.timestamp
            if t is None or (self.start and t < self.start) or (self.end and t > self.end):
                continue
            out.append(Hit(rank=len(out) + 1, score=h.score, record=h.record))
            if len(out) == top_k:
                break
        return out


class ContextFilterRetriever:
    """Keeps only memories whose `context` contains every required key/value
    (e.g. {"domain": "work"}); lets a question be answered inside one area of life."""

    name = "context"

    def __init__(self, base: Retriever, required: dict, overfetch: int = 5):
        self.base, self.required, self.overfetch = base, required, overfetch

    def search(self, query: str, top_k: int = TOP_K) -> list[Hit]:
        out: list[Hit] = []
        for h in self.base.search(query, top_k * self.overfetch):
            if all(str(h.record.context.get(k, "")).lower() == str(v).lower() for k, v in self.required.items()):
                out.append(Hit(rank=len(out) + 1, score=h.score, record=h.record))
                if len(out) == top_k:
                    break
        return out
