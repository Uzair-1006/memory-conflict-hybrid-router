"""Direct-answer baseline: ONE LLM call does everything.

This is the comparison system from the base paper's controlled experiment:
the model must find matching facts, reject distractors, ignore its own
pretrained knowledge, apply the freshness rule and write the answer, all at
once. We use it to measure how much extract-then-execute actually adds when
the retrieved evidence is identical.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLMClient
from .retrieval import TOP_K, Hit, Retriever

DIRECT_PROMPT = """Answer the question using ONLY the retrieved items below. Do not use outside knowledge.

Each item is tagged with its version. If several items give different values for the same subject and relationship, the item with the NEWEST version is the correct one. Ignore items about other subjects or other relationships.

Question: {question}

Items:
{pool}

Return ONLY valid JSON: {{"answer": "<short answer>"}}. If no item answers the question, use {{"answer": null}}."""


def _tag(h: Hit) -> str:
    r = h.record
    if r.serial is not None:
        return f"version {r.serial}"
    if r.timestamp is not None:
        return f"date {r.timestamp.date().isoformat()}"
    return "version unknown"


@dataclass(frozen=True)
class DirectResult:
    question: str
    answer: object
    abstained: bool
    hits: list[Hit]
    parse_ok: bool = True


class DirectAnswerer:
    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int = TOP_K):
        self.retriever, self.llm, self.top_k = retriever, llm, top_k

    def run(self, question: str) -> DirectResult:
        hits = self.retriever.search(question, self.top_k)
        if not hits:
            return DirectResult(question, None, True, [])
        pool = "\n".join(f"[{h.record.fact_id} | {_tag(h)}] {h.record.text}" for h in hits)
        raw = self.llm.complete_json(DIRECT_PROMPT.format(question=question, pool=pool))
        try:
            ans = json.loads(raw).get("answer")
        except (json.JSONDecodeError, AttributeError):
            return DirectResult(question, None, True, hits, parse_ok=False)
        if ans is None or (isinstance(ans, str) and not ans.strip()):
            return DirectResult(question, None, True, hits)
        return DirectResult(question, ans if not isinstance(ans, str) else ans.strip(), False, hits)
