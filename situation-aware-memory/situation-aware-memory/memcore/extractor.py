"""Extractor E: retrieved evidence -> structured candidates K.

Adapted from the base paper's `CANDIDATE_PROMPT` (cvikasreddy/
memory-conflict-resolution, scripts/_pipeline.py). Same contract: strict
subject + predicate matching, include ALL matching versions, never pick a
winner. Two deliberate differences:

1. The LLM returns only `fact_id` and `answer_entity`. Serial / timestamp /
   source / context are copied from the memory store, so the model cannot
   mis-copy or invent version metadata (the paper trusts the LLM's copy).
2. Any fact_id the model returns that was not in the retrieved set is
   dropped and counted, so hallucinated evidence cannot reach a policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLMClient
from .retrieval import Hit
from .schema import Candidate

CANDIDATE_PROMPT = """You are given retrieved items from a memory store. Each item starts with its ID.

Your job: identify EVERY item that DIRECTLY answers the question, and extract the answer entity from each.

Do NOT decide which item is newest or best. Do NOT use outside knowledge. Include ALL items that match.

Rules:
1. An item directly answers the question only if BOTH its subject AND its predicate exactly match what the question asks about. The predicate noun used in the question (e.g., "spouse", "city", "profession") must be the same noun used in the item. Related-but-different predicates do NOT match.
2. The subject named in the question must appear in the matching item. A different entity with a similar name does NOT match.
3. If a subject has multiple conflicting values (the same subject with different values in different items), INCLUDE ALL of them as separate candidates. Do not pick.
4. If no item answers the question, return an empty list.

Question: {question}

Items:
{pool}

Return ONLY valid JSON: {{"candidates": [{{"fact_id": "<ID of the item>", "answer_entity": "<extracted answer>", "confidence": <0.0-1.0>}}]}}"""


@dataclass(frozen=True)
class ExtractionResult:
    candidates: list[Candidate]
    parse_ok: bool = True
    dropped_unknown_ids: int = 0
    raw: str = ""
    notes: list[str] = field(default_factory=list)


class Extractor:
    def __init__(self, llm: LLMClient, prompt: str = CANDIDATE_PROMPT):
        self.llm = llm
        self.prompt = prompt

    @staticmethod
    def format_pool(hits: list[Hit]) -> str:
        return "\n".join(f"[{h.record.fact_id}] {h.record.text}" for h in hits)

    def extract(self, question: str, hits: list[Hit], found_ids: tuple[str, ...] = ()) -> ExtractionResult:
        if not hits:
            return ExtractionResult([], notes=["no retrieved evidence"])
        by_id = {h.record.fact_id: h.record for h in hits}
        prompt = self.prompt.format(question=question, pool=self.format_pool(hits),
                                    found=", ".join(found_ids) or "none")
        raw = self.llm.complete_json(prompt)
        try:
            items = json.loads(raw).get("candidates", [])
            if not isinstance(items, list):
                raise ValueError("candidates is not a list")
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            return ExtractionResult([], parse_ok=False, raw=raw, notes=[f"parse error: {e}"])

        out: list[Candidate] = []
        seen: set[str] = set()
        dropped = 0
        for it in items:
            if not isinstance(it, dict):
                dropped += 1
                continue
            fid = str(it.get("fact_id", "")).strip().strip("[]")
            rec = by_id.get(fid)
            value = it.get("answer_entity")
            if rec is None or not isinstance(value, str) or not value.strip():
                dropped += 1
                continue
            if fid in seen:
                continue
            seen.add(fid)
            try:
                conf = min(1.0, max(0.0, float(it.get("confidence", 1.0))))
            except (TypeError, ValueError):
                conf = 1.0
            out.append(Candidate(
                fact_id=rec.fact_id, text=rec.text, value=value.strip(),
                serial=rec.serial, timestamp=rec.timestamp, source=rec.source,
                context=dict(rec.context), confidence=conf,
            ))
        return ExtractionResult(out, True, dropped, raw)
