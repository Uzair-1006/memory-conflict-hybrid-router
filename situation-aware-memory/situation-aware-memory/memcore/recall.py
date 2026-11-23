"""Recall-boosting extraction.

Motivation (measured on FactConsolidation, real LLM): every remaining error of
the extract-then-execute pipeline was an EXTRACTION miss. The retrieved memory
contained the needed fact, but the extractor returned only the value that
matches real-world knowledge ("Japanese", "Paris", "Satya Nadella") and left
out the newer, counter-intuitive version. Policies and retrieval were never the
problem, so we raise extraction recall, two ways that can be combined:

sibling  : deterministic, no LLM call. For each extracted candidate, any OTHER
           retrieved memory with the same text except for the value (same
           prefix and suffix around the value) is the same subject and the same
           relationship, so it is added as a candidate. Exact for templated
           memories; finds nothing on free-form text.
recall   : a second LLM call that lists any items the first pass missed and is
           told that memory may contain updated or surprising facts that are
           still valid. +1 LLM call per question.
both     : sibling, then recall.
"""
from __future__ import annotations

from .extractor import ExtractionResult, Extractor
from .llm import LLMClient
from .retrieval import Hit
from .schema import Candidate

RECALL_PROMPT = """<RECALL>
A first pass already extracted some items that answer the question. Find any OTHER items in the list that also state the same kind of fact about the same subject, including items whose value is different, newer, surprising, or contradicts common knowledge. The memory may contain deliberate updates and corrections; they are valid and must be reported. Do NOT use outside knowledge to reject an item. Do NOT decide which item is newest or best.

Already found item IDs: {found}
</RECALL>

Question: {question}

Items:
{pool}

Return ONLY valid JSON: {{"candidates": [{{"fact_id": "<ID of an item NOT already found>", "answer_entity": "<extracted answer>", "confidence": <0.0-1.0>}}]}}. If nothing was missed, return {{"candidates": []}}."""

MODES = ("sibling", "recall", "both")


def sibling_candidates(found: list[Candidate], hits: list[Hit]) -> list[Candidate]:
    """Other retrieved memories that differ from an extracted one ONLY in the value."""
    have = {c.fact_id for c in found}
    out: list[Candidate] = []
    for c in found:
        i = c.text.lower().find(c.value.lower())
        if i < 0:
            continue
        prefix, suffix = c.text[:i].lower(), c.text[i + len(c.value):].lower()
        if len(prefix) + len(suffix) < 8:      # template too short to be meaningful
            continue
        for h in hits:
            r = h.record
            if r.fact_id in have:
                continue
            t = r.text.lower()
            if len(t) > len(prefix) + len(suffix) and t.startswith(prefix) and t.endswith(suffix):
                value = r.text[len(prefix): len(r.text) - len(suffix)].strip()
                if value and len(value.split()) <= 8:
                    have.add(r.fact_id)
                    out.append(Candidate(
                        fact_id=r.fact_id, text=r.text, value=value, serial=r.serial,
                        timestamp=r.timestamp, source=r.source,
                        context={**r.context, "expanded": "sibling"}, confidence=0.6))
    return out


class RecallBoostExtractor:
    """Drop-in replacement for `Extractor` (same `extract` signature)."""

    def __init__(self, llm: LLMClient, mode: str = "both", base: Extractor | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.base = base or Extractor(llm)
        self.recall = Extractor(llm, prompt=RECALL_PROMPT)

    def extract(self, question: str, hits: list[Hit], found_ids: tuple[str, ...] = ()) -> ExtractionResult:
        first = self.base.extract(question, hits)
        cands = list(first.candidates)
        notes = list(first.notes)
        dropped, parse_ok = first.dropped_unknown_ids, first.parse_ok

        if self.mode in ("sibling", "both"):
            extra = sibling_candidates(cands, hits)
            if extra:
                cands += extra
                notes.append(f"sibling: +{len(extra)}")

        if self.mode in ("recall", "both") and hits:
            have = {c.fact_id for c in cands}
            if len(have) < len(hits):                    # something is still unextracted
                second = self.recall.extract(question, hits, tuple(sorted(have)))
                parse_ok = parse_ok and second.parse_ok
                dropped += second.dropped_unknown_ids
                new = [c for c in second.candidates if c.fact_id not in have]
                if new:
                    cands += new
                    notes.append(f"recall: +{len(new)}")
        return ExtractionResult(cands, parse_ok, dropped, first.raw, notes)
