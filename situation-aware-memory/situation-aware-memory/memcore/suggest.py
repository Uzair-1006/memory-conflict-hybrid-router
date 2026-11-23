"""Goal-aware, explainable suggestions.

1. Plan:   the LLM lists the memory questions a good suggestion needs.
2. Resolve: each question goes through the Agent (router -> retrieval -> extraction -> policy),
            so the facts used are CURRENT / correctly versioned, with evidence ids.
3. Write:  the LLM writes a short suggestion from ONLY those resolved facts and must list
            every fact it relies on as a claim with supporting evidence ids.
4. Check:  code (not an LLM) verifies every cited id is real evidence. A suggestion whose claims
            are not all supported is flagged `grounded=False`.
If something needed cannot be resolved, the result carries clarifying questions instead of a guess.
`NaiveSuggester` is the baseline: raw top-k retrieval, no resolution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from .agent import Agent
from .llm import LLMClient
from .retrieval import Retriever

PLAN_PROMPT = """<PLAN>
A user has the goal below. List up to 3 questions about the user's stored memory whose answers are needed to give a useful, personal suggestion. Each question must ask for ONE current fact, in the form "What is my current <thing>?" or "Who is my <relation>?". Only ask about things that appear in the memory sample; never ask about something that is not there.
</PLAN>

Goal: {goal}

Memory sample:
{preview}

Return ONLY valid JSON: {{"questions": ["...", "..."]}}"""

SUGGEST_PROMPT = """<SUGGEST>
Write a short, practical suggestion (2-4 sentences) that helps the user make progress on their goal, using ONLY the facts listed below. Do not use any other information about the user. Keep it low-risk and general; do not give medical, legal or financial instructions. Every fact you rely on must appear as a claim with the IDs of the memory items that support it. If the facts are not enough for a useful suggestion, leave "suggestion" empty and put ONE question to ask the user in "clarification".
</SUGGEST>

Goal: {goal}
Today: {today}

Resolved facts:
{facts}

Not found in memory:
{unresolved}

Return ONLY valid JSON: {{"suggestion": "...", "claims": [{{"text": "...", "evidence": ["<id>"]}}], "clarification": null}}"""


@dataclass(frozen=True)
class Claim:
    text: str
    evidence: list[str]


@dataclass(frozen=True)
class Grounding:
    n_claims: int
    n_supported: int
    invalid_ids: list[str]

    @property
    def score(self) -> float:
        return self.n_supported / self.n_claims if self.n_claims else 0.0


@dataclass(frozen=True)
class ResolvedFact:
    question: str
    answer: object
    evidence_ids: list[str]


@dataclass(frozen=True)
class Suggestion:
    goal: str
    text: str
    claims: list[Claim] = field(default_factory=list)
    facts: list[ResolvedFact] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    grounding: Grounding = Grounding(0, 0, [])
    clarification: Optional[str] = None
    parse_ok: bool = True

    @property
    def grounded(self) -> bool:
        return bool(self.text) and self.parse_ok and self.grounding.n_claims > 0 \
            and self.grounding.score == 1.0 and not self.grounding.invalid_ids

    @property
    def cited_ids(self) -> list[str]:
        return sorted({i for c in self.claims for i in c.evidence})


def check_grounding(claims: list[Claim], allowed_ids: set[str]) -> Grounding:
    invalid = sorted({i for c in claims for i in c.evidence if i not in allowed_ids})
    supported = sum(1 for c in claims if c.evidence and all(i in allowed_ids for i in c.evidence))
    return Grounding(len(claims), supported, invalid)


def _parse(raw: str) -> tuple[str, list[Claim], Optional[str]] | None:
    try:
        d = json.loads(raw)
        text = d.get("suggestion") or ""
        claims = [Claim(str(c["text"]), [str(i) for i in c.get("evidence", [])]) for c in d.get("claims", [])]
        clar = d.get("clarification")
        return str(text).strip(), claims, (str(clar).strip() if clar else None)
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None


class SuggestionEngine:
    def __init__(self, agent: Agent, llm: LLMClient, preview_size: int = 25):
        self.agent, self.llm, self.preview_size = agent, llm, preview_size

    def _preview(self, goal: str) -> str:
        """The stored memories most relevant to the goal (NOT the most recent ones: an old fact
        that is still current, e.g. a city stated years ago, must stay visible to the planner)."""
        hits = self.agent.retrievers["hybrid"].search(goal, self.preview_size)
        return "\n".join(f"- {h.record.text}" for h in hits) or "(empty)"

    def plan(self, goal: str) -> list[str]:
        raw = self.llm.complete_json(PLAN_PROMPT.format(goal=goal, preview=self._preview(goal)))
        try:
            qs = json.loads(raw)["questions"]
            return [q.strip() for q in qs if isinstance(q, str) and q.strip()][:3]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return []

    def suggest(self, goal: str, today: date | None = None) -> Suggestion:
        questions = self.plan(goal)
        facts: list[ResolvedFact] = []
        unresolved: list[str] = []
        for q in questions:
            r = self.agent.answer(q)
            if r.abstained:
                unresolved.append(q)
            else:
                facts.append(ResolvedFact(q, r.answer, list(r.evidence)))
        if not facts:
            note = "I could not find anything in memory that helps with this goal."
            if unresolved:
                note += " I could not answer: " + "; ".join(unresolved) + "."
            return Suggestion(goal, "", facts=facts, unresolved=unresolved,
                              clarification=note + " What should I take into account?")
        lines, allowed = [], set()
        for f in facts:
            lines.append(f"Q: {f.question} A: {f.answer}")
            for i in f.evidence_ids:
                rec = self.agent.store.get(i)
                if rec:
                    allowed.add(i)
                    lines.append(f"[{i}] {rec.text}")
        raw = self.llm.complete_json(SUGGEST_PROMPT.format(
            goal=goal, today=(today or date.today()).isoformat(), facts="\n".join(lines),
            unresolved="\n".join(f"- {u}" for u in unresolved) or "(none)"))
        parsed = _parse(raw)
        if parsed is None:
            return Suggestion(goal, "", facts=facts, unresolved=unresolved, parse_ok=False)
        text, claims, clar = parsed
        return Suggestion(goal, text, claims, facts, unresolved, check_grounding(claims, allowed), clar)


class NaiveSuggester:
    """Baseline: raw top-k retrieval of the goal text, no resolution of versions or conflicts."""

    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int = 10):
        self.retriever, self.llm, self.top_k = retriever, llm, top_k

    def suggest(self, goal: str, today: date | None = None) -> Suggestion:
        hits = self.retriever.search(goal, self.top_k)
        allowed = {h.record.fact_id for h in hits}
        lines = [f"[{h.record.fact_id}] {h.record.text}" for h in hits]
        raw = self.llm.complete_json(SUGGEST_PROMPT.format(
            goal=goal, today=(today or date.today()).isoformat(), facts="\n".join(lines), unresolved="(none)"))
        parsed = _parse(raw)
        if parsed is None:
            return Suggestion(goal, "", parse_ok=False)
        text, claims, clar = parsed
        return Suggestion(goal, text, claims, [], [], check_grounding(claims, allowed), clar)
