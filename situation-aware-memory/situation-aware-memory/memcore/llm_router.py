"""LLM router: the language model classifies the question; code validates it.

The LLM only fills a small, closed schema (type, multi-hop flag, dates). It
never chooses code to run. Anything invalid or missing falls back to the
deterministic `RuleRouter`, so a bad model response cannot break the system.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Optional

from .llm import LLMClient
from .router import RoutePlan, RuleRouter

ROUTER_PROMPT = """<ROUTER>
Classify this question about a user's memory. Today's date: {today}.

query_type, exactly one of:
- "current": asks for the latest / present value
- "previous": asks for the value before the current one
- "as_of": asks what the value was at some past time
- "count": asks how many (different values, changes, times) there were
multi_hop: true only if the answer needs a chain (e.g. "the city of my manager": find the manager first, then the city).
at: for as_of only, the LAST day to consider as YYYY-MM-DD (for "in March 2025" use 2025-03-31; for "before 2025" use 2024-12-31). Otherwise null.
start, end: for count only, the window as YYYY-MM-DD, or null for all time.
</ROUTER>

<QUESTION>
{question}
</QUESTION>

Return ONLY valid JSON: {{"query_type": "...", "multi_hop": false, "at": null, "start": null, "end": null}}"""

_TYPES = {"current", "previous", "as_of", "count"}


def _day(s, end_of_day: bool) -> Optional[datetime]:
    d = date.fromisoformat(s)
    return datetime(d.year, d.month, d.day, 23, 59, 59) if end_of_day else datetime(d.year, d.month, d.day)


class LLMRouter:
    def __init__(self, llm: LLMClient, fallback: RuleRouter | None = None,
                 today: date | None = None):
        self.llm, self.fallback = llm, fallback or RuleRouter()
        self.today = today or date.today()

    def route(self, question: str) -> RoutePlan:
        rule = self.fallback.route(question)       # supplies retriever choice + fallback
        try:
            raw = self.llm.complete_json(
                ROUTER_PROMPT.format(question=question, today=self.today.isoformat()))
            d = json.loads(raw)
            qtype = d["query_type"]
            if qtype not in _TYPES:
                raise ValueError(f"bad query_type {qtype!r}")
            multi = d.get("multi_hop", False)
            if not isinstance(multi, bool):
                raise ValueError("multi_hop must be boolean")
            if qtype == "current":
                policy, params = "latest", {}
            elif qtype == "previous":
                policy, params = "previous", {"n": 1}
            elif qtype == "as_of":
                if not d.get("at"):
                    raise ValueError("as_of needs 'at'")
                policy, params = "as_of", {"at": _day(d["at"], True)}
            else:
                policy, params = "count_distinct", {}
                if d.get("start"):
                    params["start"] = _day(d["start"], False)
                if d.get("end"):
                    params["end"] = _day(d["end"], True)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as e:
            return RoutePlan(rule.query_type, rule.retriever, rule.policy, rule.policy_params,
                             rule.multi_hop, rule.reasons + (f"LLM router invalid ({e}); used rules",))
        return RoutePlan(qtype, rule.retriever, policy, params, multi,
                         (f"LLM router: {qtype}",) + rule.reasons[-1:])
