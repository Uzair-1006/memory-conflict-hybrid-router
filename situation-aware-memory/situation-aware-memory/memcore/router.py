"""Situation-aware router: question -> RoutePlan(retriever, policy, multi-hop).

`RuleRouter` is deterministic and needs no API key. It is the transparent
baseline; an LLM router can implement the same `route()` interface later and
be compared against it.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))


@dataclass(frozen=True)
class RoutePlan:
    query_type: str                 # current | previous | as_of | count
    retriever: str                  # bm25 | dense | hybrid
    policy: str                     # key in policies.POLICY_FACTORIES
    policy_params: dict[str, Any] = field(default_factory=dict)
    multi_hop: bool = False
    reasons: tuple[str, ...] = ()


def _end_of_day(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, 23, 59, 59)


def find_period(text: str) -> Optional[tuple[datetime, datetime]]:
    """First date expression in `text` as an inclusive (start, end) period."""
    t = text.lower()
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d), _end_of_day(y, mo, d)
    m = re.search(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", t)
    if m:
        mo, d, y = MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d), _end_of_day(y, mo, d)
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON})\.?,?\s+(\d{{4}})\b", t)
    if m:
        d, mo, y = int(m.group(1)), MONTHS[m.group(2)], int(m.group(3))
        return datetime(y, mo, d), _end_of_day(y, mo, d)
    m = re.search(rf"\b({_MON})\.?,?\s+(\d{{4}})\b", t)
    if m:
        mo, y = MONTHS[m.group(1)], int(m.group(2))
        return datetime(y, mo, 1), _end_of_day(y, mo, calendar.monthrange(y, mo)[1])
    m = re.search(r"\b(19\d{2}|20\d{2})\b", t)
    if m:
        y = int(m.group(1))
        return datetime(y, 1, 1), _end_of_day(y, 12, 31)
    return None


_COUNT = re.compile(r"\bhow many\b|\bnumber of (?:different|distinct|times)\b")
_BEFORE = re.compile(r"\bbefore\b|\bprior to\b|\bearlier than\b")
_PREVIOUS = re.compile(r"\bprevious(?:ly)?\b|\bused to\b|\bformerly\b|\bprior\b|\bbefore that\b|\bold\b")
# Multi-hop cues that stay quiet on titles/names ("author of The Marriage of Figaro",
# "head of the Tucson government"): a chain through the user ("of my manager",
# "my manager's city"), or a lower-case relation chain ("of the spouse of").
_MULTIHOP = re.compile(r"\bof (?:my|his|her|their) \w+|\b(?:my|his|her|their) \w+'s \w+")
_CHAIN_OF = re.compile(r"\bof the \w+ of\b")
_POSSESSIVE = re.compile(r"\w+'s \w+")
_NAMED = re.compile(r"(?<!^)\b[A-Z][a-z]+\b|\d")
_DATEISH = re.compile(rf"\b(?:{_MON})\b\.?|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{4}}\b|\b\d{{1,2}}(?:st|nd|rd|th)\b", re.I)


class RuleRouter:
    def route(self, question: str) -> RoutePlan:
        q = question.lower()
        period = find_period(question)
        reasons: list[str] = []

        multi_hop = (bool(_MULTIHOP.search(q)) or len(_POSSESSIVE.findall(q)) >= 2
                     or bool(_CHAIN_OF.search(question)))
        if multi_hop:
            reasons.append("relation chain ('of my X' / possessive) -> multi-hop")

        if _COUNT.search(q):
            qtype, policy, params = "count", "count_distinct", {}
            reasons.append("'how many' -> aggregation")
            if period:
                params = {"start": period[0], "end": period[1]}
                reasons.append("date window found -> windowed count")
        elif period and _BEFORE.search(q):
            qtype, policy = "as_of", "as_of"
            params = {"at": period[0] - timedelta(seconds=1)}
            reasons.append("'before <date>' -> as_of just before the period starts")
        elif period:
            qtype, policy = "as_of", "as_of"
            params = {"at": period[1]}
            reasons.append("date found -> as_of end of the period")
        elif _PREVIOUS.search(q):
            qtype, policy, params = "previous", "previous", {"n": 1}
            reasons.append("'previous/used to' -> previous value")
        else:
            qtype, policy, params = "current", "latest", {}
            reasons.append("no temporal cue -> latest value")

        # Lexical retrieval for exact-token questions (names, numbers), hybrid otherwise.
        exact = bool(_NAMED.search(_DATEISH.sub(" ", question).strip()))
        retriever = "bm25" if exact else "hybrid"
        reasons.append(f"{'proper-noun/number cues' if exact else 'no exact-token cues'} -> {retriever}")
        return RoutePlan(qtype, retriever, policy, params, multi_hop, tuple(reasons))
