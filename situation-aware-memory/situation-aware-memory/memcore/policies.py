"""Policy operators: deterministic code over a candidate set K.

Each policy is the paper's A_pi. All of them take the same input
(list[Candidate]) and return the same output (PolicyResult), so a router can
swap them freely and every decision is logged, testable and explainable.

A policy NEVER guesses: if the evidence cannot support an answer (no
candidates, missing ordering metadata, an unresolved tie between different
values) it abstains with a reason, which the agent can turn into a
clarification question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Protocol

from .schema import Candidate, OrderingError, order_key, ordering_mode


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    answer: Any = None                       # str for value policies, int for count
    chosen: tuple[Candidate, ...] = ()       # evidence supporting the answer
    rejected: tuple[Candidate, ...] = ()     # candidates considered and not used
    abstained: bool = False
    reason: str = ""                         # why we abstained / how we decided
    confidence: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.abstained


class Policy(Protocol):
    name: str

    def __call__(self, candidates: list[Candidate]) -> PolicyResult: ...


def _abstain(name: str, reason: str, rejected: list[Candidate] | None = None) -> PolicyResult:
    return PolicyResult(policy=name, abstained=True, reason=reason,
                        rejected=tuple(rejected or ()))


def _sorted(candidates: list[Candidate]) -> tuple[list[Candidate], str]:
    mode = ordering_mode(candidates)
    return sorted(candidates, key=lambda c: order_key(c, mode)), mode


# ── Latest (reproduces the paper's max(serial), generalised to timestamps) ──
@dataclass(frozen=True)
class Latest:
    name: str = "latest"

    def __call__(self, candidates: list[Candidate]) -> PolicyResult:
        if not candidates:
            return _abstain(self.name, "no candidates")
        try:
            ordered, mode = _sorted(candidates)
        except OrderingError as e:
            return _abstain(self.name, str(e), candidates)
        top_key = order_key(ordered[-1], mode)
        top = [c for c in ordered if order_key(c, mode) == top_key]
        if len({c.norm_value() for c in top}) > 1:
            return _abstain(self.name,
                            f"tie at newest {mode}={top_key} with different values", candidates)
        chosen = top[-1]
        return PolicyResult(
            policy=self.name, answer=chosen.value, chosen=(chosen,),
            rejected=tuple(c for c in ordered if c not in top),
            reason=f"largest {mode}={top_key}", confidence=chosen.confidence,
        )


# ── Previous: the value before the current one (n=1) or n changes ago ──
@dataclass(frozen=True)
class Previous:
    n: int = 1
    name: str = "previous"

    def __call__(self, candidates: list[Candidate]) -> PolicyResult:
        if not candidates:
            return _abstain(self.name, "no candidates")
        try:
            ordered, mode = _sorted(candidates)
        except OrderingError as e:
            return _abstain(self.name, str(e), candidates)
        # Collapse repeated statements of the same value into one version.
        history: list[Candidate] = []
        for c in ordered:
            if not history or history[-1].norm_value() != c.norm_value():
                history.append(c)
        if len(history) <= self.n:
            return _abstain(self.name,
                            f"only {len(history)} distinct version(s); need {self.n + 1}",
                            candidates)
        chosen = history[-1 - self.n]
        return PolicyResult(
            policy=self.name, answer=chosen.value, chosen=(chosen,),
            rejected=tuple(c for c in ordered if c is not chosen),
            reason=f"{self.n} change(s) before the current value", confidence=chosen.confidence,
        )


# ── AsOf: the value that was true at time t ──
@dataclass(frozen=True)
class AsOf:
    at: datetime
    name: str = "as_of"

    def __call__(self, candidates: list[Candidate]) -> PolicyResult:
        if not candidates:
            return _abstain(self.name, "no candidates")
        if any(c.timestamp is None for c in candidates):
            return _abstain(self.name, "as_of needs a timestamp on every candidate", candidates)
        eligible = [c for c in candidates if c.timestamp <= self.at]
        if not eligible:
            return _abstain(self.name, f"no version existed at or before {self.at.isoformat()}",
                            candidates)
        inner = Latest()(eligible)
        if inner.abstained:
            return _abstain(self.name, inner.reason, candidates)
        later = tuple(c for c in candidates if c not in eligible)
        return PolicyResult(
            policy=self.name, answer=inner.answer, chosen=inner.chosen,
            rejected=inner.rejected + later,
            reason=f"newest version at or before {self.at.isoformat()}",
            confidence=inner.confidence,
        )


# ── CountDistinct: how many different values in a time window ──
@dataclass(frozen=True)
class CountDistinct:
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    name: str = "count_distinct"

    def __call__(self, candidates: list[Candidate]) -> PolicyResult:
        if not candidates:
            return _abstain(self.name, "no candidates")
        if (self.start or self.end) and any(c.timestamp is None for c in candidates):
            return _abstain(self.name, "windowed count needs timestamps", candidates)
        inside = [
            c for c in candidates
            if (self.start is None or c.timestamp >= self.start)
            and (self.end is None or c.timestamp <= self.end)
        ]
        values = {c.norm_value() for c in inside}
        return PolicyResult(
            policy=self.name, answer=len(values), chosen=tuple(inside),
            rejected=tuple(c for c in candidates if c not in inside),
            reason=f"{len(values)} distinct value(s) in window",
            confidence=min((c.confidence for c in inside), default=0.0),
        )


# ── Registry: lets a router select a policy by name ──
POLICY_FACTORIES: dict[str, Callable[..., Policy]] = {
    "latest": Latest,
    "previous": Previous,
    "as_of": AsOf,
    "count_distinct": CountDistinct,
}


def make_policy(name: str, **params: Any) -> Policy:
    try:
        return POLICY_FACTORIES[name](**params)
    except KeyError:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(POLICY_FACTORIES)}") from None
