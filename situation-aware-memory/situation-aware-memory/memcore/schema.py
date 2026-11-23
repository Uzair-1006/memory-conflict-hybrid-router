"""Structured candidate representation (the paper's set K).

The base paper's candidates carry only (serial, fact_text, answer_entity).
We add timestamp, source and context so that historical, as-of and
aggregation policies can run on the same representation, and so that every
answer can be traced back to its evidence (provenance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class Candidate:
    """One version of one fact, as extracted from retrieved memory."""

    fact_id: str                       # stable id of the memory record
    text: str                          # verbatim memory text (evidence)
    value: str                         # the extracted answer entity
    serial: Optional[int] = None       # explicit version number (paper's setting)
    timestamp: Optional[datetime] = None  # real-world time (personal memory setting)
    source: str = ""                   # provenance: conversation / document / date
    context: dict[str, Any] = field(default_factory=dict)  # e.g. {"situation": "work"}
    confidence: float = 1.0            # extractor's confidence in the match, 0..1

    def norm_value(self) -> str:
        return " ".join(self.value.casefold().split())


class OrderingError(ValueError):
    """Candidates cannot be put in a total order (missing or mixed metadata)."""


def order_key(c: Candidate, mode: str) -> Any:
    if mode == "serial":
        return c.serial
    return c.timestamp


def ordering_mode(candidates: list[Candidate]) -> str:
    """Pick the ordering metadata shared by ALL candidates.

    Serials win when present on every candidate (matches the paper);
    otherwise timestamps; otherwise the order is undefined.
    """
    if candidates and all(c.serial is not None for c in candidates):
        return "serial"
    if candidates and all(c.timestamp is not None for c in candidates):
        return "timestamp"
    raise OrderingError(
        "candidates lack a shared ordering key (need serial on all, or timestamp on all)"
    )
