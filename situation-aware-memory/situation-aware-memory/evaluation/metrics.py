"""Metrics: answer matching, Wilson interval, exact McNemar test."""
from __future__ import annotations

import math
import re


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).casefold()).split())


def match(pred, golds: list[str]) -> bool:
    """SubEM (gold is a substring of the prediction), as in the paper.
    Numeric golds (counts) need an exact match, since '1' is a substring of '10'."""
    if pred is None:
        return False
    p = _norm(pred)
    for g in golds:
        gn = _norm(g)
        if gn.isdigit():
            if p == gn:
                return True
        elif gn and gn in p:
            return True
    return False


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Paired exact McNemar test. b01 = only A right, b10 = only B right."""
    assert len(a_correct) == len(b_correct)
    only_a = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    only_b = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_a": only_a, "only_b": only_b, "p": p}
