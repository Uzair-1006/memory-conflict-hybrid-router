"""LongMemEval loader (knowledge-update subset by default).

Each LongMemEval question carries its OWN chat history, so every question
becomes its own one-question `Dataset` with its own `MemoryStore`.

ASSUMED schema (LongMemEval's published format; not yet checked against the
real download, so verify on first run): a JSON list of objects with
question_id, question_type, question, answer, question_date,
haystack_session_ids, haystack_dates (e.g. "2023/05/20 (Sat) 02:21"), and
haystack_sessions = list of sessions, each a list of {"role", "content"}.
Download (needs network): https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from memcore.memory import MemoryRecord, MemoryStore

from .datasets import Dataset, Question

_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?:[^\d]+(\d{2}):(\d{2}))?")


def parse_lme_date(s: str) -> datetime | None:
    m = _DATE.search(s or "")
    if not m:
        return None
    y, mo, d, hh, mm = m.groups()
    return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0))


def question_to_dataset(item: dict, roles: tuple[str, ...] = ("user",)) -> Dataset:
    recs: list[MemoryRecord] = []
    sids = item.get("haystack_session_ids") or [str(i) for i in range(len(item["haystack_sessions"]))]
    dates = item.get("haystack_dates") or [""] * len(sids)
    for sid, when, session in zip(sids, dates, item["haystack_sessions"]):
        base = parse_lme_date(when)
        for i, turn in enumerate(session):
            text = (turn.get("content") or "").strip()
            if not text or turn.get("role") not in roles:
                continue
            ts = base + timedelta(seconds=i) if base else None
            recs.append(MemoryRecord(f"{sid}:{i}", text, timestamp=ts, source=str(sid)))
    ans = item["answer"]
    q = Question(str(item["question_id"]), item["question"], [str(ans)], "current")
    return Dataset(f"lme-{item['question_id']}", MemoryStore(recs), [q])


def load_longmemeval(path: str | Path, qtypes: tuple[str, ...] = ("knowledge-update",),
                     limit: int | None = None, roles: tuple[str, ...] = ("user",)) -> list[Dataset]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for item in data:
        if qtypes and item.get("question_type") not in qtypes:
            continue
        out.append(question_to_dataset(item, roles))
        if limit and len(out) >= limit:
            break
    return out
