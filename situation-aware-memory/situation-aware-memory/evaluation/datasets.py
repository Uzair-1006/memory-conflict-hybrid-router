"""Datasets: a synthetic personal-memory generator, JSONL loading, and the
paper's FactConsolidation benchmark.

The synthetic generator builds ONE user's evolving memory (versioned facts with
timestamps, paraphrased statements, noise) and questions of every type the
router must handle. Gold answers are computed by independent code here, not by
the policies under test.
"""
from __future__ import annotations

import calendar
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from memcore.memory import MemoryRecord, MemoryStore


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    gold: list[str]
    qtype: str                      # current | previous | as_of | count | multi_hop


@dataclass
class Dataset:
    name: str
    store: MemoryStore
    questions: list[Question]
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)   # fact_id -> subject/attr/value (oracle only)


# ── synthetic personal memory ───────────────────────────────────────────────
ATTRS = {
    "city": ("city of residence", ["Pune", "Chennai", "Bengaluru", "Hyderabad", "Mumbai", "Delhi", "Kochi"]),
    "employer": ("employer", ["Infosys", "TCS", "Wipro", "Zoho", "Flipkart", "Google"]),
    "cuisine": ("favorite cuisine", ["Italian", "Thai", "South Indian", "Mexican", "Japanese"]),
    "gym": ("gym membership", ["CultFit", "Anytime Fitness", "FitHub", "PowerZone"]),
}
TEMPLATES = ["My {p} is {v}.", "Update: my {p} is now {v}.", "These days my {p} is {v}."]
PERSON_TEMPLATES = ["{n}'s city is {v}.", "Update: {n}'s city is now {v}.", "These days {n}'s city is {v}."]
NAMES = ["Priya", "Rahul", "Anita", "Karthik", "Meena", "Suresh"]
NOISE = [
    "I watched a cricket match on Friday.", "Remember to renew the library card.",
    "The monsoon arrived early this year.", "I read a book about gardening.",
    "Dinner with my cousins was fun.", "I need to buy a new laptop bag.",
    "Started learning the guitar last week.", "The train to the airport was delayed.",
    "I finished a long hike on Sunday.", "Booked tickets for the science fair.",
]
START, END = datetime(2023, 1, 1), datetime(2026, 10, 1)


def _timeline(rng: random.Random, k: int) -> list[datetime]:
    span = (END - START).days
    days = sorted(rng.sample(range(0, span), k))
    # keep versions at least 45 days apart so months/years are unambiguous
    out, last = [], -999
    for d in days:
        if d - last >= 45:
            out.append(START + timedelta(days=d))
            last = d
    return out


def _values(rng: random.Random, pool: list[str], k: int) -> list[str]:
    vals: list[str] = []
    while len(vals) < k:
        v = rng.choice(pool)
        if not vals or v != vals[-1]:
            vals.append(v)
    return vals


def generate_synthetic(seed: int = 0) -> Dataset:
    rng = random.Random(seed)
    recs: list[MemoryRecord] = []
    meta: dict[str, dict[str, Any]] = {}
    qs: list[Question] = []
    versions: dict[str, list[tuple[datetime, str]]] = {}

    def add(fid, text, ts, subject, attr, value):
        recs.append(MemoryRecord(fid, text, timestamp=ts, source=f"chat-{ts:%Y-%m}"))
        meta[fid] = {"subject": subject, "attr": attr, "value": value}

    def make_timeline(pool, lo=2, hi=4):
        ts = _timeline(rng, rng.randint(lo, hi) + 3)
        k = min(len(ts), rng.randint(lo, hi))
        ts = ts[:k]
        return list(zip(ts, _values(rng, pool, len(ts))))

    for attr, (phrase, pool) in ATTRS.items():
        vs = make_timeline(pool)
        versions[attr] = vs
        for i, (t, v) in enumerate(vs):
            add(f"{attr}-{i}", rng.choice(TEMPLATES).format(p=phrase, v=v), t, "me", attr, v)

    mgr_vs = make_timeline(NAMES[:], 2, 3)
    mgr_vs = [(t, n) for t, n in mgr_vs]
    versions["manager"] = mgr_vs
    for i, (t, n) in enumerate(mgr_vs):
        add(f"manager-{i}", rng.choice(TEMPLATES).format(p="manager", v=n), t, "me", "manager", n)
    person_city: dict[str, list[tuple[datetime, str]]] = {}
    for n in {n for _, n in mgr_vs}:
        pv = make_timeline(ATTRS["city"][1], 1, 3)
        person_city[n] = pv
        for i, (t, v) in enumerate(pv):
            add(f"{n}-city-{i}", rng.choice(PERSON_TEMPLATES).format(n=n, v=v), t, n, "city", v)

    for i in range(40):
        recs.append(MemoryRecord(f"noise-{i}", rng.choice(NOISE),
                                 timestamp=START + timedelta(days=rng.randint(0, (END - START).days))))
    rng.shuffle(recs)

    def month_end(t: datetime) -> datetime:
        return datetime(t.year, t.month, calendar.monthrange(t.year, t.month)[1], 23, 59, 59)

    n = 0

    def q(text, gold, qtype):
        nonlocal n
        n += 1
        qs.append(Question(f"s{seed}-q{n}", text, [str(g) for g in gold], qtype))

    for attr, vs in versions.items():
        phrase = ATTRS[attr][0] if attr in ATTRS else attr
        q(f"What is my current {phrase}?", [vs[-1][1]], "current")
        if len(vs) >= 2:
            q(f"What was my previous {phrase}?", [vs[-2][1]], "previous")
        t, _ = rng.choice(vs)
        cutoff = month_end(t)
        gold = [v for tt, v in vs if tt <= cutoff][-1]
        q(f"What was my {phrase} in {calendar.month_name[t.month]} {t.year}?", [gold], "as_of")
        q(f"How many different {phrase} entries have I mentioned?", [len({v for _, v in vs})], "count")
        yr = rng.choice(vs)[0].year
        q(f"How many different {phrase} entries did I mention in {yr}?",
          [len({v for tt, v in vs if tt.year == yr})], "count")

    cur_mgr = mgr_vs[-1][1]
    q("What is the city of my manager?", [person_city[cur_mgr][-1][1]], "multi_hop")
    return Dataset(f"synthetic-seed{seed}", MemoryStore(recs), qs, meta)


# ── JSONL / FactConsolidation ───────────────────────────────────────────────
def load_jsonl_dataset(memory_path: str | Path, questions_path: str | Path, name: str = "custom") -> Dataset:
    """questions.jsonl lines: {"qid", "question", "gold": [..], "type"}"""
    store = MemoryStore.from_jsonl(memory_path)
    qs = []
    for line in Path(questions_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            qs.append(Question(str(d["qid"]), d["question"], [str(g) for g in d["gold"]],
                               d.get("type", "current")))
    return Dataset(name, store, qs)


def load_factconsolidation(source: str = "factconsolidation_sh_6k", limit: int | None = None,
                           offset: int = 0) -> Dataset:
    """The paper's benchmark (MemoryAgentBench, Conflict_Resolution split).
    Needs `pip install datasets` and network access. Field names follow the
    base paper's loader; verify on first run."""
    from datasets import load_dataset  # lazy

    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
    row = next(r for r in ds if r["metadata"]["source"] == source)
    context = row["context"]
    matches = list(re.finditer(r"(\d+)\.\s", context))
    seen, recs = set(), []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        if idx in seen:
            continue
        seen.add(idx)
        recs.append(MemoryRecord(str(idx), context[m.end():end].strip().rstrip("."), serial=idx))
    qtype = "multi_hop" if "_mh_" in source else "current"
    qs = []
    for i, (qq, a) in enumerate(zip(row["questions"], row["answers"])):
        if i < offset:
            continue
        gold = [str(x) for x in (a if isinstance(a, (list, tuple)) else [a])]
        qs.append(Question(f"{source}-{i}", qq, gold, qtype))
        if limit and len(qs) >= limit:
            break
    return Dataset(source, MemoryStore(recs), qs)
