"""Ingestion: turn a chat message into atomic, timestamped memory records.

The LLM rewrites personal facts as short standalone first-person sentences and
keeps the predicate phrase stable across updates ("my city of residence"), so
later versions of the same fact are found together and ordered by time.
New information is ADDED (a new version); old versions stay for history queries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .llm import LLMClient
from .memory import MemoryRecord
from .persist import PersistentMemory

INGEST_PROMPT = """<INGEST>
Extract the personal facts the user states in this message and rewrite each as ONE short, standalone, first-person sentence, e.g. "My city of residence is Pune." or "My manager is Priya."
Rules:
1. One fact per sentence. Do not merge facts. Do not invent anything the user did not say.
2. Keep the relationship phrase stable and explicit ("my city of residence", "my employer", "my gym membership") so later updates use the same phrase.
3. Facts about other people use their name: "Priya's city is Chennai."
4. Skip questions, greetings, instructions to the assistant and opinions about the assistant. If there is no personal fact, return an empty list.
5. domain: one word for the area of life (work, health, family, finance, leisure, home, other).
</INGEST>

Message (written on {date}): {message}

Return ONLY valid JSON: {{"facts": [{{"text": "...", "domain": "..."}}]}}"""

MAX_FACTS, MAX_LEN = 10, 300


@dataclass(frozen=True)
class IngestResult:
    records: list[MemoryRecord]
    parse_ok: bool = True
    skipped: int = 0
    notes: list[str] = field(default_factory=list)


class Ingestor:
    def __init__(self, memory: PersistentMemory, llm: LLMClient):
        self.memory, self.llm = memory, llm

    def ingest(self, message: str, when: Optional[datetime] = None, source: str = "chat") -> IngestResult:
        when = when or datetime.now()
        raw = self.llm.complete_json(INGEST_PROMPT.format(date=when.date().isoformat(), message=message))
        try:
            facts = json.loads(raw)["facts"]
            if not isinstance(facts, list):
                raise ValueError("facts is not a list")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            return IngestResult([], False, 0, [f"parse error: {e}"])
        added, skipped = [], 0
        for f in facts[:MAX_FACTS]:
            text = f.get("text") if isinstance(f, dict) else None
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_LEN:
                skipped += 1
                continue
            domain = f.get("domain") if isinstance(f.get("domain"), str) else "other"
            added.append(self.memory.add(text, when, source, {"domain": domain.strip().lower() or "other"}))
        skipped += max(0, len(facts) - MAX_FACTS)
        return IngestResult(added, True, skipped)
