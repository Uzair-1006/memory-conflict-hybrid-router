"""Memory store: versioned records that retrieval searches over.

Unlike the paper's loader, which re-parses a text blob with a regex, records
are structured from the start. Metadata (serial / timestamp / source /
context) always comes from here, never from an LLM's copy of it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class MemoryRecord:
    fact_id: str
    text: str
    serial: Optional[int] = None
    timestamp: Optional[datetime] = None
    source: str = ""
    context: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def __init__(self, records: Iterable[MemoryRecord] = ()):
        self._records: dict[str, MemoryRecord] = {}
        for r in records:
            self.add(r)

    def add(self, record: MemoryRecord) -> None:
        if record.fact_id in self._records:
            raise ValueError(f"duplicate fact_id {record.fact_id!r}")
        self._records[record.fact_id] = record

    def remove(self, fact_id: str) -> bool:
        return self._records.pop(fact_id, None) is not None

    def replace(self, record: MemoryRecord) -> None:
        if record.fact_id not in self._records:
            raise KeyError(record.fact_id)
        self._records[record.fact_id] = record

    def get(self, fact_id: str) -> Optional[MemoryRecord]:
        return self._records.get(fact_id)

    def all(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MemoryStore":
        """One JSON object per line: fact_id, text, and optional serial,
        timestamp (ISO 8601), source, context."""
        store = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            ts = d.get("timestamp")
            store.add(MemoryRecord(
                fact_id=str(d["fact_id"]), text=d["text"], serial=d.get("serial"),
                timestamp=datetime.fromisoformat(ts) if ts else None,
                source=d.get("source", ""), context=d.get("context", {}),
            ))
        return store
