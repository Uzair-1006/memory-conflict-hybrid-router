"""Persistent, user-correctable memory.

An append-only JSONL event log (add / edit / delete) is replayed on start, so
nothing is ever silently lost and every change is auditable. New information
is added as a NEW timestamped record: versioning (not overwriting) is what lets
the policies answer "current", "previous" and "as of" questions. `correct`
edits a record in place and is meant for typos and wrong extractions, not for
facts that changed in real life.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .memory import MemoryRecord, MemoryStore


class PersistentMemory:
    def __init__(self, path: str | Path | None = None, clock: Callable[[], datetime] = datetime.now):
        self.path = Path(path) if path else None
        self.store = MemoryStore()
        self._clock = clock
        self._next = 1
        if self.path and self.path.exists():
            self._replay()

    # -- persistence -------------------------------------------------------
    def _log(self, event: dict[str, Any]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _replay(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["op"] == "add":
                ts = datetime.fromisoformat(e["timestamp"]) if e.get("timestamp") else None
                self.store.add(MemoryRecord(e["fact_id"], e["text"], timestamp=ts,
                                            source=e.get("source", ""), context=e.get("context", {})))
                m = re.fullmatch(r"m(\d+)", e["fact_id"])
                if m:
                    self._next = max(self._next, int(m.group(1)) + 1)
            elif e["op"] == "edit":
                rec = self.store.get(e["fact_id"])
                if rec:
                    self.store.replace(replace(rec, text=e["text"]))
            elif e["op"] == "delete":
                self.store.remove(e["fact_id"])

    # -- operations --------------------------------------------------------
    def add(self, text: str, timestamp: Optional[datetime] = None, source: str = "",
            context: Optional[dict[str, Any]] = None) -> MemoryRecord:
        text = " ".join(text.split())
        if not text:
            raise ValueError("empty memory text")
        rec = MemoryRecord(f"m{self._next}", text, timestamp=timestamp or self._clock(),
                           source=source, context=dict(context or {}))
        self._next += 1
        self.store.add(rec)
        self._log({"op": "add", "fact_id": rec.fact_id, "text": rec.text,
                   "timestamp": rec.timestamp.isoformat(), "source": rec.source, "context": rec.context})
        return rec

    def correct(self, fact_id: str, new_text: str) -> bool:
        rec = self.store.get(fact_id)
        new_text = " ".join(new_text.split())
        if rec is None or not new_text:
            return False
        self.store.replace(replace(rec, text=new_text))
        self._log({"op": "edit", "fact_id": fact_id, "text": new_text})
        return True

    def delete(self, fact_id: str) -> bool:
        ok = self.store.remove(fact_id)
        if ok:
            self._log({"op": "delete", "fact_id": fact_id})
        return ok

    def records(self) -> list[MemoryRecord]:
        return sorted(self.store.all(), key=lambda r: (r.timestamp or datetime.min, r.fact_id))
