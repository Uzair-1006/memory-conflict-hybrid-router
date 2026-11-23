"""User-defined goals (what suggestions are aligned with)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Goal:
    goal_id: str
    text: str
    created: str
    active: bool = True


class GoalStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._goals: dict[str, Goal] = {}
        if self.path and self.path.exists():
            for d in json.loads(self.path.read_text(encoding="utf-8")):
                self._goals[d["goal_id"]] = Goal(**d)

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(g) for g in self._goals.values()], indent=1), encoding="utf-8")

    def add(self, text: str) -> Goal:
        text = " ".join(text.split())
        if not text:
            raise ValueError("empty goal")
        g = Goal(f"g{len(self._goals) + 1}", text, datetime.now().isoformat())
        self._goals[g.goal_id] = g
        self._save()
        return g

    def get(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def list(self, active_only: bool = True) -> list[Goal]:
        return [g for g in self._goals.values() if g.active or not active_only]

    def deactivate(self, goal_id: str) -> bool:
        g = self._goals.get(goal_id)
        if not g:
            return False
        g.active = False
        self._save()
        return True
