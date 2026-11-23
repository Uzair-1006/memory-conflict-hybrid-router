"""The application facade: persistent memory + ingestion + agent + goals + suggestions."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from memcore.agent import Agent, AgentResult
from memcore.goals import Goal, GoalStore
from memcore.ingest import IngestResult, Ingestor
from memcore.llm import LLMClient
from memcore.persist import PersistentMemory
from memcore.suggest import Suggestion, SuggestionEngine


class MemorySystem:
    def __init__(self, data_dir: str | Path, llm: LLMClient, confidence_threshold: float = 0.0,
                 clock: Callable[[], datetime] = datetime.now):
        d = Path(data_dir)
        self.llm, self.clock = llm, clock
        self.memory = PersistentMemory(d / "memory.jsonl", clock)
        self.goals = GoalStore(d / "goals.json")
        self.ingestor = Ingestor(self.memory, llm)
        self.agent = Agent(self.memory.store, llm, confidence_threshold=confidence_threshold)
        self.engine = SuggestionEngine(self.agent, llm)

    # memory ----------------------------------------------------------------
    def remember(self, text: str, when: Optional[datetime] = None) -> IngestResult:
        res = self.ingestor.ingest(text, when or self.clock())
        self.agent.refresh()
        return res

    def forget(self, fact_id: str) -> bool:
        ok = self.memory.delete(fact_id)
        self.agent.refresh()
        return ok

    def correct(self, fact_id: str, new_text: str) -> bool:
        ok = self.memory.correct(fact_id, new_text)
        self.agent.refresh()
        return ok

    def ask(self, question: str, context: Optional[dict] = None) -> AgentResult:
        return self.agent.answer(question, context)

    # goals -----------------------------------------------------------------
    def add_goal(self, text: str) -> Goal:
        return self.goals.add(text)

    def suggest(self, goal_id: str | None = None) -> list[Suggestion]:
        goals = [self.goals.get(goal_id)] if goal_id else self.goals.list()
        return [self.engine.suggest(g.text, self.clock().date()) for g in goals if g]

    # views (JSON-friendly) -------------------------------------------------
    def memories(self) -> list[dict[str, Any]]:
        return [{"id": r.fact_id, "text": r.text, "when": r.timestamp.isoformat(timespec="minutes") if r.timestamp else "",
                 "domain": r.context.get("domain", "")} for r in self.memory.records()]

    def answer_view(self, r: AgentResult) -> dict[str, Any]:
        ev = [{"id": i, "text": (self.memory.store.get(i).text if self.memory.store.get(i) else "")} for i in r.evidence]
        return {"answer": r.answer, "abstained": r.abstained, "message": r.message, "tentative": r.tentative,
                "confidence": round(r.confidence, 2), "evidence": ev, "how": r.explanation}

    def suggestion_view(self, s: Suggestion) -> dict[str, Any]:
        texts = {i: (self.memory.store.get(i).text if self.memory.store.get(i) else "") for i in s.cited_ids}
        return {"goal": s.goal, "suggestion": s.text, "grounded": s.grounded, "clarification": s.clarification,
                "unresolved": s.unresolved, "invalid_ids": s.grounding.invalid_ids,
                "claims": [{"text": c.text, "evidence": [{"id": i, "text": texts.get(i, "")} for i in c.evidence]}
                           for c in s.claims]}
