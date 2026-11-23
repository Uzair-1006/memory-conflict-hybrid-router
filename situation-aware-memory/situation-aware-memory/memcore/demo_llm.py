"""A tiny rule-based stand-in LLM for offline demos and tests of the app layer.

NOT a language model. It only understands simple statements such as
"My city of residence is Pune." / "I live in Pune." and templated suggestions.
Use `OpenAIChat` for real behaviour.
"""
from __future__ import annotations

import json
import re

_STOP = {"what", "is", "my", "the", "a", "an", "of", "do", "did", "does", "i", "me", "current", "previous", "was",
         "were", "in", "how", "many", "different", "have", "mentioned", "entries", "mention", "who", "where",
         "which", "are", "to", "for", "and", "on", "at", "now", "update", "these", "days"}


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _value(text: str) -> str:
    m = re.search(r"\b(?:is|are|in|at|to)\s+(?:now\s+)?([^.]+?)\.?\s*$", text)
    return m.group(1).strip() if m else text.rstrip(".").strip()


class HeuristicLLM:
    def complete_json(self, prompt: str) -> str:
        if "<INGEST>" in prompt:
            return self._ingest(prompt)
        if "<PLAN>" in prompt:
            return self._plan(prompt)
        if "<SUGGEST>" in prompt:
            return self._suggest(prompt)
        if "<TASK>" in prompt:
            q = prompt.split("<QUESTION>\n")[-1].split("\n</QUESTION>")[0]
            return json.dumps({"hops": [{"id": 1, "query": q}]})
        if "<ROUTER>" in prompt:
            return "{}"                                  # invalid on purpose: the rule router takes over
        if "Answer the question using ONLY" in prompt:
            raise NotImplementedError("HeuristicLLM cannot imitate a direct-answer LLM")
        return self._extract(prompt)

    def _ingest(self, prompt: str) -> str:
        msg = re.search(r"Message \(written on [^)]*\): (.*?)\n\nReturn ONLY", prompt, re.S).group(1)
        facts = []
        for sent in re.split(r"(?<=[.!?])\s+|\n+", msg):
            sent = sent.strip()
            if not sent or sent.endswith("?") or len(sent.split()) < 3:
                continue
            facts.append({"text": sent if sent.endswith(".") else sent.rstrip("!") + ".", "domain": "other"})
        return json.dumps({"facts": facts})

    def _extract(self, prompt: str) -> str:
        q = re.search(r"Question: (.*)\n", prompt).group(1)
        q = re.sub(r"\b(as of|before|after|between|until|since|in|on)\b|\d{4}(-\d{2}(-\d{2})?)?", " ", q, flags=re.I)
        content = [t for t in _tokens(q) if t not in _STOP]
        about_me = bool(set(_tokens(q)) & {"my", "i"})
        out = []
        for m in re.finditer(r"^\[([^\]]+)\] (.*)$", prompt.split("Items:\n")[1], re.M):
            toks = set(_tokens(m.group(2)))
            if content and set(content) <= toks and (not about_me or toks & {"my", "i"}):
                out.append({"fact_id": m.group(1), "answer_entity": _value(m.group(2)), "confidence": 0.9})
        return json.dumps({"candidates": out})

    def _plan(self, prompt: str) -> str:
        goal = re.search(r"Goal: (.*)\n", prompt).group(1)
        preview = prompt.split("Memory sample:\n")[1].split("\n\nReturn ONLY")[0]
        phrases: list[str] = []
        for m in re.finditer(r"\bmy (.+?) is\b", preview, re.I):
            p = m.group(1).lower()
            if p not in phrases:
                phrases.append(p)
        gt = set(_tokens(goal))
        picked = [p for p in phrases if gt & set(_tokens(p)) - _STOP] or phrases
        return json.dumps({"questions": [f"What is my current {p}?" for p in picked[:3]]})

    def _suggest(self, prompt: str) -> str:
        goal = re.search(r"Goal: (.*)\n", prompt).group(1)
        ev = re.findall(r"^\[([^\]]+)\] (.*)$", prompt.split("Goal:", 1)[1], re.M)
        if not ev:
            return json.dumps({"suggestion": "", "claims": [], "clarification": "What should I take into account?"})
        claims = [{"text": t, "evidence": [i]} for i, t in ev]
        text = f"To make progress on '{goal}', build on what you already have: " + " ".join(t for _, t in ev)
        return json.dumps({"suggestion": text, "claims": claims, "clarification": None})
