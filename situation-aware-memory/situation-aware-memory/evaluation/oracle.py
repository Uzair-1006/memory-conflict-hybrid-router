"""An idealised, rule-based stand-in for the LLM, for the synthetic dataset only.

It reads the true (subject, attribute, value) of every memory from the
generator's metadata, so it is a PERFECT extractor/decomposer. Accuracy
measured with it therefore reflects retrieval + router + policy + multi-hop
search, NOT language-model quality. Real extraction quality needs a real LLM.
"""
from __future__ import annotations

import json
import re

from .datasets import ATTRS, Dataset


class OracleLLM:
    def __init__(self, ds: Dataset, miss_rate: float = 0.0, seed: int = 0):
        """miss_rate: probability that a truly matching item is NOT returned,
        imitating the paper's strict-extraction recall loss. Deterministic."""
        self.meta = ds.meta
        self.miss_rate, self.seed = miss_rate, seed
        self.phrases = {a: p for a, (p, _) in ATTRS.items()}
        self.phrases["manager"] = "manager"

    def _route(self, prompt: str) -> str:
        """Plumbing test only: answers the LLM-router prompt with the rule router's plan."""
        from memcore.router import RuleRouter
        q = prompt.split("<QUESTION>\n")[-1].split("\n</QUESTION>")[0]
        p = RuleRouter().route(q)
        iso = lambda d: d.date().isoformat() if d else None
        return json.dumps({"query_type": p.query_type, "multi_hop": p.multi_hop,
                           "at": iso(p.policy_params.get("at")),
                           "start": iso(p.policy_params.get("start")),
                           "end": iso(p.policy_params.get("end"))})

    def complete_json(self, prompt: str) -> str:
        if "<PLAN>" in prompt or "<SUGGEST>" in prompt:      # plumbing: templated, not a real writer
            from memcore.demo_llm import HeuristicLLM
            return HeuristicLLM().complete_json(prompt)
        if "<ROUTER>" in prompt:
            return self._route(prompt)
        if "Answer the question using ONLY the retrieved items" in prompt:
            raise NotImplementedError("the oracle cannot imitate a direct-answer LLM; use --llm openai")
        if "<TASK>" in prompt:
            return self._decompose(prompt)
        return self._extract(prompt, salt="recall" if "<RECALL>" in prompt else "")

    def _decompose(self, prompt: str) -> str:
        q = prompt.split("<QUESTION>\n")[-1].split("\n</QUESTION>")[0]
        m = re.search(r"What is the (.+?) of my (.+?)\?", q)
        if m:
            return json.dumps({"hops": [
                {"id": 1, "query": f"Who is my {m.group(2)}?"},
                {"id": 2, "query": f"What is the {m.group(1)} of {{hop_1_answer}}?"}]})
        return json.dumps({"hops": [{"id": 1, "query": q}]})

    def _extract(self, prompt: str, salt: str = "") -> str:
        question = re.search(r"Question: (.*)\n", prompt).group(1).lower()
        out = []
        for line in prompt.split("Items:\n")[1].splitlines():
            m = re.match(r"\[([^\]]+)\]", line)
            info = self.meta.get(m.group(1)) if m else None
            if not info:
                continue
            if info["subject"] == "me":
                ok = re.search(r"\b(my|i)\b", question) and self.phrases[info["attr"]] in question
            else:
                ok = info["subject"].lower() in question and info["attr"] in question
            if ok and self.miss_rate:
                import hashlib
                h = int(hashlib.md5(f"{self.seed}|{salt}|{question}|{m.group(1)}".encode()).hexdigest()[:8], 16)
                ok = (h / 0xFFFFFFFF) >= self.miss_rate
            if ok:
                out.append({"fact_id": m.group(1), "answer_entity": info["value"], "confidence": 0.95})
        return json.dumps({"candidates": out})
