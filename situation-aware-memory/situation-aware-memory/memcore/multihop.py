"""Multi-hop resolution (the paper's CAR) with beam-limited backtracking.

Like CAR: decompose the question into single-hop queries, resolve each hop with
the retrieve -> extract -> policy primitive, and substitute each answer into the
next query. Unlike CAR, a hop keeps its top `beam` candidate answers, and if a
later hop finds no evidence the search backtracks to the next-best earlier
answer instead of committing to a wrong first hop (the paper's main FC-MH
failure mode).

The decomposition prompt is adapted from the paper's `DECOMP_PROMPT`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .extractor import Extractor
from .llm import LLMClient
from .pipeline import SingleHopPipeline
from .policies import Latest, Policy
from .retrieval import TOP_K, Retriever
from .schema import Candidate, OrderingError, order_key, ordering_mode

DECOMP_PROMPT = """<TASK>
Decompose this multi-hop question into a chain of atomic single-hop queries. Each hop must ask about ONE relationship only.
</TASK>

<RULES>
1. Each hop asks ONE question about ONE entity. The entity must be named in the original question (or be "my" for the user) OR be the answer of a previous hop, referenced as {hop_N_answer}.
2. Chain inside-out: start from the innermost entity, then move outward one relationship at a time.
3. No padding hops. Every hop's answer must feed a later hop or be the final answer.
</RULES>

<EXAMPLE>
<QUESTION>"What is the country of the company where the manager of Alice works?"</QUESTION>
{"hops": [
  {"id": 1, "query": "Who is the manager of Alice?"},
  {"id": 2, "query": "What company does {hop_1_answer} work at?"},
  {"id": 3, "query": "What country is {hop_2_answer} based in?"}
]}
</EXAMPLE>

<QUESTION>
{question}
</QUESTION>

Return ONLY valid JSON: {"hops": [{"id": 1, "query": "..."}, ...]}"""

_PLACEHOLDER = re.compile(r"\{hop_(\d+)_answer\}")


@dataclass(frozen=True)
class HopTrace:
    hop_id: int
    query: str
    chosen: Optional[str]
    alternatives: tuple[str, ...]
    abstained: bool
    reason: str = ""


@dataclass(frozen=True)
class MultiHopResult:
    question: str
    answer: object
    abstained: bool
    evidence: list[str]
    hops: list[HopTrace] = field(default_factory=list)
    backtracked: bool = False
    reason: str = ""
    confidence: float = 0.0


class MultiHopPipeline:
    def __init__(self, retriever: Retriever, extractor: Extractor, llm: LLMClient,
                 final_policy: Policy | None = None, beam: int = 2, top_k: int = TOP_K,
                 max_hop_runs: int = 24):
        self.retriever, self.extractor, self.llm = retriever, extractor, llm
        self.final_policy = final_policy or Latest()
        self.beam, self.top_k, self.max_hop_runs = beam, top_k, max_hop_runs

    # -- decomposition -----------------------------------------------------
    def decompose(self, question: str) -> list[dict] | None:
        raw = self.llm.complete_json(DECOMP_PROMPT.replace("{question}", question))
        try:
            hops = json.loads(raw)["hops"]
            ok = isinstance(hops, list) and hops and all(
                isinstance(h, dict) and "query" in h and "id" in h for h in hops)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        return hops if ok else None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _alternatives(cands: list[Candidate]) -> list[Candidate]:
        """One candidate per distinct value, newest first (confidence-first if
        the candidates cannot be ordered)."""
        try:
            mode = ordering_mode(cands)
            ranked = sorted(cands, key=lambda c: order_key(c, mode), reverse=True)
        except OrderingError:
            ranked = sorted(cands, key=lambda c: -c.confidence)
        seen, out = set(), []
        for c in ranked:
            if c.norm_value() not in seen:
                seen.add(c.norm_value())
                out.append(c)
        return out

    @staticmethod
    def _substitute(query: str, bound: dict[int, str]) -> str | None:
        def rep(m):
            return bound[int(m.group(1))]
        try:
            return _PLACEHOLDER.sub(rep, query)
        except KeyError:
            return None

    # -- search ------------------------------------------------------------
    def run(self, question: str) -> MultiHopResult:
        hops = self.decompose(question)
        if hops is None:
            return MultiHopResult(question, None, True, [], reason="decomposition failed")
        trace: list[HopTrace] = []
        budget = [self.max_hop_runs]
        backtracked = [False]

        def solve(i: int, bound: dict[int, str]):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            hop = hops[i]
            query = self._substitute(hop["query"], bound)
            if query is None:
                return None
            last = i == len(hops) - 1
            pipe = SingleHopPipeline(self.retriever, self.extractor,
                                     self.final_policy if last else Latest(), self.top_k)
            pr = pipe.run(query)
            if last:
                trace.append(HopTrace(hop["id"], query, None if pr.abstained else str(pr.answer),
                                      (), pr.abstained, pr.policy.reason))
                return pr if not pr.abstained else None
            alts = self._alternatives(pr.extraction.candidates)[: self.beam]
            trace.append(HopTrace(hop["id"], query, alts[0].value if alts else None,
                                  tuple(a.value for a in alts), not alts, pr.policy.reason))
            for k, alt in enumerate(alts):
                if k > 0:
                    backtracked[0] = True
                sub = solve(i + 1, {**bound, int(hop["id"]): alt.value})
                if sub is not None:
                    return sub
            return None

        final = solve(0, {})
        if final is None:
            return MultiHopResult(question, None, True, [], trace, backtracked[0],
                                  "no hop chain produced an answer")
        return MultiHopResult(question, final.answer, False, final.evidence, trace,
                              backtracked[0], final.policy.reason, final.policy.confidence)
