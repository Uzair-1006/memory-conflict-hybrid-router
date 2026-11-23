"""Single-hop pipeline: retrieve -> extract -> policy.

This is the paper's SH-CONFLICT, rebuilt with a pluggable retriever and
policy. The router (next milestone) will choose both per question.
"""
from __future__ import annotations

from dataclasses import dataclass

from .extractor import ExtractionResult, Extractor
from .policies import Latest, Policy, PolicyResult
from .retrieval import TOP_K, Hit, Retriever


@dataclass(frozen=True)
class PipelineResult:
    question: str
    answer: object
    abstained: bool
    policy: PolicyResult
    hits: list[Hit]
    extraction: ExtractionResult

    @property
    def evidence(self) -> list[str]:
        """Provenance: ids of the memories that support the answer."""
        return [c.fact_id for c in self.policy.chosen]


class SingleHopPipeline:
    def __init__(self, retriever: Retriever, extractor: Extractor,
                 policy: Policy | None = None, top_k: int = TOP_K):
        self.retriever, self.extractor = retriever, extractor
        self.policy = policy or Latest()
        self.top_k = top_k

    def run(self, question: str, policy: Policy | None = None) -> PipelineResult:
        hits = self.retriever.search(question, self.top_k)
        extraction = self.extractor.extract(question, hits)
        result = (policy or self.policy)(extraction.candidates)
        return PipelineResult(question, result.answer, result.abstained,
                              result, hits, extraction)
