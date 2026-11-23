"""The full system: route -> (temporal / context filter) -> retrieve -> extract -> policy (-> multi-hop)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .embeddings import DenseRetriever, Embedder, HashingEmbedder
from .extractor import Extractor
from .hybrid import ContextFilterRetriever, HybridRetriever, TimeWindowRetriever
from .llm import LLMClient
from .memory import MemoryStore
from .multihop import MultiHopPipeline, MultiHopResult
from .pipeline import PipelineResult, SingleHopPipeline
from .policies import make_policy
from .recall import RecallBoostExtractor
from .retrieval import TOP_K, BM25Retriever
from .router import RoutePlan, RuleRouter


@dataclass(frozen=True)
class AgentResult:
    question: str
    plan: RoutePlan
    answer: Any
    abstained: bool
    evidence: list[str]
    detail: PipelineResult | MultiHopResult
    message: str = ""
    confidence: float = 0.0
    tentative: Any = None          # answer withheld because confidence was below the threshold

    @property
    def explanation(self) -> str:
        why = "; ".join(self.plan.reasons)
        return f"[{self.plan.query_type} via {self.plan.retriever}/{self.plan.policy}] {why}"


class Agent:
    def __init__(self, store: MemoryStore, llm: LLMClient, embedder: Embedder | None = None,
                 router: RuleRouter | None = None, top_k: int = TOP_K,
                 force_retriever: str | None = None, recall_mode: str | None = None,
                 use_temporal: bool = True, confidence_threshold: float = 0.0):
        self.store, self.embedder = store, embedder or HashingEmbedder()
        self.router = router or RuleRouter()
        self.extractor = RecallBoostExtractor(llm, recall_mode) if recall_mode else Extractor(llm)
        self.llm, self.top_k, self.force_retriever = llm, top_k, force_retriever
        self.use_temporal, self.confidence_threshold = use_temporal, confidence_threshold
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the search indexes. Call after the store changes."""
        bm25 = BM25Retriever(self.store)
        dense = DenseRetriever(self.store, self.embedder)
        self.retrievers = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])}
        self._has_timestamps = any(r.timestamp for r in self.store.all())

    def _retriever_for(self, plan: RoutePlan, context: Optional[dict]):
        r = self.retrievers[self.force_retriever or plan.retriever]
        if self.use_temporal and self._has_timestamps:
            p = plan.policy_params
            if plan.policy == "as_of":
                r = TimeWindowRetriever(r, end=p.get("at"))
            elif plan.policy == "count_distinct" and (p.get("start") or p.get("end")):
                r = TimeWindowRetriever(r, p.get("start"), p.get("end"))
        if context:
            r = ContextFilterRetriever(r, context)
        return r

    def answer(self, question: str, context: Optional[dict] = None) -> AgentResult:
        plan = self.router.route(question)
        retriever = self._retriever_for(plan, context)
        policy = make_policy(plan.policy, **plan.policy_params)
        if plan.multi_hop:
            res = MultiHopPipeline(retriever, self.extractor, self.llm, policy,
                                   top_k=self.top_k).run(question)
            conf = res.confidence
            reason = res.reason
        else:
            res = SingleHopPipeline(retriever, self.extractor, policy, self.top_k).run(question)
            conf = res.policy.confidence
            reason = res.policy.reason

        if res.abstained:
            msg = ("I could not find reliable evidence in memory to answer this "
                   f"({reason}). Could you clarify or add the missing information?")
            return AgentResult(question, plan, None, True, res.evidence, res, msg, 0.0)
        if conf < self.confidence_threshold:
            msg = (f"I found '{res.answer}' but I am only {conf:.0%} sure it is right "
                   f"(threshold {self.confidence_threshold:.0%}). Can you confirm it?")
            return AgentResult(question, plan, None, True, res.evidence, res, msg, conf, tentative=res.answer)
        return AgentResult(question, plan, res.answer, False, res.evidence, res, "", conf)
