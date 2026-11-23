"""Runs systems over datasets and reports accuracy, router accuracy, cost, significance."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Protocol

from memcore.agent import Agent
from memcore.direct import DirectAnswerer
from memcore.llm_router import LLMRouter
from memcore.llm import CountingLLM, LLMClient
from memcore.pipeline import SingleHopPipeline
from memcore.extractor import Extractor
from memcore.recall import RecallBoostExtractor
from memcore.policies import Latest
from memcore.retrieval import BM25Retriever

from .datasets import Dataset, Question
from .metrics import match, mcnemar_exact, wilson


@dataclass
class Output:
    answer: object
    abstained: bool
    predicted_type: str | None
    llm_calls: int
    seconds: float
    diag: dict | None = None


def _diag_single(pr) -> dict | None:
    """What was retrieved and extracted (single-hop pipeline results only)."""
    if not (hasattr(pr, "hits") and hasattr(pr, "extraction")):
        return None
    return {"retrieved_ids": [h.record.fact_id for h in pr.hits],
            "retrieved_texts": [h.record.text for h in pr.hits],
            "candidate_values": [c.value for c in pr.extraction.candidates]}


def classify_failure(golds: list[str], diag: dict | None, correct: bool) -> str | None:
    """Where did a wrong answer go wrong? (heuristic, by string match on the gold answer)

    retrieval_miss       no retrieved memory contains the gold answer
    extraction_miss      gold answer was retrieved but the extractor did not return it
    policy_or_selection  extractor returned it but the final answer is still wrong
    model_error          (direct baseline) gold was retrieved but the answer is wrong
    n/a                  not classifiable (counts, multi-hop, no diagnostics)
    """
    if correct:
        return None
    if diag is None or any(str(g).strip().isdigit() for g in golds):
        return "n/a"
    if not any(match(t, golds) for t in diag["retrieved_texts"]):
        return "retrieval_miss"
    cands = diag["candidate_values"]
    if cands is None:
        return "model_error"
    if not any(match(v, golds) for v in cands):
        return "extraction_miss"
    return "policy_or_selection"


class System(Protocol):
    name: str

    def answer(self, question: str) -> Output: ...


class FixedBaseline:
    """Fixed retrieval (BM25) + fixed policy (latest), single hop: the paper's
    SH-CONFLICT applied to every question regardless of type."""

    def __init__(self, name: str, ds: Dataset, llm: LLMClient, recall_mode: str | None = None):
        self.name = name
        self.llm = CountingLLM(llm)
        ext = RecallBoostExtractor(self.llm, recall_mode) if recall_mode else Extractor(self.llm)
        self.pipe = SingleHopPipeline(BM25Retriever(ds.store), ext, Latest())

    def answer(self, question: str) -> Output:
        c0, t0 = self.llm.calls, time.perf_counter()
        r = self.pipe.run(question)
        return Output(r.answer, r.abstained, None, self.llm.calls - c0, time.perf_counter() - t0,
                      _diag_single(r))


class DirectSystem:
    """The paper's direct-answer baseline: BM25 top-10 then ONE LLM call."""

    def __init__(self, name: str, ds: Dataset, llm: LLMClient):
        self.name = name
        self.llm = CountingLLM(llm)
        self.direct = DirectAnswerer(BM25Retriever(ds.store), self.llm)

    def answer(self, question: str) -> Output:
        c0, t0 = self.llm.calls, time.perf_counter()
        r = self.direct.run(question)
        diag = {"retrieved_ids": [h.record.fact_id for h in r.hits],
                "retrieved_texts": [h.record.text for h in r.hits], "candidate_values": None}
        return Output(r.answer, r.abstained, None, self.llm.calls - c0, time.perf_counter() - t0, diag)


class RoutedSystem:
    def __init__(self, name: str, ds: Dataset, llm: LLMClient, force_retriever: str | None = None,
                 llm_router: bool = False, recall_mode: str | None = None, use_temporal: bool = True):
        self.name = name
        self.llm = CountingLLM(llm)
        router = LLMRouter(self.llm) if llm_router else None
        self.agent = Agent(ds.store, self.llm, router=router, force_retriever=force_retriever,
                           recall_mode=recall_mode, use_temporal=use_temporal)

    def answer(self, question: str) -> Output:
        c0, t0 = self.llm.calls, time.perf_counter()
        r = self.agent.answer(question)
        qt = "multi_hop" if r.plan.multi_hop else r.plan.query_type
        return Output(r.answer, r.abstained, qt, self.llm.calls - c0, time.perf_counter() - t0,
                      _diag_single(r.detail))


SYSTEM_BUILDERS: dict[str, Callable[[Dataset, LLMClient], System]] = {
    "fixed_bm25_latest": lambda ds, llm: FixedBaseline("fixed_bm25_latest", ds, llm),
    "routed_policy_only": lambda ds, llm: RoutedSystem("routed_policy_only", ds, llm, "bm25"),
    "routed_full": lambda ds, llm: RoutedSystem("routed_full", ds, llm),
    "routed_llm": lambda ds, llm: RoutedSystem("routed_llm", ds, llm, llm_router=True),
    "direct_bm25": lambda ds, llm: DirectSystem("direct_bm25", ds, llm),
}


def build_system(name: str, ds: Dataset, llm: LLMClient) -> System:
    """`<system>` or `<system>+sibling|+recall|+both` (recall-boosting extraction)."""
    base, _, mode = name.partition("+")
    if mode and mode not in ("sibling", "recall", "both"):
        raise ValueError(f"unknown recall mode {mode!r} in {name!r}")
    if base == "fixed_bm25_latest":
        return FixedBaseline(name, ds, llm, mode or None)
    if base in ("routed_full", "routed_policy_only", "routed_llm", "routed_notemporal"):
        return RoutedSystem(name, ds, llm, "bm25" if base == "routed_policy_only" else None,
                            base == "routed_llm", mode or None, base != "routed_notemporal")
    if mode:
        raise ValueError(f"{base!r} does not take a recall mode")
    return SYSTEM_BUILDERS[base](ds, llm)


def evaluate(datasets: list[Dataset], llm_factory: Callable[[Dataset], LLMClient],
             systems: list[str]) -> dict:
    rows: dict[str, list[dict]] = defaultdict(list)
    for ds in datasets:
        llm = llm_factory(ds)
        for name in systems:
            sysm = build_system(name, ds, llm)
            for qn in ds.questions:
                o = sysm.answer(qn.text)
                ok = match(o.answer, qn.gold)
                rows[name].append({
                    "qid": qn.qid, "qtype": qn.qtype, "question": qn.text, "gold": qn.gold,
                    "answer": o.answer, "abstained": o.abstained, "correct": ok,
                    "failure": classify_failure(qn.gold, o.diag, ok), "diag": o.diag,
                    "predicted_type": o.predicted_type, "llm_calls": o.llm_calls, "seconds": o.seconds,
                })
    return {"rows": rows, "summary": summarize(rows, systems)}


def summarize(rows: dict[str, list[dict]], systems: list[str]) -> dict:
    out: dict = {"systems": {}, "mcnemar": {}}
    for name in systems:
        r = rows[name]
        n, k = len(r), sum(x["correct"] for x in r)
        by_type: dict[str, list[bool]] = defaultdict(list)
        for x in r:
            by_type[x["qtype"]].append(x["correct"])
        routed = [x for x in r if x["predicted_type"] is not None]
        out["systems"][name] = {
            "n": n, "accuracy": k / n if n else 0.0, "ci95": wilson(k, n),
            "by_type": {t: (sum(v) / len(v), len(v)) for t, v in sorted(by_type.items())},
            "abstain_rate": sum(x["abstained"] for x in r) / n if n else 0.0,
            "router_type_accuracy": (sum(x["predicted_type"] == x["qtype"] for x in routed) / len(routed)
                                     if routed else None),
            "llm_calls_per_q": sum(x["llm_calls"] for x in r) / n if n else 0.0,
            "ms_per_q": 1000 * sum(x["seconds"] for x in r) / n if n else 0.0,
            "failures": dict(Counter(x["failure"] for x in r if x.get("failure"))),
        }
    base = systems[0]
    for name in systems[1:]:
        out["mcnemar"][f"{name} vs {base}"] = mcnemar_exact(
            [x["correct"] for x in rows[name]], [x["correct"] for x in rows[base]])
    return out


def format_report(summary: dict) -> str:
    lines = []
    types = sorted({t for s in summary["systems"].values() for t in s["by_type"]})
    head = f"{'system':<22}{'n':>5}{'acc':>8}{'95% CI':>15}" + "".join(f"{t:>11}" for t in types) \
           + f"{'abst':>7}{'rtr_acc':>9}{'calls/q':>9}{'ms/q':>8}"
    lines.append(head)
    for name, s in summary["systems"].items():
        lo, hi = s["ci95"]
        row = f"{name:<22}{s['n']:>5}{s['accuracy']:>8.3f}{f'[{lo:.2f},{hi:.2f}]':>15}"
        for t in types:
            row += f"{s['by_type'][t][0]:>11.2f}" if t in s["by_type"] else f"{'-':>11}"
        ra = s["router_type_accuracy"]
        row += f"{s['abstain_rate']:>7.2f}{('-' if ra is None else format(ra, '.2f')):>9}" \
               f"{s['llm_calls_per_q']:>9.2f}{s['ms_per_q']:>8.1f}"
        lines.append(row)
    lines.append("")
    for name, s in summary["systems"].items():
        if s.get("failures"):
            lines.append(f"failures[{name}]: " + ", ".join(f"{k}={v}" for k, v in sorted(s["failures"].items())))
    lines.append("")
    for k, v in summary["mcnemar"].items():
        lines.append(f"McNemar {k}: only-first-right={v['only_a']}, only-baseline-right={v['only_b']}, p={v['p']:.4g}")
    return "\n".join(lines)
