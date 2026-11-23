"""Evaluates goal-aware suggestions on the synthetic personas.

Compares OUR pipeline (plan -> resolve with router/policies -> grounded writing) with a naive
RAG baseline (top-k retrieval of the goal text -> writing). Automatic metrics, no LLM judge:

  grounding   share of claims whose cited ids are all real evidence given to the writer
  staleness   share of cited memories that are OUTDATED (a newer version of that fact exists)
  coverage    share of the facts the goal needs whose CURRENT version is cited
  clarified   share of goals answered with a clarifying question instead of a suggestion

Relevance/usefulness still needs human ratings: use --export-ratings to get a CSV to fill in.

    python -m evaluation.suggest_eval --llm oracle --seeds 5     # offline plumbing check
    python -m evaluation.suggest_eval --llm openai --seeds 3 --export-ratings ratings.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.datasets import ATTRS, Dataset, generate_synthetic
from evaluation.oracle import OracleLLM
from memcore.agent import Agent
from memcore.retrieval import BM25Retriever
from memcore.suggest import NaiveSuggester, SuggestionEngine

GOALS = [
    (("gym", "cuisine"), "Build a healthy weekly plan using my gym membership and my favorite cuisine."),
    (("city", "employer"), "Plan my work week around my city of residence and my employer."),
    (("manager",), "Prepare for my next catch-up with my manager."),
]


def latest_ids(ds: Dataset) -> dict[tuple[str, str], str]:
    """(subject, attr) -> fact_id of the newest version."""
    best: dict[tuple[str, str], tuple] = {}
    for r in ds.store.all():
        m = ds.meta.get(r.fact_id)
        if not m:
            continue
        key = (m["subject"], m["attr"])
        if key not in best or r.timestamp > best[key][0]:
            best[key] = (r.timestamp, r.fact_id)
    return {k: v[1] for k, v in best.items()}


def score(ds: Dataset, sug, needed: tuple[str, ...]) -> dict:
    latest = latest_ids(ds)
    cited = sug.cited_ids
    stale = [i for i in cited if i in ds.meta and latest[(ds.meta[i]["subject"], ds.meta[i]["attr"])] != i]
    need_ids = {latest[("me", a)] for a in needed if ("me", a) in latest}
    return {
        "grounding": sug.grounding.score if sug.text else None,
        "staleness": (len(stale) / len(cited)) if cited else None,
        "coverage": len(need_ids & set(cited)) / len(need_ids) if need_ids and sug.text else None,
        "clarified": 1.0 if (sug.clarification and not sug.text) else 0.0,
    }


def run(datasets: list[Dataset], llm_factory):
    rows = defaultdict(list)
    for ds in datasets:
        llm = llm_factory(ds)
        agent = Agent(ds.store, llm)
        ours = SuggestionEngine(agent, llm)
        naive = NaiveSuggester(BM25Retriever(ds.store), llm)
        for needed, goal in GOALS:
            for name, system in (("resolved (ours)", ours), ("naive_rag", naive)):
                s = system.suggest(goal)
                rows[name].append({**score(ds, s, needed), "goal": goal, "suggestion": s.text,
                                   "cited": s.cited_ids, "dataset": ds.name, "grounded": s.grounded})
    return rows


def report(rows) -> str:
    out = [f"{'system':<18}{'n':>4}{'grounding':>11}{'staleness':>11}{'coverage':>10}{'clarified':>11}"]
    for name, rs in rows.items():
        def mean(k):
            v = [r[k] for r in rs if r[k] is not None]
            return f"{sum(v) / len(v):.2f}" if v else "-"
        out.append(f"{name:<18}{len(rs):>4}{mean('grounding'):>11}{mean('staleness'):>11}"
                   f"{mean('coverage'):>10}{mean('clarified'):>11}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["oracle", "openai"], default="oracle")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--export-ratings", help="CSV with the suggestions for human relevance ratings (1-5)")
    a = ap.parse_args(argv)
    datasets = [generate_synthetic(s) for s in range(a.seeds)]
    if a.llm == "oracle":
        print("NOTE: oracle/heuristic LLM = templated writer; checks the pipeline, not language quality.\n")
        factory = lambda ds: OracleLLM(ds)
    else:
        from memcore.llm import OpenAIChat
        from memcore.guard import GuardedLLM
        llm = GuardedLLM(OpenAIChat())
        factory = lambda ds: llm
    rows = run(datasets, factory)
    print(report(rows))
    if a.export_ratings:
        with open(a.export_ratings, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["system", "dataset", "goal", "suggestion", "cited_ids", "relevance_1to5", "comment"])
            for name, rs in rows.items():
                for r in rs:
                    w.writerow([name, r["dataset"], r["goal"], r["suggestion"], " ".join(r["cited"]), "", ""])
        print(f"\nratings sheet -> {a.export_ratings}")


if __name__ == "__main__":
    main()
