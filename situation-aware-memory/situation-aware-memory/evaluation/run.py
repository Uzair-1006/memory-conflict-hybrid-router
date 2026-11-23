"""CLI.

Offline sanity run (no API key; idealised extractor, see evaluation/oracle.py):
    python -m evaluation.run --dataset synthetic --llm oracle --seeds 5

Real run on your machine (needs OPENAI_API_KEY; add `pip install openai datasets`):
    python -m evaluation.run --dataset synthetic --llm openai --seeds 3
    python -m evaluation.run --dataset fc --source factconsolidation_sh_6k --limit 100 --llm openai
    python -m evaluation.run --dataset lme --path longmemeval_s_cleaned.json --limit 45 --llm openai \
        --systems direct_bm25,fixed_bm25_latest,routed_full,routed_llm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.longmemeval import load_longmemeval
from evaluation.datasets import generate_synthetic, load_factconsolidation, load_jsonl_dataset
from evaluation.oracle import OracleLLM
from evaluation.runner import evaluate, format_report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "fc", "jsonl", "lme"], default="synthetic")
    ap.add_argument("--llm", choices=["oracle", "openai"], default="oracle")
    ap.add_argument("--seeds", type=int, default=5, help="synthetic: number of personas")
    ap.add_argument("--source", default="factconsolidation_sh_6k")
    ap.add_argument("--miss-rate", type=float, default=0.0, help="oracle only: extractor recall loss")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0, help="fc: skip the first N questions (use 100 for a held-out slice)")
    ap.add_argument("--path", help="lme: path to the LongMemEval json file")
    ap.add_argument("--qtype", default="knowledge-update", help="lme: question_type filter ('' = all)")
    ap.add_argument("--memory"); ap.add_argument("--questions")
    ap.add_argument("--systems", default="fixed_bm25_latest,routed_policy_only,routed_full")
    ap.add_argument("--out", default=None, help="write per-question JSON here")
    a = ap.parse_args(argv)

    if a.dataset == "synthetic":
        datasets = [generate_synthetic(s) for s in range(a.seeds)]
    elif a.dataset == "fc":
        datasets = [load_factconsolidation(a.source, a.limit, a.offset)]
    elif a.dataset == "lme":
        datasets = load_longmemeval(a.path, (a.qtype,) if a.qtype else (), a.limit)
    else:
        datasets = [load_jsonl_dataset(a.memory, a.questions)]

    total = sum(len(ds.questions) for ds in datasets)
    if total == 0:
        ap.error("no questions to evaluate. If you used --offset, it is >= the number of questions "
                 "in this source (factconsolidation_sh_6k has 100). Lower --offset or drop it.")

    if a.llm == "oracle":
        if "direct_bm25" in a.systems:
            ap.error("direct_bm25 needs a real LLM (--llm openai)")
        if a.dataset != "synthetic":
            ap.error("--llm oracle only works with --dataset synthetic")
        factory = lambda ds: OracleLLM(ds, miss_rate=a.miss_rate)
        print("NOTE: oracle LLM = idealised extractor; measures retrieval/router/policy, not LLM quality.\n")
    else:
        from memcore.llm import OpenAIChat
        from memcore.guard import GuardedLLM
        llm = GuardedLLM(OpenAIChat())
        factory = lambda ds: llm

    res = evaluate(datasets, factory, a.systems.split(","))
    print(format_report(res["summary"]))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        print(f"\nper-question results -> {a.out}")


if __name__ == "__main__":
    main()
