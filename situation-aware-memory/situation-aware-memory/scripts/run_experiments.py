"""Runs every experiment for the paper in one command and writes tables.

    python scripts/run_experiments.py --llm openai --repeats 3 --outdir results
    python scripts/run_experiments.py --llm openai --lme-path longmemeval_s_cleaned.json
    python scripts/run_experiments.py --llm oracle --only synthetic --repeats 2   # offline test of this script

Writes, per experiment, results/<exp>_run<k>.json (every question, answer, failure label),
plus results/summary.md and results/summary.tex (booktabs) with accuracy as mean +- std over
repeats. LLM output varies between runs, so one run is never enough for a paper.
Cost note: ~1000-2000 LLM calls per repeat with the default systems; start with --repeats 1.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.datasets import generate_synthetic, load_factconsolidation
from evaluation.longmemeval import load_longmemeval
from evaluation.oracle import OracleLLM
from evaluation.runner import evaluate

FC_SYSTEMS = ["direct_bm25", "fixed_bm25_latest", "fixed_bm25_latest+sibling", "fixed_bm25_latest+recall",
              "fixed_bm25_latest+both", "routed_full", "routed_full+both"]
MH_SYSTEMS = ["fixed_bm25_latest", "routed_full", "routed_llm", "routed_full+both"]
SYN_SYSTEMS = ["fixed_bm25_latest", "routed_policy_only", "routed_full", "routed_llm", "routed_full+recall", "routed_full+both"]
LME_SYSTEMS = ["direct_bm25", "fixed_bm25_latest", "routed_full", "routed_llm", "routed_full+both"]


def experiments(a):
    exps = {
        "fc_sh_6k": (lambda: [load_factconsolidation("factconsolidation_sh_6k", a.limit)], FC_SYSTEMS),
        "fc_mh_6k": (lambda: [load_factconsolidation("factconsolidation_mh_6k", a.limit)], MH_SYSTEMS),
        "synthetic": (lambda: [generate_synthetic(s) for s in range(a.seeds)], SYN_SYSTEMS),
    }
    if a.lme_path:
        exps["lme_ku"] = (lambda: load_longmemeval(a.lme_path, ("knowledge-update",), a.limit), LME_SYSTEMS)
    return exps


def ms(vals):
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["openai", "oracle"], default="openai")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5, help="synthetic personas")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--lme-path")
    ap.add_argument("--only", help="comma list: fc_sh_6k,fc_mh_6k,synthetic,lme_ku")
    ap.add_argument("--miss-rate", type=float, default=0.0, help="oracle only")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    exps = experiments(a)
    wanted = a.only.split(",") if a.only else list(exps)
    if a.llm == "oracle":
        wanted = [w for w in wanted if w == "synthetic"]
        factory = lambda ds: OracleLLM(ds, miss_rate=a.miss_rate)
        print("NOTE: oracle LLM = idealised extractor (offline test of the pipeline, not paper results)")
    else:
        from memcore.llm import OpenAIChat
        from memcore.guard import GuardedLLM
        llm = GuardedLLM(OpenAIChat())
        factory = lambda ds: llm
        print(f"model: {llm.model}")

    md, tex = ["# Experiment summary", f"model: {'oracle' if a.llm == 'oracle' else llm.model}; repeats: {a.repeats}", ""], []
    for name in wanted:
        build, systems = exps[name]
        datasets = build()
        if sum(len(d.questions) for d in datasets) == 0:
            print(f"[{name}] no questions, skipped")
            continue
        per_sys: dict[str, dict] = {s: {"acc": [], "calls": [], "ms": [], "fail": {}, "by_type": {}} for s in systems}
        n = 0
        for k in range(a.repeats):
            print(f"[{name}] repeat {k + 1}/{a.repeats} ...", flush=True)
            res = evaluate(datasets, factory, systems)
            (out / f"{name}_run{k + 1}.json").write_text(json.dumps(res, indent=1, default=str))
            for s in systems:
                st = res["summary"]["systems"][s]
                n = st["n"]
                per_sys[s]["acc"].append(st["accuracy"])
                per_sys[s]["calls"].append(st["llm_calls_per_q"])
                per_sys[s]["ms"].append(st["ms_per_q"])
                for t, (acc, cnt) in st["by_type"].items():
                    per_sys[s]["by_type"].setdefault(t, []).append(acc)
                for f, c in st["failures"].items():
                    per_sys[s]["fail"].setdefault(f, []).append(c)
        types = sorted({t for s in systems for t in per_sys[s]["by_type"]})
        header = ["System", "Acc. (mean±sd)"] + (types if len(types) > 1 else []) + ["Calls/q", "ms/q"]
        md += [f"## {name} (n={n} questions per run)", "", "| " + " | ".join(header) + " |",
               "|" + "---|" * len(header)]
        tex += [f"% {name}", "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}", "\\toprule",
                " & ".join(header).replace("±", "$\\pm$").replace("_", "\\_") + " \\\\", "\\midrule"]
        for s in systems:
            m, sd = ms(per_sys[s]["acc"])
            cells = [s.replace("_", " "), f"{100 * m:.1f} ± {100 * sd:.1f}"]
            if len(types) > 1:
                cells += [f"{100 * statistics.mean(per_sys[s]['by_type'][t]):.0f}" if t in per_sys[s]["by_type"] else "-"
                          for t in types]
            cells += [f"{statistics.mean(per_sys[s]['calls']):.2f}", f"{statistics.mean(per_sys[s]['ms']):.0f}"]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(c.replace("±", "$\\pm$").replace("_", "\\_") for c in cells) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}", ""]
        md += ["", "Failure counts (mean over repeats): " + "; ".join(
            f"{s}: " + ", ".join(f"{f}={statistics.mean(v):.1f}" for f, v in sorted(per_sys[s]['fail'].items()))
            for s in systems if per_sys[s]["fail"]), ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "summary.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"\nwrote {out}/summary.md and {out}/summary.tex")


if __name__ == "__main__":
    main()
