"""Checks our pure-Python BM25 against the `rank_bm25` library the paper used.

    pip install rank_bm25
    python scripts/check_bm25_parity.py

Compares the score of EVERY document for every question across several
synthetic personas. Passes if the largest absolute difference is < 1e-9.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("rank_bm25 is not installed. Run: pip install rank_bm25")

from evaluation.datasets import generate_synthetic
from memcore.retrieval import BM25Retriever, tokenize

worst, n = 0.0, 0
for seed in range(10):
    ds = generate_synthetic(seed)
    recs = ds.store.all()
    ours = BM25Retriever(ds.store)
    lib = BM25Okapi([tokenize(r.text) for r in recs])
    for q in ds.questions:
        toks = tokenize(q.text)
        ref = lib.get_scores(toks)
        for i in range(len(recs)):
            worst = max(worst, abs(ours._score(toks, i) - float(ref[i])))
            n += 1
print(f"compared {n} (question, document) scores; max abs difference = {worst:.3e}")
print("PARITY OK" if worst < 1e-9 else "PARITY FAILED: tell Claude this output")
