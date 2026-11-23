# Situation-Aware Hybrid Memory Retrieval and Execution

Final-year project (MITS, batch 65). Builds on *Reliable Post-Retrieval Assembly
for Agent Memory* (Reddy & Challaram, COLM 2026 workshop;
https://github.com/cvikasreddy/memory-conflict-resolution).

## Architecture

```
question -> Router (rules) -> {retriever, policy, multi-hop?}
  retrieval: BM25 | dense | hybrid (RRF) | temporal window
    -> Extractor E (LLM) -> candidates K  [fact, value, serial/timestamp, source, context, confidence]
    -> Policy A_pi: latest | previous | as_of(t) | count_distinct(window)
    -> answer + evidence ids, or abstain + clarification message
  multi-hop: decompose -> resolve each hop -> beam-limited backtracking
```

## Modules

| File | Purpose |
|---|---|
| `memcore/schema.py`, `policies.py` | `Candidate`; deterministic policies with abstention |
| `memcore/memory.py` | versioned `MemoryRecord` / `MemoryStore` (JSONL loader) |
| `memcore/retrieval.py` | pure-Python BM25 (paper settings) |
| `memcore/embeddings.py` | `HashingEmbedder` (offline, not semantic), `OpenAIEmbedder`, `DenseRetriever` |
| `memcore/hybrid.py` | RRF fusion, `TimeWindowRetriever` |
| `memcore/extractor.py`, `llm.py` | paper's extractor, adapted; LLM interface, test double, OpenAI backend |
| `memcore/pipeline.py` | single-hop retrieve -> extract -> policy |
| `memcore/router.py` | rule-based `RuleRouter` (date parsing, question-type cues) |
| `memcore/multihop.py` | CAR-style multi-hop with backtracking |
| `memcore/llm_router.py` | LLM classifies the question into a closed schema; invalid output falls back to rules |
| `memcore/direct.py` | the paper's direct-answer baseline (one LLM call) |
| `memcore/agent.py` | the full system (`Agent.answer`) |
| `evaluation/` | synthetic generator, metrics (SubEM, Wilson, McNemar), runner, CLI, FactConsolidation and LongMemEval loaders |
| `scripts/check_bm25_parity.py` | compares our BM25 with the `rank_bm25` library the paper used |

## Run

```bash
python -m unittest discover -s tests -v        # 81 tests, offline
python examples/demo.py                        # single-hop walkthrough
python -m evaluation.run --seeds 10            # offline sanity evaluation
python -m evaluation.run --seeds 10 --miss-rate 0.25   # with extractor recall loss
```

Real runs (your machine; needs `pip install openai datasets`):

```bash
python -m evaluation.run --dataset synthetic --llm openai --seeds 3
python -m evaluation.run --dataset fc --source factconsolidation_sh_6k --limit 100 --llm openai
python -m evaluation.run --dataset fc --source factconsolidation_mh_6k --limit 100 --llm openai
python -m evaluation.run --dataset lme --path longmemeval_s_cleaned.json --limit 45 --llm openai \
    --systems direct_bm25,fixed_bm25_latest,routed_full,routed_llm
pip install rank_bm25 && python scripts/check_bm25_parity.py   # settles the BM25 parity question
```

For OpenAI, set `OPENAI_API_KEY`. For OpenRouter, set `OPENROUTER_API_KEY`
in `.env`; the client will use `https://openrouter.ai/api/v1` automatically.
Set `PIPELINE_MODEL=openai/gpt-4o-mini` for OpenRouter. An OpenRouter key must
not be sent to the OpenAI API endpoint.

## First real result (FactConsolidation single-hop 6K, first 50 questions, real LLM)

| System | Accuracy |
|---|---|
| direct_bm25 (one LLM call) | 0.46 |
| fixed_bm25_latest (extract, then pick newest in code) | 0.86 to 0.88 across two runs |
| routed_full | 0.88 |

Extract-then-execute beats direct answering by about 40 points (paired McNemar p < 1e-5).
Re-running the same system changes 1 to 2 answers, so differences that small are noise.
On this dataset every question is "current value", so routing cannot help; its benefit
needs mixed question types (synthetic set, LongMemEval). The router once sent 4 of these 50
questions to the multi-hop path ("of the X" rule); fixed, with the 50 questions kept as a
regression test (`tests/fixtures/fc_sh_questions.txt`).

Each result row now has `failure` (retrieval_miss / extraction_miss / policy_or_selection /
model_error / n/a) and `diag` (retrieved ids and extracted candidates); the summary prints
a failure breakdown per system.

## Recall-boosting extraction (new)

Finding: on FactConsolidation (real LLM) every error left in extract-then-execute was an
extraction miss. The extractor returned only the real-world value ("Japanese", "Paris") and
skipped the newer counter-intuitive fact, even though retrieval had found it.
`memcore/recall.py` adds `sibling` (no extra call: same-template memories), `recall`
(second LLM call) and `both`. Select with a suffix: `fixed_bm25_latest+both`, `routed_full+recall`.

```bash
# fair held-out test: --offset 100 skips the 100 questions the fix was designed from
python -m evaluation.run --dataset fc --source factconsolidation_sh_6k --offset 100 --limit 100 --llm openai \
  --systems fixed_bm25_latest,fixed_bm25_latest+sibling,fixed_bm25_latest+recall,fixed_bm25_latest+both --out results_fc_recall.json
python -m evaluation.run --dataset synthetic --seeds 3 --llm openai \
  --systems routed_full,routed_full+sibling,routed_full+recall,routed_full+both
```

Offline (stand-in LLM, 25% misses drawn independently per pass: MECHANISM ONLY, not real gains):
routed_full 0.65, +sibling 0.75, +recall 0.90, +both 0.92. Cost: `recall`/`both` add one LLM call per question.

## One command for all paper experiments

```bash
python scripts/run_experiments.py --llm openai --repeats 3 --outdir results
python scripts/run_experiments.py --llm openai --repeats 3 --lme-path longmemeval_s_cleaned.json   # adds LongMemEval
```
Runs FactConsolidation single- and multi-hop, the synthetic set and (optionally) LongMemEval, repeats each
`--repeats` times (LLM output varies between runs), saves every question to `results/<exp>_run<k>.json`, and writes
`results/summary.md` and `results/summary.tex` (accuracy mean +- sd, per-type accuracy, calls and time per question).
Start with `--repeats 1 --limit 20` to check cost and errors.

## Offline sanity results (idealised extractor; NOT LLM results)

`OracleLLM` reads true metadata, so it is a perfect extractor. These numbers
test retrieval + router + policies + multi-hop logic only (260 questions, 10 personas).

| Extractor recall loss | fixed BM25 + latest | routed (full) |
|---|---|---|
| 0%  (5 personas, 130 q) | 0.31 | 1.00 |
| 10% | 0.29 | 0.81 |
| 25% | 0.30 | 0.60 |

Under recall loss, questions that need EVERY version (previous, count) fall
fastest: the base paper's strict-extraction weakness, and an open problem here.
`routed_full` equals `routed_policy_only` because the offline embedder is not
semantic; the retrieval-routing benefit can only be measured with real embeddings.

## Not yet verified

- Real LLM extraction quality (needs API key + run).
- BM25 parity: run `scripts/check_bm25_parity.py` (written, but `rank_bm25` was unavailable where it was built, so the script itself has not run to completion).
- Field names in `load_factconsolidation` and `evaluation/longmemeval.py`: written from the published formats, tested only on fixtures that follow the assumed schema.
- `routed_llm` offline numbers only test plumbing (the stand-in LLM copies the rule router).
- LongMemEval relative dates ("last month") are not resolved against the question date yet.


For GitHub Models, set in `.env` (never commit it):

```dotenv
GITHUB_TOKEN=your_github_models_token
GITHUB_MODELS_ENDPOINT=https://models.github.ai/inference
GITHUB_MODELS_MODEL=openai/gpt-4o-mini
PIPELINE_MODEL=openai/gpt-4o-mini
```

`GITHUB_MODELS_MODEL` is used by GitHub Models integrations; `PIPELINE_MODEL` selects the model passed to `OpenAIChat`.
GitHub Models has tight per-minute and per-day request limits, so start with small `--limit` values.

## Application layer (new)

Persistent personal memory + goal-aware suggestions, runnable without any API key:

```bash
python -m app.cli --data-dir data --llm demo      # terminal chat (try /help)
python -m app.web --data-dir data --llm demo      # browser UI at http://127.0.0.1:8000
# with a real LLM (uses your .env / OpenRouter config in memcore/llm.py):
python -m app.cli --data-dir data --llm openai --confidence 0.6
```

Text ending in `?` is a question; anything else is remembered. Commands: `/memories /forget <id> /correct <id> <text> /goal <text> /goals /suggest /help /quit`.

New modules: `memcore/persist.py` (event-sourced JSONL memory with edit/delete), `ingest.py` (LLM turns a message into atomic facts),
`goals.py`, `suggest.py` (goal-aware plan -> resolve facts with the policy engine -> grounded, cited suggestion; `NaiveSuggester` baseline),
`demo_llm.py` (rule-based stand-in so the app runs offline), `agent.py` (temporal and context filters, confidence-gated confirmation).

Evaluate suggestions: `python -m evaluation.suggest_eval --llm oracle --seeds 10` (offline plumbing check) or `--llm openai --seeds 3 --export-ratings ratings.csv`.

### Honest caveats
- `--llm demo` is a pattern matcher: good for checking the plumbing, not for judging quality.
- Offline suggestion numbers (staleness 0.00 vs naive 0.42) use a templated writer; they show the mechanism, not real-LLM quality.
- Confidence values are LLM self-reported and uncalibrated. Relative dates ("last week") are not resolved. Consent-based reminders are not built.
- Real-LLM gain of the temporal filter and recall boost is not yet measured.
