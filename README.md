# Beyond Max-Serial: A Hybrid Router for Current and Historical Memory Queries in LLM Agents

An experiment extending the paper **"Don't Ask the LLM to Track Freshness"** (MemoryAgentBench FactConsolidation benchmark). It tests a two-branch hybrid that handles both *current-value* and *previous-value* questions over a memory containing conflicting facts.

## The Problem

LLM agents with long-term memory often hold conflicting facts: an old value and a newer update.

- **Current-value questions** ("What *is* the capital of France?"): the paper showed that Python's `max(serial)`, not the LLM, should pick the newest fact. This reaches 78–95% accuracy.
- **Previous/relative questions** ("What *was* it before the latest update?"): `max()` is the wrong operator here, and the paper admits failure on this case.
- **Prior override:** LLMs also answer from real-world knowledge ("Paris") even when memory says otherwise ("Lyon").

## The Approach

A two-branch router:

| Question type | Branch | How it works |
|---|---|---|
| Current | Deterministic | The LLM extracts candidate `(serial, value)` pairs and Python picks `max(serial)` |
| Previous | Virtual-context | An isolation prompt tells the model that the memory is the *only* world that exists, which suppresses prior override |

## Conditions Tested

| ID | Condition | Description |
|---|---|---|
| A | LLM Judgment | Baseline: the LLM reads all facts and answers |
| B | Deterministic Max | LLM extracts candidates, Python selects the newest |
| C | Virtual-Context | Isolation framing prompt, LLM answers directly |
| D | Hybrid Router | B for current questions, C for previous questions |

## Dataset

- 16 prior-conflict pairs, each a strong real-world fact (Paris) plus a counterfactual with a higher serial (Lyon)
- 224 facts in total: 6 fictional distractors before and 6 between each pair, shuffled
- 32 questions: each pair is asked as a `current` question and a `previous` question

## Results

Model: `openai/gpt-4o-mini` via OpenRouter, `temperature=0`. Full per-question detail is in [RESULTS.md](RESULTS.md).

| Condition | Current (n=16) | Previous (n=16) | Overall (n=32) |
|---|---|---|---|
| A: LLM Judgment | 0/16 (0%) | 15/16 (93%) | 15/32 (46%) |
| B: Deterministic | 15/16 (93%) | 1/16 (6%) | 16/32 (50%) |
| C: Virtual-Context | 0/16 (0%) | 15/16 (93%) | 15/32 (46%) |
| D: Hybrid Router | 15/16 (93%) | 15/16 (93%) | **30/32 (93%)** |

**Findings**

- A scores 0% on current questions because it answers from real-world priors.
- B is strong on current questions and near 0% on previous ones, as predicted.
- D is the only condition that does well on both types.
- C did **not** beat A, so the hypothesis that framing suppresses prior override is not supported on this model.

## Limitations

- D routes using the ground-truth `qtype`, so its score is an upper bound. A real system needs a question-type classifier.
- One model and only 16 items, so the sample is small.
- For previous questions the gold answer equals the real-world fact, so A can score well from priors alone. A cleaner test would use fictional priors.
- B's extraction is noisy, because the LLM sometimes returns wrong serials.
- The previous-value question assumes exactly two versions of each fact.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenRouter key:

```
OPENROUTER_API_KEY=your_key
```

## Usage

```bash
# Verify the dataset
python dataset.py

# Load the key, then do a cheap 4-question dry run
set -a; source .env; set +a
python experiment.py --dry-run

# Full run (writes results.json)
python experiment.py
```

| Flag | Description |
|---|---|
| `--dry-run` | Run 4 questions only (first 2 subjects × 2 types) |
| `--model MODEL` | Override the default model string |
| `--conditions A,B,C,D` | Run a subset of conditions |

## Project Structure

```
optionb/
├── dataset.py        # corpus and question builder
├── experiment.py     # 4-condition pipeline and evaluation
├── results.json      # raw per-question results (generated)
├── RESULTS.md        # formatted results report (generated)
├── requirements.txt
├── SKILL.md          # original build spec
└── README.md
```

## Hypotheses

1. B > A on current questions
2. B ≈ 0% on previous questions
3. C > A on prior-heavy items (**not supported**)
4. C < B on current questions
5. D ≈ best on both types
