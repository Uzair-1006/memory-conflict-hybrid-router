Markdown · SKILL.md

# Option B Hybrid — Memory Conflict Resolution Experiment

---

## name: optionb-hybrid-experiment description: Build the Option B Hybrid memory conflict resolution experiment from scratch — dataset, 4-condition pipeline, OpenRouter evaluation, and results table.

## Research Context

Final-year research paper extending the SOTA paper **"Don't Ask the LLM to Track Freshness"** (MemoryAgentBench FactConsolidation benchmark).

**Paper's core insight**: When memory has conflicting facts, `max(serial)` in Python — not the LLM — should select the newest fact. Achieves 78–95% accuracy vs ≤54% for all published systems.

**Gap this project fills**: The paper only solves *current-value* questions ("What IS the X?"). It explicitly admits failure on *relative/previous* questions where `max()` is the wrong operator. Option B closes that gap.

**Novel contribution — two-branch hybrid**:

- Deterministic branch (current-value Qs): LLM extracts candidates → Python picks `max(serial)`
- Virtual-context branch (previous/relative Qs): isolation framing suppresses LLM prior-override

---

## Step 1 — Build dataset.py (start here, folder is empty)

16 prior-conflict pairs. Each = strong real-world prior + counterfactual with higher serial.

```python
PRIOR_CONFLICTS = [
    ("France", "capital city", "Paris", "Lyon"),
    ("Japan", "national sport", "sumo", "pesapallo"),
    ("Australia", "capital city", "Canberra", "Perth"),
    ("the Mona Lisa", "painter", "Leonardo da Vinci", "Raphael"),
    ("Mount Everest", "country it is located in", "Nepal", "Chile"),
    ("the speed of light medal", "awarded discipline", "physics", "botany"),
    ("the Nile", "continent", "Africa", "Antarctica"),
    ("Shakespeare", "most famous play", "Hamlet", "The Tempest"),
    ("the dollar", "issuing country", "United States", "Iceland"),
    ("the kangaroo", "native continent", "Australia", "Europe"),
    ("Einstein", "field of work", "physics", "marine biology"),
    ("the Great Wall", "country", "China", "Peru"),
    ("the Amazon river", "continent", "South America", "Asia"),
    ("gold", "chemical symbol", "Au", "Zz"),
    ("the Sahara", "type of region", "desert", "rainforest"),
    ("Beethoven", "primary profession", "composer", "architect"),
]
```

Function: `build_corpus_and_questions(n_distractors_per_item=6, seed=0)`

Returns:

- `corpus` — list of `(serial, text)` tuples, shuffled. 224 facts total.
- `questions` — 32 dicts (16 current + 16 previous)

Question dict schema:

```python
{
    "qid": "France|current",
    "question": "What is the capital city of France?",
    "subject": "France",
    "relation": "capital city",
    "prior": "Paris",
    "counterfactual": "Lyon",
    "gold": "Lyon",       # current → newest wins; previous → prior value wins
    "qtype": "current",   # or "previous"
}
```

Corpus fact format:

- `"The {relation} of {subject} is {value}."`
- Prior gets lower serial, counterfactual gets higher serial
- Sprinkle 6 distractor facts before and between each conflict pair
- `rng.shuffle(corpus)` at the end (realistic: not pre-sorted)

---

## Step 2 — Build experiment.py

### API

```python
import os
from openai import OpenAI   # pip install openai>=1.0 if needed

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)
MODEL = "openai/gpt-4o-mini"
```

### Corpus display format

Each fact: `"[Serial #{serial}] {text}"`

---

### Condition A — LLM Judgment (baseline)

```
System: You are a memory agent. Answer using only the facts provided. 
        Be concise — answer in one word or short phrase only.

User:   Facts:
        {corpus_text}

        Question: {question}
```

---

### Condition B — Deterministic Max

Step 1 — extract candidates:

```
From the facts below, extract all facts about the {relation} of {subject}.
Return ONLY a JSON list: [{"serial": <int>, "value": "<str>"}]
If none found, return [].

Facts:
{corpus_text}
```

Step 2 — Python selects answer:

```python
answer = max(candidates, key=lambda x: x["serial"])["value"]
```

Correct on "current" questions. Always wrong on "previous" (returns counterfactual, but gold is prior).

---

### Condition C — Virtual-Context Framing

```
System: You are an AI assistant operating inside an isolated memory database.
        This database contains the ONLY facts that exist in your world.
        You have NO knowledge of the real world — answer SOLELY from the facts below.
        Imagine these facts define your entire reality.
        Be concise — answer in one word or short phrase only.

User:   Facts:
        {corpus_text}

        Question: {question}
```

Targets prior-override failure. Does NOT fix serial-drift.

---

### Condition D — Hybrid Router (the contribution)

```python
if question["qtype"] == "current":
    answer = run_condition_B(question, corpus)
else:
    answer = run_condition_C(question, corpus)
```

For real deployment: auto-detect qtype with keyword classifier or LLM. For this experiment: use `qtype` field as ground-truth routing → measures upper-bound performance.

---

### Evaluation

```python
correct = prediction.strip().lower() == gold.strip().lower()
```

Save `results.json` — list of:

```json
{
  "qid": "France|current",
  "qtype": "current",
  "condition": "A",
  "question": "What is the capital city of France?",
  "gold": "Lyon",
  "prediction": "Paris",
  "correct": false
}
```

---

### Summary Table (print after all runs)

```
Condition        | Current (n=16) | Previous (n=16) | Overall (n=32)
A: LLM Judgment  |  X/16 (XX%)    |  X/16 (XX%)     |  X/32 (XX%)
B: Deterministic |  X/16 (XX%)    |  X/16 (XX%)     |  X/32 (XX%)
C: Virtual-Ctx   |  X/16 (XX%)    |  X/16 (XX%)     |  X/32 (XX%)
D: Hybrid Router |  X/16 (XX%)    |  X/16 (XX%)     |  X/32 (XX%)
```

---

### CLI Flags

- `--dry-run` — run 4 questions only (first 2 subjects × 2 qtypes) to verify before spending budget
- `--model MODEL` — override default model string
- `--conditions A,B,C,D` — run subset of conditions

---

## Hypothesis (add as comments in experiment.py)

```python
# HYPOTHESIS:
# B > A on current questions   → deterministic beats raw LLM judgment
# B ≈ 0% on previous questions → max() always returns counterfactual, gold is prior
# C > A on prior-heavy items   → virtual-context suppresses prior-override
# C < B on current questions   → framing alone can't match deterministic serial selection
# D ≈ best on BOTH types       → routes each question to the correct branch
```

---

## File Structure

```
optionb/
├── dataset.py       ← build first
├── experiment.py    ← build second
├── results.json     ← generated at runtime
├── requirements.txt ← openai>=1.0
└── SKILL.md         ← this file
```

---

## First Commands After Building

```bash
# verify dataset
python dataset.py

# test pipeline (4 questions only)
export OPENROUTER_API_KEY=your_key
python experiment.py --dry-run

# full run
python experiment.py
```