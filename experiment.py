"""Option B Hybrid — memory conflict resolution experiment.

# HYPOTHESIS:
# B > A on current questions   → deterministic beats raw LLM judgment
# B ≈ 0% on previous questions → max() always returns counterfactual, gold is prior
# C > A on prior-heavy items   → virtual-context suppresses prior-override
# C < B on current questions   → framing alone can't match deterministic serial selection
# D ≈ best on BOTH types       → routes each question to the correct branch
"""
import argparse
import json
import os
import re

from openai import OpenAI

from dataset import build_corpus_and_questions

DEFAULT_MODEL = "openai/gpt-4o-mini"
NAMES = {"A": "A: LLM Judgment", "B": "B: Deterministic", "C": "C: Virtual-Ctx", "D": "D: Hybrid Router"}

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    return _client


def chat(model, system, user):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    r = client().chat.completions.create(model=model, messages=msgs, temperature=0)
    return (r.choices[0].message.content or "").strip()


def corpus_text(corpus):
    return "\n".join(f"[Serial #{s}] {t}" for s, t in sorted(corpus))


def run_A(q, corpus, model):
    system = ("You are a memory agent. Answer using only the facts provided. "
              "Be concise — answer in one word or short phrase only.")
    return chat(model, system, f"Facts:\n{corpus_text(corpus)}\n\nQuestion: {q['question']}")


def run_B(q, corpus, model):
    prompt = (f"From the facts below, extract all facts about the {q['relation']} of {q['subject']}.\n"
              'Return ONLY a JSON list: [{"serial": <int>, "value": "<str>"}]\n'
              "\"value\" must be ONLY the bare value (e.g. \"Paris\"), not the full sentence.\n"
              "\"serial\" must be the exact number shown in the [Serial #N] tag of that fact.\n"
              "If none found, return [].\n\n"
              f"Facts:\n{corpus_text(corpus)}")
    raw = chat(model, None, prompt)
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        candidates = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        candidates = []
    candidates = [c for c in candidates if isinstance(c, dict) and "serial" in c and "value" in c]
    if not candidates:
        return ""
    return str(max(candidates, key=lambda x: int(x["serial"]))["value"])


def run_C(q, corpus, model):
    system = ("You are an AI assistant operating inside an isolated memory database.\n"
              "This database contains the ONLY facts that exist in your world.\n"
              "You have NO knowledge of the real world — answer SOLELY from the facts below.\n"
              "Imagine these facts define your entire reality.\n"
              "Be concise — answer in one word or short phrase only.")
    return chat(model, system, f"Facts:\n{corpus_text(corpus)}\n\nQuestion: {q['question']}")


def run_D(q, corpus, model):
    # Ground-truth qtype routing → upper-bound performance.
    return run_B(q, corpus, model) if q["qtype"] == "current" else run_C(q, corpus, model)


RUNNERS = {"A": run_A, "B": run_B, "C": run_C, "D": run_D}


def norm(x):
    return re.sub(r"[^\w\s]", "", x).strip().lower()


def summary(results, conds):
    print(f"\n{'Condition':<17}| {'Current':<15}| {'Previous':<15}| Overall")
    for c in conds:
        rows = [r for r in results if r["condition"] == c]
        cells = []
        for qt in ("current", "previous", None):
            sub = [r for r in rows if qt is None or r["qtype"] == qt]
            k, n = sum(r["correct"] for r in sub), len(sub)
            cells.append(f"{k}/{n} ({100 * k // n if n else 0}%)")
        print(f"{NAMES[c]:<17}| {cells[0]:<15}| {cells[1]:<15}| {cells[2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conditions", default="A,B,C,D")
    args = ap.parse_args()
    conds = [c.strip().upper() for c in args.conditions.split(",")]

    corpus, questions = build_corpus_and_questions()
    if args.dry_run:
        questions = questions[:4]

    results = []
    for q in questions:
        for c in conds:
            try:
                pred = RUNNERS[c](q, corpus, args.model)
            except Exception as e:
                pred = f"ERROR: {e}"
            ok = norm(pred) == norm(q["gold"])
            results.append({"qid": q["qid"], "qtype": q["qtype"], "condition": c,
                            "question": q["question"], "gold": q["gold"], "prediction": pred, "correct": ok})
            print(f"{q['qid']:<32} {c} {'✓' if ok else '✗'}  gold={q['gold']!r} pred={pred!r}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    summary(results, conds)


if __name__ == "__main__":
    main()
