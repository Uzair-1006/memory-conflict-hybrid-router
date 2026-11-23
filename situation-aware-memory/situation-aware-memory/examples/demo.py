"""Offline demo: shows each stage of the pipeline on the paper's CEO example.
Run:  python examples/demo.py
Uses a scripted stand-in for the LLM, so no API key or internet is needed."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memcore import (BM25Retriever, Extractor, MemoryRecord, MemoryStore,
                     Previous, ScriptedLLM, SingleHopPipeline)

store = MemoryStore([
    MemoryRecord("12",  "The CEO of Company X is Alice.", serial=12),
    MemoryRecord("847", "The CEO of Company X is Bob.", serial=847),
    MemoryRecord("30",  "The CFO of Company X is Carol.", serial=30),
    MemoryRecord("55",  "The headquarters of Company Y is Paris.", serial=55),
    MemoryRecord("56",  "Dogs are popular pets in many countries.", serial=56),
])

def fake_llm(prompt):  # stands in for gpt-4o-mini: keeps only 'CEO of Company X' items
    out = []
    for line in prompt.split("Items:\n")[1].splitlines():
        if line.startswith("[") and "CEO of Company X" in line:
            out.append({"fact_id": line[1:line.index("]")],
                        "answer_entity": line.rstrip(".").split(" is ")[-1]})
    return json.dumps({"candidates": out})

pipe = SingleHopPipeline(BM25Retriever(store), Extractor(ScriptedLLM(fake_llm)))
q = "Who is the CEO of Company X?"

for label, policy in [("latest (default)", None), ("previous", Previous())]:
    r = pipe.run(q, policy=policy)
    print(f"\n=== policy: {label} ===")
    print("1. RETRIEVED (BM25 top hits):")
    for h in r.hits[:4]:
        print(f"   #{h.rank} score={h.score:5.2f}  [{h.record.fact_id}] {h.record.text}")
    print("2. EXTRACTED candidates K:")
    for c in r.extraction.candidates:
        print(f"   serial={c.serial}  value={c.value}")
    print(f"3. POLICY '{r.policy.policy}': {r.policy.reason}")
    print(f"=> ANSWER: {r.answer}   evidence ids: {r.evidence}")

r = SingleHopPipeline(BM25Retriever(store), Extractor(ScriptedLLM(
    lambda _: '{"candidates": []}'))).run("What is the capital of Mars?")
print("\n=== no evidence ===")
print("abstained:", r.abstained, "| reason:", r.policy.reason)
