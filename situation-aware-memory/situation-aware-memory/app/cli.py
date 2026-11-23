"""Terminal chat.  python -m app.cli --data-dir data --llm demo"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.system import MemorySystem

HELP = """Type a statement to remember it ("My city of residence is Pune."), or a question ending with ? to ask.
  /memories            list stored memories      /forget <id>          delete a memory
  /correct <id> <text> fix a memory's text       /goal <text>          add a goal
  /goals               list goals                /suggest [goal id]    goal-aware suggestions
  /help  /quit"""


def make_llm(kind: str):
    if kind == "demo":
        from memcore.demo_llm import HeuristicLLM
        return HeuristicLLM()
    from memcore.llm import OpenAIChat
    from memcore.guard import GuardedLLM
    return GuardedLLM(OpenAIChat())


def handle(system: MemorySystem, line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith("/"):
        cmd, _, rest = line[1:].partition(" ")
        rest = rest.strip()
        if cmd == "help":
            return HELP
        if cmd == "memories":
            ms = system.memories()
            return "\n".join(f"[{m['id']}] {m['when']}  {m['text']}" for m in ms) or "(no memories yet)"
        if cmd == "forget":
            return f"Deleted {rest}." if system.forget(rest) else f"No memory with id {rest!r}."
        if cmd == "correct":
            fid, _, text = rest.partition(" ")
            return f"Corrected {fid}." if system.correct(fid, text) else "Usage: /correct <id> <new text> (id must exist)."
        if cmd == "goal":
            if not rest:
                return "Usage: /goal <text>"
            g = system.add_goal(rest)
            return f"Added goal {g.goal_id}: {g.text}"
        if cmd == "goals":
            return "\n".join(f"{g.goal_id}: {g.text}" for g in system.goals.list()) or "(no goals yet)"
        if cmd == "suggest":
            out = []
            for s in system.suggest(rest or None):
                v = system.suggestion_view(s)
                if not v["suggestion"]:
                    out.append(f"Goal: {v['goal']}\n  I need more information: {v['clarification'] or 'no suggestion produced'}")
                    continue
                out.append(f"Goal: {v['goal']}\n  {v['suggestion']}" + ("" if v["grounded"] else "\n  (warning: not fully grounded in memory)"))
                for c in v["claims"]:
                    out.append(f"    - {c['text']}  <- " + ", ".join(f"[{e['id']}]" for e in c["evidence"]))
            return "\n".join(out) or "(no active goals; add one with /goal <text>)"
        return f"Unknown command /{cmd}. Try /help."
    if line.endswith("?"):
        v = system.answer_view(system.ask(line))
        if v["abstained"]:
            return v["message"]
        ev = "\n".join(f"  evidence [{e['id']}] {e['text']}" for e in v["evidence"])
        return f"{v['answer']}\n{ev}\n  how: {v['how']}"
    res = system.remember(line)
    if not res.parse_ok:
        return "I could not read that as a memory. Please rephrase."
    return "Remembered:\n" + "\n".join(f"  [{r.fact_id}] {r.text}" for r in res.records) if res.records \
        else "I did not find a personal fact to remember in that."


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--llm", choices=["demo", "openai"], default="openai")
    ap.add_argument("--confidence", type=float, default=0.0, help="ask for confirmation below this confidence (0-1)")
    a = ap.parse_args(argv)
    try:
        llm = make_llm(a.llm)
    except RuntimeError as e:
        sys.exit(f"{e}\nTip: run with --llm demo to try the app without an API key.")
    system = MemorySystem(a.data_dir, llm, a.confidence)
    print("Personal memory agent. /help for commands, /quit to exit.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() in ("/quit", "/exit"):
            break
        out = handle(system, line)
        if out:
            print(out)


if __name__ == "__main__":
    main()
