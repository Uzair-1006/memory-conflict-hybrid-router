import json
import threading
import urllib.request
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer

import pytest

from app.cli import handle
from app.system import MemorySystem
from app.web import make_handler
from memcore import (Agent, ContextFilterRetriever, GoalStore, HeuristicLLM, MemoryRecord,
                     MemoryStore, NaiveSuggester, PersistentMemory, ScriptedLLM, check_grounding)
from memcore.suggest import Claim


class Clock:
    def __init__(self):
        self.t = datetime(2026, 1, 1)

    def __call__(self):
        self.t += timedelta(days=1)
        return self.t


@pytest.fixture
def system(tmp_path):
    return MemorySystem(tmp_path, HeuristicLLM(), clock=Clock())


def test_persistence_replay_add_edit_delete(tmp_path):
    p = tmp_path / "m.jsonl"
    m = PersistentMemory(p, Clock())
    a = m.add("My city of residence is Pune.")
    b = m.add("My gym membership is CultFit.")
    assert m.correct(a.fact_id, "My city of residence is Delhi.")
    assert m.delete(b.fact_id)
    assert not m.delete("nope") and not m.correct("nope", "x")
    m2 = PersistentMemory(p, Clock())                      # reload from disk
    assert [(r.fact_id, r.text) for r in m2.records()] == [(a.fact_id, "My city of residence is Delhi.")]
    c = m2.add("new")                                       # ids never reused
    assert c.fact_id not in (a.fact_id, b.fact_id)


def test_goals_persist(tmp_path):
    g = GoalStore(tmp_path / "g.json")
    x = g.add("save money")
    assert [i.text for i in GoalStore(tmp_path / "g.json").list()] == ["save money"]
    assert g.deactivate(x.goal_id) and g.list() == []


def test_store_remove_replace():
    s = MemoryStore([MemoryRecord("a", "x")])
    s.replace(MemoryRecord("a", "y"))
    assert s.get("a").text == "y"
    with pytest.raises(KeyError):
        s.replace(MemoryRecord("zz", "q"))
    assert s.remove("a") and not s.remove("a")


def test_remember_then_ask_current_and_previous(system):
    system.remember("My city of residence is Pune.")
    system.remember("My city of residence is Hyderabad.")
    assert system.ask("What is my city of residence?").answer == "Hyderabad"
    assert system.ask("What was my previous city of residence?").answer == "Pune"


def test_forget_and_correct_update_answers(system):
    system.remember("My city of residence is Pune.")
    r = system.remember("My city of residence is Hyderabad.")
    system.correct(r.records[0].fact_id, "My city of residence is Mumbai.")
    assert system.ask("What is my city of residence?").answer == "Mumbai"
    system.forget(r.records[0].fact_id)
    assert system.ask("What is my city of residence?").answer == "Pune"


def test_abstains_when_no_evidence(system):
    r = system.ask("What is my favourite colour?")
    assert r.abstained and r.message


def test_ingest_parse_failure_stores_nothing(tmp_path):
    s = MemorySystem(tmp_path, ScriptedLLM(["not json"] * 3))
    res = s.remember("hello")
    assert not res.parse_ok and s.memories() == []


def test_confidence_threshold_asks_for_confirmation(tmp_path):
    s = MemorySystem(tmp_path, HeuristicLLM(), confidence_threshold=0.99)
    s.remember("My city of residence is Pune.")
    r = s.ask("What is my city of residence?")
    assert r.abstained and r.tentative == "Pune" and "confirm" in r.message


def test_context_filter():
    st = MemoryStore([MemoryRecord("a", "My city is Pune", context={"domain": "home"}),
                      MemoryRecord("b", "My city is Delhi", context={"domain": "work"})])
    from memcore import BM25Retriever
    hits = ContextFilterRetriever(BM25Retriever(st), {"domain": "work"}).search("city")
    assert [h.record.fact_id for h in hits] == ["b"]


def test_temporal_filter_as_of(tmp_path):
    s = MemorySystem(tmp_path, HeuristicLLM(), clock=Clock())
    s.remember("My city of residence is Pune.", datetime(2020, 1, 1))
    s.remember("My city of residence is Hyderabad.", datetime(2024, 1, 1))
    assert s.ask("What was my city of residence as of 2021-06-01?").answer == "Pune"


def test_grounding_detects_invented_citation():
    g = check_grounding([Claim("x", ["m1"]), Claim("y", ["zz"])], {"m1"})
    assert g.invalid_ids == ["zz"] and g.score < 1.0


def test_suggestions_grounded_and_naive_baseline(system):
    system.remember("My gym membership is CultFit.")
    system.add_goal("Plan a fitness routine around my gym")
    s = system.suggest()[0]
    assert s.grounded and s.cited_ids
    n = NaiveSuggester(system.agent.retrievers["hybrid"], HeuristicLLM()).suggest("fitness gym")
    assert n.claims


def test_cli_commands(system):
    assert "Remembered" in handle(system, "My city of residence is Pune.")
    assert "Pune" in handle(system, "What is my city of residence?")
    assert "[m1]" in handle(system, "/memories")
    assert "Added goal" in handle(system, "/goal be fit")
    assert "Deleted" in handle(system, "/forget m1")
    assert "No memory" in handle(system, "/forget m1")
    assert "Unknown" in handle(system, "/bogus")
    assert handle(system, "") == ""


def test_web_api(system):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(system))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        assert post("/api/remember", {"text": "My city of residence is Pune."})["parse_ok"]
        assert post("/api/ask", {"question": "What is my city of residence?"})["answer"] == "Pune"
        assert json.loads(urllib.request.urlopen(base + "/api/memories").read())["memories"]
        assert b"<title>" in urllib.request.urlopen(base + "/").read()
    finally:
        srv.shutdown()


def test_guard_stops_on_empty_replies():
    from memcore.guard import GuardedLLM
    class Dead:
        model = "m"
        def complete_json(self, p): return "{}"
    g = GuardedLLM(Dead())
    assert g.model == "m"
    with pytest.raises(RuntimeError):
        for _ in range(5):
            g.complete_json("x")


def test_guard_flags_non_json_error_text():
    from memcore.guard import GuardedLLM
    class Bad:
        def complete_json(self, p): return "Unauthorized: bad credentials"
    g = GuardedLLM(Bad())
    with pytest.raises(RuntimeError) as e:
        for _ in range(5):
            g.complete_json("x")
    assert "Unauthorized" in str(e.value)
    ok = GuardedLLM(type("G", (), {"complete_json": lambda s, p: '{"candidates": []}'})())
    assert ok.complete_json("x") == '{"candidates": []}'
