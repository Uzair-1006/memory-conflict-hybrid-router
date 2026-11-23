"""Local web UI.  python -m app.web --data-dir data --llm demo  -> http://127.0.0.1:8000"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli import make_llm
from app.system import MemorySystem

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Memory Agent</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:15px system-ui;max-width:820px;margin:24px auto;padding:0 16px;color:#222}
section{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}
input{width:70%;padding:6px}button{padding:6px 10px}pre{white-space:pre-wrap;background:#f6f6f6;padding:8px;border-radius:6px}
li{margin:4px 0}.warn{color:#b45309}small{color:#666}</style></head><body>
<h2>Personal memory agent</h2>
<section><b>Remember</b><br><input id="t" placeholder="My city of residence is Pune."> <button onclick="rem()">Save</button><pre id="r"></pre></section>
<section><b>Ask</b><br><input id="q" placeholder="What is my city of residence?"> <button onclick="ask()">Ask</button><pre id="a"></pre></section>
<section><b>Goals</b><br><input id="g" placeholder="Plan a fitness routine"> <button onclick="goal()">Add goal</button>
<button onclick="sug()">Suggest</button><ul id="gl"></ul><div id="s"></div></section>
<section><b>Memories</b> <button onclick="mem()">Refresh</button><ul id="m"></ul></section>
<script>
const $=i=>document.getElementById(i);
async function api(p,b){const r=await fetch(p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{});return r.json()}
function li(t){const e=document.createElement('li');e.textContent=t;return e}
async function rem(){const d=await api('/api/remember',{text:$('t').value});
 $('r').textContent=!d.parse_ok?'Could not read that as a memory.':(d.records.length?'Remembered: '+d.records.map(x=>'['+x.id+'] '+x.text).join(' | '):'No personal fact found.');$('t').value='';mem()}
async function ask(){const d=await api('/api/ask',{question:$('q').value});
 $('a').textContent=d.abstained?d.message:d.answer+'\\n'+d.evidence.map(e=>'evidence ['+e.id+'] '+e.text).join('\\n')+'\\nhow: '+d.how}
async function mem(){const d=await api('/api/memories');const u=$('m');u.textContent='';
 d.memories.forEach(x=>{const l=li('['+x.id+'] '+x.when+'  '+x.text+' ');const b=document.createElement('button');b.textContent='forget';
 b.onclick=async()=>{await api('/api/forget',{id:x.id});mem()};l.appendChild(b);u.appendChild(l)})}
async function goals(){const d=await api('/api/goals');const u=$('gl');u.textContent='';d.goals.forEach(g=>u.appendChild(li(g.id+': '+g.text)))}
async function goal(){await api('/api/goal',{text:$('g').value});$('g').value='';goals()}
async function sug(){const d=await api('/api/suggest',{});const s=$('s');s.textContent='';
 d.suggestions.forEach(x=>{const p=document.createElement('pre');
 p.textContent='Goal: '+x.goal+'\\n'+(x.suggestion||('Need more info: '+(x.clarification||'none')))+(x.suggestion&&!x.grounded?'\\n(warning: not fully grounded)':'')
 +x.claims.map(c=>'\\n - '+c.text+' <- '+c.evidence.map(e=>'['+e.id+']').join(', ')).join('');s.appendChild(p)})}
mem();goals();
</script></body></html>"""


def make_handler(system: MemorySystem):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200, ctype="application/json"):
            body = (obj if isinstance(obj, str) else json.dumps(obj)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(PAGE, ctype="text/html")
            if self.path == "/api/memories":
                return self._send({"memories": system.memories()})
            if self.path == "/api/goals":
                return self._send({"goals": [{"id": g.goal_id, "text": g.text} for g in system.goals.list()]})
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                d = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/remember":
                    r = system.remember(str(d.get("text", "")))
                    return self._send({"parse_ok": r.parse_ok, "records": [{"id": x.fact_id, "text": x.text} for x in r.records]})
                if self.path == "/api/ask":
                    return self._send(system.answer_view(system.ask(str(d.get("question", "")), d.get("context"))))
                if self.path == "/api/forget":
                    return self._send({"ok": system.forget(str(d.get("id", "")))})
                if self.path == "/api/correct":
                    return self._send({"ok": system.correct(str(d.get("id", "")), str(d.get("text", "")))})
                if self.path == "/api/goal":
                    g = system.add_goal(str(d.get("text", "")))
                    return self._send({"id": g.goal_id, "text": g.text})
                if self.path == "/api/suggest":
                    return self._send({"suggestions": [system.suggestion_view(s) for s in system.suggest(d.get("goal_id"))]})
                self._send({"error": "not found"}, 404)
            except Exception as e:  # keep the server alive; report to the caller
                self._send({"error": f"{type(e).__name__}: {e}"}, 500)
    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--llm", choices=["demo", "openai"], default="openai")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--confidence", type=float, default=0.0)
    a = ap.parse_args(argv)
    try:
        llm = make_llm(a.llm)
    except RuntimeError as e:
        sys.exit(f"{e}\nTip: run with --llm demo to try the app without an API key.")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(MemorySystem(a.data_dir, llm, a.confidence)))
    print(f"Open http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
