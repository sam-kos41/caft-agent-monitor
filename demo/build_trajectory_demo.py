"""Build demo/index.html — one real SWE-agent trajectory with each
human-factors / agent-native construct's per-step signal overlaid.

Faithful + reproducible: the trajectory is selected deterministically
from the frozen sample; every per-step signal is computed with the
SAME frozen logic the committed extractors use (imported, not
re-derived). No analysis runs in the page — data is baked in.

Honest framing baked into the page: this is ONE illustrative
trajectory showing *what the constructs measure*; the validated
result is population-level (2000 trajectories) — see
docs/PROJECT_SNAPSHOT.html.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from agentdiag.adapters.swe_agent import _extract_action
from agentdiag.validation.pilot_tac import (
    _thought, _LEX, _CAT, _WORD,
)
from agentdiag.validation.pilot_error import _ERR

SAMPLE = "/tmp/caft_pilot/sample.jsonl"
# Select by ROW INDEX, not instance_id: instance_id is non-unique
# (the same SWE-bench instance appears ~23x across model scales in the
# frozen sample) — the same lesson as TAC-robustness amendment A1.
ROW_IDX = 237                              # deterministic, reproducible
INSTANCE = "beetbox__beets-3868"           # (label only; row 237 == this)
OBS_PATH = {"open", "cat", "scroll_down", "scroll_up", "goto",
            "head", "tail"}
MUT = {"edit", "create", "insert", "append"}
CAT_COLOR = {"observe": "#58a6ff", "search": "#3fb950",
             "mutate": "#d29922", "verify": "#a371f7",
             "submit": "#8b97a7", "other": "#6e7681"}


def _base(p):
    return (os.path.basename(str(p).rstrip("/")) or str(p)) if p else ""


def _target_tok(args: str) -> str:
    m = re.search(r'["\']([^"\']+)["\']', args or "")
    raw = m.group(1) if m else (args.split(None, 1)[0]
                                if (args or "").strip() else "")
    b = _base(raw)
    mt = re.search(r"[A-Za-z_][A-Za-z0-9_]+", b)
    return (mt.group(0) if mt else b).lower()


def build():
    rows = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]
    row = rows[ROW_IDX]
    assert row.get("instance_id") == INSTANCE, (
        f"row {ROW_IDX} is {row.get('instance_id')}, expected {INSTANCE}")

    traj = row.get("trajectory") or []
    steps = []
    seen_paths: set[str] = set()
    # pre-index: for each ai step, was the immediately following
    # observation an error, and did the next action change?
    seq = []  # (kind, payload, raw_turn_index)
    for ti, t in enumerate(traj):
        r = t.get("role")
        if r in ("ai", "assistant"):
            a = _extract_action(t.get("text", ""))
            if a:
                seq.append(("act", (a[0], a[1], t.get("text", "")), ti))
        elif r in ("user", "tool") and ti > 1:
            seq.append(("err" if _ERR.search(t.get("text", "") or "")
                        else "obs", t.get("text", "") or "", ti))

    acts = [x for x in seq if x[0] == "act"]
    for n, (_, (verb, args, ai_text), ti) in enumerate(acts):
        v = verb.lower()
        cat = _CAT.get(v, "other")
        tok = _target_tok(args)
        thought = _thought(ai_text)
        tw = set(_WORD.findall(thought.lower()))

        # Thought-Action Coherence (frozen pilot_tac rule)
        verb_align = 1 if (cat != "other" and (tw & _LEX[cat])) else 0
        if tok:
            target_present = 1 if tok in thought.lower() else 0
        else:
            target_present = verb_align
        tac = 0.5 * verb_align + 0.5 * target_present

        # Perception (frozen pilot_sa rule): mutate target seen earlier
        perception = None
        if cat == "mutate":
            perception = bool(tok and tok in seen_paths)
        if v in OBS_PATH and tok:
            seen_paths.add(tok)

        # Error / recovery: is the next seq element an error obs, and
        # does the following action change (strategy change)?
        errored = False
        recovered = None
        gi = seq.index(("act", (verb, args, ai_text), ti))
        if gi + 1 < len(seq) and seq[gi + 1][0] == "err":
            errored = True
            # next action after the error
            for j in range(gi + 2, len(seq)):
                if seq[j][0] == "act":
                    nv = (seq[j][1][0], seq[j][1][1])
                    recovered = (nv != (verb, args))
                    break

        steps.append({
            "i": n + 1,
            "verb": verb, "cat": cat,
            "target": tok or "",
            "thought": " ".join(thought.split())[:240],
            "wl": len(thought),
            "tac": tac,
            "perception": perception,   # True/False/None
            "errored": errored,
            "recovered": recovered,     # True/False/None
        })

    maxwl = max((s["wl"] for s in steps), default=1) or 1
    for s in steps:
        s["wl_norm"] = round(s["wl"] / maxwl, 3)

    data = {
        "instance": INSTANCE,
        "row_idx": ROW_IDX,
        "model": row.get("model_name"),
        "resolved": bool(row.get("target")),
        "n_steps": len(steps),
        "steps": steps,
    }
    html = _HTML.replace("/*DATA*/", json.dumps(data))
    out = Path(__file__).parent / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(steps)} steps, instance {INSTANCE})")


_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human-factors constructs over a real agent trajectory</title>
<style>
:root{--bg:#0d1017;--card:#161b25;--line:#272e3b;--ink:#e6edf3;
--dim:#8b97a7;--ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);
color:var(--ink);font:14px/1.55 -apple-system,system-ui,Segoe UI,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:34px 18px 70px}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13.5px;margin:0 0 14px}
.banner{background:#0f2236;border:1px solid #21456e;border-radius:8px;
padding:11px 15px;font-size:13px;color:#cfe3fb;margin:0 0 20px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;
color:var(--dim);margin:0 0 14px}
.legend b{color:var(--ink)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;
background:var(--card)}
.grid{display:grid;grid-auto-flow:column;min-width:max-content}
.col{width:54px;border-right:1px solid #20262f;position:relative}
.col:hover{background:#1b2230}
.rowlab{position:sticky;left:0;z-index:2;background:var(--card);
border-right:2px solid var(--line);width:118px;min-width:118px}
.cell{height:46px;border-bottom:1px solid #20262f;display:flex;
align-items:center;justify-content:center;font-size:11px}
.rowlab .cell{justify-content:flex-start;padding:0 12px;color:var(--dim);
font-size:11.5px;font-weight:600}
.act{font-size:10px;color:#0d1017;font-weight:700;border-radius:4px;
padding:3px 4px;max-width:46px;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}
.wlbar{width:14px;background:linear-gradient(#f85149,#d29922,#3fb950);
border-radius:2px;align-self:flex-end;margin-bottom:5px}
.mk{font-size:13px;font-weight:700}
.dotn{color:#3b424d}
.tip{position:fixed;max-width:340px;background:#0b0e14;
border:1px solid #30363d;border-radius:8px;padding:11px 13px;
font-size:12px;color:var(--ink);pointer-events:none;opacity:0;
transition:opacity .1s;z-index:10;box-shadow:0 8px 30px #000a}
.tip b{color:var(--accent)} .tip .t{color:var(--dim);font-style:italic}
.tip .cs{margin-top:7px;border-top:1px solid #222;padding-top:6px}
.tip .cs div{margin:3px 0}
footer{margin-top:26px;color:var(--dim);font-size:12.5px;
border-top:1px solid var(--line);padding-top:14px}
a{color:var(--accent)}
.idx{position:sticky;left:0;background:var(--card);
border-right:2px solid var(--line)}
</style></head><body><div class="wrap">
<h1>What each construct measures — on one real agent trajectory</h1>
<p class="sub" id="meta"></p>
<div class="banner">One real SWE-agent trajectory, shown to make the
constructs concrete. <b>The validated result is population-level</b>
(2,000 trajectories): all four constructs predict task outcome on
their own, yet none add predictive value beyond information-theoretic
<i>action-sequence structure</i> — see
<a href="../docs/PROJECT_SNAPSHOT.html">the project snapshot</a>.
A single trajectory illustrates the measurements; it is not evidence
for per-instance detection (that regime was tested and failed).</div>
<div class="legend">
<span><b>Action ribbon</b> = the shape IT structure measures</span>
<span><b>Workload</b> = reasoning length (Sweller/Wickens)</span>
<span><b>Perception</b> = read file before editing it? (Endsley L1)</span>
<span><b>Error/recovery</b> = errored → strategy change? (Reason)</span>
<span><b>Thought–Action</b> = stated intent matches action?</span>
</div>
<div class="scroll"><div class="grid" id="grid"></div></div>
<div class="tip" id="tip"></div>
<footer>Trajectory <code id="inst"></code> = frozen-sample row
<code id="ridx"></code> (selected by position; instance ids are
non-unique across model scales). Per-step signals computed with the
committed extractors' frozen logic, reproducible on
<code>construct-validation-pivot</code>. No analysis runs in this
page.</footer>
</div>
<script>
const D=/*DATA*/;
document.getElementById('meta').textContent=
 `instance ${D.instance} · model ${D.model} · `+
 `${D.n_steps} action steps · outcome: ${D.resolved?'resolved':'not resolved'}`;
document.getElementById('inst').textContent=D.instance;
document.getElementById('ridx').textContent=D.row_idx;
const ROWS=[
 {k:'idx',  lab:'step'},
 {k:'act',  lab:'action (IT ribbon)'},
 {k:'wl',   lab:'workload'},
 {k:'perc', lab:'perception'},
 {k:'err',  lab:'error / recovery'},
 {k:'tac',  lab:'thought–action'}];
const CC={observe:'#58a6ff',search:'#3fb950',mutate:'#d29922',
verify:'#a371f7',submit:'#8b97a7',other:'#6e7681'};
const g=document.getElementById('grid');
// label column
const lc=document.createElement('div');lc.className='col rowlab';
ROWS.forEach(r=>{const c=document.createElement('div');c.className='cell';
c.textContent=r.lab;lc.appendChild(c);});g.appendChild(lc);
// step columns
D.steps.forEach(s=>{
 const col=document.createElement('div');col.className='col';
 col.dataset.i=s.i;
 ROWS.forEach(r=>{
  const c=document.createElement('div');c.className='cell';
  if(r.k==='idx'){c.textContent=s.i;c.style.color='#6e7681';}
  else if(r.k==='act'){const a=document.createElement('div');
   a.className='act';a.style.background=CC[s.cat]||CC.other;
   a.textContent=s.verb;c.appendChild(a);}
  else if(r.k==='wl'){const b=document.createElement('div');
   b.className='wlbar';b.style.height=(6+s.wl_norm*32)+'px';
   c.appendChild(b);}
  else if(r.k==='perc'){
   if(s.perception===null){c.innerHTML='<span class=dotn>·</span>';}
   else{c.innerHTML=s.perception?
    '<span class=mk style="color:var(--ok)">✓</span>':
    '<span class=mk style="color:var(--bad)">✗</span>';}}
  else if(r.k==='err'){
   if(!s.errored){c.innerHTML='<span class=dotn>·</span>';}
   else{c.innerHTML=(s.recovered===true)?
    '<span class=mk style="color:var(--warn)">✗→↻</span>':
    '<span class=mk style="color:var(--bad)">✗</span>';}}
  else if(r.k==='tac'){const v=s.tac;
   const col2=v>=1?'var(--ok)':v>=0.5?'var(--warn)':'var(--bad)';
   c.innerHTML=`<span class=mk style="color:${col2}">`+
    (v>=1?'●':v>=0.5?'◐':'○')+'</span>';}
  col.appendChild(c);});
 col.addEventListener('mousemove',e=>showTip(e,s));
 col.addEventListener('mouseleave',hideTip);
 g.appendChild(col);
});
const tip=document.getElementById('tip');
function showTip(e,s){
 const pc=s.perception===null?'n/a (not an edit)':
  (s.perception?'edited a file it had read ✓':
   'edited a file it had NOT read ✗');
 const ec=!s.errored?'no error after this action':
  (s.recovered===true?'errored, then changed strategy (recovery)':
   s.recovered===false?'errored, then repeated the same action':
   'errored (end of trajectory)');
 const tc=s.tac>=1?'intent matches action ●':
  s.tac>=0.5?'partial match ◐':'intent does not match ○';
 tip.innerHTML=`<b>step ${s.i} · ${s.verb} ${s.target||''}</b>`+
  `<div class="t">“${s.thought||'(no reasoning text)'}”</div>`+
  `<div class="cs">`+
  `<div><b>IT</b>: action type “${s.cat}” — one token in the sequence IT scores</div>`+
  `<div><b>Workload</b>: ${s.wl} chars of reasoning</div>`+
  `<div><b>Perception</b>: ${pc}</div>`+
  `<div><b>Error/recovery</b>: ${ec}</div>`+
  `<div><b>Thought–Action</b>: ${tc}</div></div>`;
 tip.style.opacity=1;
 const x=Math.min(e.clientX+16,window.innerWidth-356);
 const y=Math.min(e.clientY+16,window.innerHeight-220);
 tip.style.left=x+'px';tip.style.top=y+'px';
}
function hideTip(){tip.style.opacity=0;}
</script></body></html>"""


if __name__ == "__main__":
    build()
