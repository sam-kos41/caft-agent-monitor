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
    seen_cats: set[str] = set()   # for ribbon-redundancy display only
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
        # Ribbon redundancy: has this action type already occurred?
        # This is a *rendering* of the shown sequence's own
        # compressibility (what MI/compression score) — not a new
        # metric, just the visible signature of structure.
        repeat = cat in seen_cats
        seen_cats.add(cat)
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
            "repeat": repeat,
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
<title>Four ways to measure an AI agent — and why each reduces to the action sequence</title>
<style>
/* Palette — Dusk Blue / White / Apricot Cream / Sandy Brown /
   Pumpkin Spice (light theme; dusk blue = structure & text,
   warm ramp = accents & severity). */
:root{--bg:#ffffff;--card:#fdf3e4;--line:#ead9bc;--ink:#3a4170;
--dim:#8189ad;--accent:#4e598c;--dusk:#4e598c;--apricot:#f9c784;
--sandy:#fcaf58;--pumpkin:#ff8c42;
--ok:#4e598c;--warn:#fcaf58;--bad:#ff8c42}
*{box-sizing:border-box}body{margin:0;background:var(--bg);
color:var(--ink);font:14px/1.55 -apple-system,system-ui,Segoe UI,sans-serif}
.wrap{max-width:1160px;margin:0 auto;padding:34px 18px 70px}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.01em;
color:var(--dusk)}
.sub{color:var(--dim);font-size:13.5px;margin:0 0 14px}
.legtitle{font-size:14px;font-weight:700;color:var(--dusk);
margin:6px 0 3px}
.legnote{font-size:12px;color:var(--dim);margin:0 0 8px;
max-width:780px;font-style:italic}
.legend{display:flex;flex-direction:column;gap:4px;font-size:12px;
color:var(--dim);margin:2px 0 16px;max-width:780px}
.legend b{color:var(--dusk)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;
background:#fff}
.grid{display:grid;grid-auto-flow:column;min-width:max-content}
.col{width:54px;border-right:1px solid #f0e6d3;position:relative}
.rowlab{position:sticky;left:0;z-index:2;background:var(--card);
border-right:2px solid var(--dusk);width:102px;min-width:102px}
.cell{height:46px;border-bottom:1px solid #f0e6d3;display:flex;
align-items:center;justify-content:center;font-size:11px}
.rowlab .cell{justify-content:flex-start;padding:0 9px;
color:var(--dusk);font-size:11.5px;font-weight:600}
.wlbar{width:14px;background:linear-gradient(180deg,var(--pumpkin),
var(--sandy),var(--apricot));border-radius:2px;align-self:flex-end;
margin-bottom:5px}
.mk{font-size:13px;font-weight:700}
.dotn{color:#cdd0e0}
footer{margin-top:26px;color:var(--dim);font-size:12.5px;
border-top:1px solid var(--line);padding-top:14px}
a{color:var(--pumpkin)}
.idx{position:sticky;left:0;background:var(--card);
border-right:2px solid var(--dusk)}
.ctx{font-size:13px;color:var(--ink);margin:0 0 16px;max-width:760px}
.ctx b{color:var(--dusk)}
.badge{display:inline-block;font-size:11px;font-weight:700;
padding:2px 9px;border-radius:999px;background:var(--dusk);
color:#fff;letter-spacing:.02em;vertical-align:middle}
/* IT ribbon is the load-bearing lane — render it dominant. */
.cell.it{height:70px;flex-direction:column;gap:4px;
border-top:2px solid var(--dusk);
border-bottom:2px solid var(--dusk);background:#fbf3e6}
.rowlab .cell.it{flex-direction:column;justify-content:center;
align-items:flex-start;gap:0;color:var(--dusk);font-weight:800;
font-size:10px;line-height:1.25}
.act{font-size:10px;font-weight:700;border-radius:4px;
padding:4px 6px;text-align:center;white-space:nowrap;
box-shadow:0 1px 3px rgba(78,89,140,.18)}
.rep{font-size:9px;color:var(--dim);margin-top:2px;font-weight:700}
.new{font-size:9px;color:var(--pumpkin);margin-top:2px;font-weight:800}
.find{max-width:760px;margin:26px 0 4px}
.find h2{font-size:15px;margin:0 0 4px;color:var(--dusk)}
.find p{font-size:12.5px;color:var(--dim);margin:0 0 10px}
table.f{border-collapse:collapse;width:100%;font-size:12.5px}
table.f th,table.f td{border:1px solid var(--line);padding:6px 9px;
text-align:left}
table.f th{background:var(--card);color:var(--dusk);font-weight:700}
table.f td.n{text-align:right;font-variant-numeric:tabular-nums}
.v{color:var(--dusk);font-weight:700}
.x{color:var(--pumpkin);font-weight:700}
table.f tr.lb td{background:#fbf3e6}
.focus{max-width:880px;margin:24px 0 0}
.focus h2{font-size:15px;margin:0 0 8px;color:var(--dusk)}
.fcards{display:flex;gap:12px;flex-wrap:wrap}
.fc{flex:1;min-width:240px;background:var(--card);
border:1px solid var(--line);border-left:4px solid var(--dusk);
border-radius:8px;padding:12px 14px}
.fc .h{font-size:12px;font-weight:800;color:var(--dusk);
margin:0 0 5px}
.fc .q{font-size:12px;font-style:italic;color:var(--ink);
margin:0 0 8px}
.fc .r{font-size:11.5px;color:var(--dim);margin:3px 0}
.fc .r b{color:var(--dusk)}
</style></head><body><div class="wrap">
<h1>Four ways to measure an AI agent — and why each reduces to the
action sequence</h1>
<p class="sub" id="meta"></p>
<p class="ctx">This trajectory <b>succeeded</b>. The four constructs vary
all the way through it regardless of outcome — which is exactly why
the interesting question is discriminant: does any of them add
anything <i>beyond the action sequence itself</i>? On 2,000
trajectories, none do. The table below the trajectory is the finding;
the trajectory shows what each lens actually measures, step by step.</p>
<p class="legtitle">How I operationalize the human-factors constructs</p>
<p class="legnote">These are <b>my</b> operational definitions —
each classical human-factors construct (and one agent-native one)
reduced to something computable from an agent trace. The parenthetical
names are the source theory; the symbols are the concrete trace
measurement that stands in for it here, not the theory itself.</p>
<div class="legend">
<span><b>IT (action sequence)</b> — ● new action type · ↻ repeat
(redundancy = what compression / MI score)</span>
<span><b>Workload</b> — character count of reasoning (Sweller / Wickens)</span>
<span><b>Perception</b> — ✓ read file first · ✗ edited unread file</span>
<span><b>Error / recovery</b> — ✗ errored · ✗→↻ errored then changed
strategy (Reason)</span>
<span><b>Thought–Action</b> — ● intent matches · ◐ partial · ○ no match</span>
</div>
<div class="scroll"><div class="grid" id="grid"></div></div>

<div class="focus">
<h2>Same trajectory, three moments</h2>
<div class="fcards" id="fcards"></div>
</div>

<div class="find">
<h2>The finding (population-level, N = 2,000 — not this one trajectory)</h2>
<p>Each construct predicts task outcome <i>on its own</i>. Not one
adds predictive value <i>beyond</i> the information-theoretic action
sequence (locked decision rule: does it beat baseline + IT?). The
agent-native lens is the sharpest case — least correlated with IT
(0.43) yet still adds essentially zero.</p>
<table class="f">
<tr><th>Construct</th><th>Source</th><th class="n">Alone (H1 AUC)</th>
<th class="n">Beyond IT (ΔAUC)</th><th class="n">corr w/ IT</th>
<th>Verdict</th></tr>
<tr class="lb"><td>IT behavioral structure</td>
<td>information theory</td><td class="n v">0.75</td>
<td class="n">— (reference)</td><td class="n">—</td>
<td class="v">load-bearing</td></tr>
<tr><td>Cognitive workload</td><td>Wickens / Sweller / NASA-TLX</td>
<td class="n">0.70</td><td class="n x">+0.006 (CI∋0)</td>
<td class="n">0.74</td><td class="x">reduces to IT</td></tr>
<tr><td>Situation awareness</td><td>Endsley</td>
<td class="n">0.65</td><td class="n x">+0.003 (CI∋0)</td>
<td class="n">0.64</td><td class="x">reduces to IT</td></tr>
<tr><td>Error recovery</td><td>Reason / Hollnagel</td>
<td class="n">0.71</td><td class="n x">+0.004 (CI∋0)</td>
<td class="n">0.75</td><td class="x">reduces to IT</td></tr>
<tr><td>Thought–action coherence</td><td>agent-native</td>
<td class="n">0.67</td><td class="n x">−0.0002 (CI∋0)</td>
<td class="n">0.43</td><td class="x">reduces to IT</td></tr>
</table>
<p>Holds under a graded outcome too (Generalization G) — not a
binarization artifact. Full arc &amp; evidence ledger:
<a href="../docs/PROJECT_SNAPSHOT.html">the project snapshot</a>.</p>
</div>

<footer>A single trajectory illustrates the measurements; it is not
evidence for per-instance detection (that regime was tested and
failed). Trajectory <code id="inst"></code> = frozen-sample row
<code id="ridx"></code> (selected by position; instance ids are
non-unique across model scales). Per-step signals computed with the
committed extractors' frozen logic, reproducible on
<code>construct-validation-pivot</code>.</footer>
</div>
<script>
const D=/*DATA*/;
document.getElementById('meta').innerHTML=
 `instance ${D.instance} · model ${D.model} · `+
 `${D.n_steps} action steps · outcome: `+
 `<span class="badge">${D.resolved?'RESOLVED':'NOT RESOLVED'}</span>`;
document.getElementById('inst').textContent=D.instance;
document.getElementById('ridx').textContent=D.row_idx;
const ROWS=[
 {k:'idx',  lab:'step'},
 {k:'act',  lab:'ACTION SEQUENCE ▸ IT (load-bearing)'},
 {k:'wl',   lab:'workload'},
 {k:'perc', lab:'perception'},
 {k:'err',  lab:'error / recovery'},
 {k:'tac',  lab:'thought–action'}];
// action-ribbon palette (Dusk Blue / warm ramp) + readable text
const CC={observe:'#4e598c',search:'#7e88b4',mutate:'#ff8c42',
verify:'#fcaf58',submit:'#f9c784',other:'#c9ccdd'};
const CT={observe:'#fff',search:'#fff',mutate:'#fff',
verify:'#3a2e12',submit:'#5a3c14',other:'#3a4170'};
const g=document.getElementById('grid');
// label column
const lc=document.createElement('div');lc.className='col rowlab';
ROWS.forEach(r=>{const c=document.createElement('div');
c.className='cell'+(r.k==='act'?' it':'');
c.textContent=r.lab;lc.appendChild(c);});g.appendChild(lc);
// step columns
D.steps.forEach(s=>{
 const col=document.createElement('div');col.className='col';
 col.dataset.i=s.i;
 ROWS.forEach(r=>{
  const c=document.createElement('div');
  c.className='cell'+(r.k==='act'?' it':'');
  if(r.k==='idx'){c.textContent=s.i;c.style.color='#6e7681';}
  else if(r.k==='act'){const a=document.createElement('div');
   a.className='act';a.style.background=CC[s.cat]||CC.other;
   a.style.color=CT[s.cat]||CT.other;
   a.textContent=s.cat;c.appendChild(a);
   const rd=document.createElement('div');
   rd.className=s.repeat?'rep':'new';
   rd.textContent=s.repeat?'↻ rep':'● new';
   c.appendChild(rd);}
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
 g.appendChild(col);
});
function reads(s){
 const pc=s.perception===null?'n/a (not an edit)':
  (s.perception?'edited a file it had read ✓':
   'edited a file it had NOT read ✗');
 const ec=!s.errored?'no error after this action':
  (s.recovered===true?'errored, then changed strategy (recovery)':
   s.recovered===false?'errored, then repeated the same action':
   'errored (end of trajectory)');
 const tc=s.tac>=1?'intent matches action ●':
  s.tac>=0.5?'partial match ◐':'intent does not match ○';
 return [
  ['IT', `action “${s.cat}”, `+
   (s.repeat?'a repeat (redundancy IT scores)':'first of its type')],
  ['Workload', `${s.wl} chars of reasoning`],
  ['Perception', pc],
  ['Error/recovery', ec],
  ['Thought–Action', tc]];
}
// Three illustrative steps: executes well / struggles / heaviest.
(function(){
 const byHi=[...D.steps].sort((a,b)=>b.tac-a.tac||b.wl-a.wl);
 const hi=byHi[0];
 const lo=[...D.steps].sort((a,b)=>a.tac-b.tac||b.wl-a.wl)[0];
 const mid=D.steps.filter(s=>s.errored)
   .find(s=>s.i!==hi.i&&s.i!==lo.i)
  ||[...D.steps].sort((a,b)=>b.wl-a.wl)
   .find(s=>s.i!==hi.i&&s.i!==lo.i);
 const picks=[
  ['Executing well — highest thought–action coherence',hi],
  [(mid&&mid.errored?'Hit an error — recovery behavior':
    'Heaviest reasoning step'),mid],
  ['Struggling — lowest thought–action coherence',lo]]
  .filter(p=>p[1]);
 const seen=new Set();
 document.getElementById('fcards').innerHTML=picks
  .filter(p=>!seen.has(p[1].i)&&seen.add(p[1].i))
  .map(([h,s])=>`<div class="fc"><div class="h">${h}</div>`+
   `<div class="q">step ${s.i} · ${s.verb} ${s.target||''} — `+
   `“${s.thought||'(no reasoning text)'}”</div>`+
   reads(s).map(([k,v])=>`<div class="r"><b>${k}</b>: ${v}</div>`)
    .join('')+`</div>`).join('');
})();
</script></body></html>"""


if __name__ == "__main__":
    build()
