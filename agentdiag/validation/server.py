"""FastAPI server for the human-rating UI.

Single-page interface that lets a human rater:
  - See the list of sessions in the corpus, with rating coverage
  - Open one session: shows digest + Ollama rating + CAFT rating
  - Submit their own ratings (4 Likert + 1 categorical + reasoning)
  - View the live agreement report

The UI is intentionally low-tech (one inline HTML page) so it stays
dependency-light and matches the visualize.py / dashboard pattern.

Run with:
    python -m agentdiag.validation.server \
        --corpus path/to/sessions/ \
        --ledger validation_ledger.jsonl \
        --port 8090
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from agentdiag.validation.digest import (
    SessionDigest, build_digest, DIMENSIONS, LIKERT_DIMS, HEALTH_LABELS,
    SCALE_ANCHORS, SCALE_NOTES, HEALTH_ANCHORS,
)
from agentdiag.validation.ledger import Ledger, Rating
from agentdiag.validation.rate_caft import rate_with_caft
from agentdiag.validation.rate_ollama import (
    rate_with_ollama, is_ollama_available, DEFAULT_MODEL,
)
from agentdiag.validation.report import write_report


def _require_fastapi():
    if not HAS_FASTAPI:
        raise ImportError(
            "Validation server requires fastapi+uvicorn. Install with:\n"
            "    pip install caft[dashboard]"
        )


_corpus_root: Optional[Path] = None
_ledger: Optional[Ledger] = None
_human_id: str = "default"
_ollama_model: str = DEFAULT_MODEL
_digest_cache: dict[str, SessionDigest] = {}


def _list_sessions() -> list[Path]:
    if _corpus_root is None:
        return []
    return sorted(_corpus_root.glob("*.jsonl"))


def _get_digest(session_id: str) -> SessionDigest:
    if session_id in _digest_cache:
        return _digest_cache[session_id]
    matches = [p for p in _list_sessions() if p.stem == session_id]
    if not matches:
        raise HTTPException(404, f"session {session_id} not found in corpus")
    d = build_digest(matches[0])
    _digest_cache[session_id] = d
    return d


def _make_app():
    _require_fastapi()
    app = FastAPI(title="CAFT validation rater")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/sessions")
    async def list_sessions():
        if _ledger is None:
            return JSONResponse({"sessions": []})
        latest = _ledger.latest()
        out = []
        for p in _list_sessions():
            sess = p.stem
            raters_done = set()
            for (s, rt, rid, _dim) in latest:
                if s == sess:
                    raters_done.add(f"{rt}:{rid}")
            out.append({
                "session_id": sess,
                "filename": p.name,
                "size_mb": round(p.stat().st_size / 1e6, 2),
                "rated_by": sorted(raters_done),
            })
        return JSONResponse({"sessions": out, "human_id": _human_id,
                             "ollama_model": _ollama_model,
                             "dimensions": list(DIMENSIONS),
                             "likert_dims": list(LIKERT_DIMS),
                             "health_labels": list(HEALTH_LABELS),
                             "scale_anchors": SCALE_ANCHORS,
                             "scale_notes": SCALE_NOTES,
                             "health_anchors": HEALTH_ANCHORS})

    @app.get("/api/session/{session_id}")
    async def get_session(session_id: str):
        digest = _get_digest(session_id)
        existing = _ledger.session_ratings(session_id) if _ledger else []
        return JSONResponse({
            "digest": digest.to_dict(),
            "digest_text": digest.to_text(),
            "ratings": existing,
        })

    @app.post("/api/session/{session_id}/auto_rate")
    async def auto_rate(session_id: str):
        """Run CAFT rater (always) + Ollama rater (if reachable)."""
        digest = _get_digest(session_id)
        added: list[Rating] = []
        try:
            added.extend(rate_with_caft(digest))
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"CAFT: {e}"}, status_code=500)
        try:
            from agentdiag.validation.signals import rate_with_signals
            added.extend(rate_with_signals(digest.source_path))
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"signals: {e}"}, status_code=500)
        ollama_ok = is_ollama_available()
        ollama_err = None
        if ollama_ok:
            try:
                added.extend(rate_with_ollama(digest, model=_ollama_model))
            except Exception as e:
                ollama_err = str(e)
        if _ledger is not None:
            _ledger.append_many(added)
        return JSONResponse({
            "ok": True,
            "added": len(added),
            "ollama_ran": ollama_ok and ollama_err is None,
            "ollama_error": ollama_err,
        })

    @app.post("/api/session/{session_id}/rate_human")
    async def rate_human(session_id: str, body: dict):
        if _ledger is None:
            raise HTTPException(500, "no ledger configured")
        reasoning = str(body.get("reasoning", ""))[:500]
        ratings = body.get("ratings", {})
        rows: list[Rating] = []
        for dim in DIMENSIONS:
            entry = ratings.get(dim)
            if entry is None:
                raise HTTPException(400, f"missing dimension {dim}")
            raw_v = entry.get("value")
            conf = entry.get("confidence", "")
            if raw_v is None or raw_v == "abstain":
                value = None        # explicit abstention
                conf = ""           # confidence N/A for abstain
            elif dim in LIKERT_DIMS:
                value = int(raw_v)
            else:
                value = str(raw_v)
            rows.append(Rating(
                session_id=session_id,
                rater_type="human",
                rater_id=_human_id,
                dimension=dim,
                value=value,
                confidence=conf,
                reasoning=reasoning,
            ))
        _ledger.append_many(rows)
        return JSONResponse({"ok": True, "added": len(rows)})

    @app.get("/api/report")
    async def get_report():
        if _ledger is None:
            raise HTTPException(500, "no ledger configured")
        out = _ledger.path.with_suffix(".report.md")
        write_report(_ledger, out)
        return JSONResponse({"path": str(out), "content": out.read_text()})

    return app


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>CAFT validation rater</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif;
         background: #1a1d27; color: #e2e8f0; margin: 0; padding: 0; }
  header { background: #0f1218; padding: 12px 20px; border-bottom: 1px solid #2d3340;
           display: flex; justify-content: space-between; align-items: center; }
  header h1 { margin: 0; font-size: 18px; }
  header .meta { font-size: 12px; color: #94a3b8; }
  main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 50px); }
  .sidebar { background: #14171f; border-right: 1px solid #2d3340; overflow-y: auto; max-height: calc(100vh - 50px); }
  .session-row { padding: 10px 14px; border-bottom: 1px solid #2d3340; cursor: pointer; }
  .session-row:hover { background: #1f242f; }
  .session-row.active { background: #2a3142; }
  .session-row .id { font-family: ui-monospace, monospace; font-size: 12px; color: #cbd5e1; }
  .session-row .meta { font-size: 11px; color: #94a3b8; margin-top: 4px; }
  .session-row .badges { margin-top: 4px; display: flex; gap: 4px; flex-wrap: wrap; }
  .badge { font-size: 10px; padding: 2px 6px; border-radius: 8px; background: #2d3340; color: #94a3b8; }
  .badge.human { background: #16a34a; color: #fff; }
  .badge.ollama { background: #6366f1; color: #fff; }
  .badge.caft { background: #f59e0b; color: #1a1d27; }
  .pane { padding: 20px; overflow-y: auto; max-height: calc(100vh - 50px); }
  .placeholder { color: #94a3b8; padding: 40px; text-align: center; }
  pre.digest { background: #0f1218; border: 1px solid #2d3340; padding: 14px; border-radius: 6px;
               font-size: 12px; line-height: 1.4; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
  table.ratings { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
  table.ratings th, table.ratings td { padding: 6px 10px; border-bottom: 1px solid #2d3340; text-align: left; }
  table.ratings th { background: #1f242f; color: #cbd5e1; font-weight: 600; }
  .form-section { background: #14171f; border: 1px solid #2d3340; padding: 16px; border-radius: 6px; margin-top: 20px; }
  .form-section h3 { margin: 0 0 12px 0; }
  .dim-row { display: grid; grid-template-columns: 200px 1fr; gap: 12px; align-items: center; margin-bottom: 8px; }
  .dim-label { color: #cbd5e1; font-size: 13px; }
  .likert { display: flex; gap: 4px; }
  .likert button { flex: 1; padding: 6px; background: #2d3340; color: #e2e8f0; border: 1px solid #3d4350;
                   border-radius: 4px; cursor: pointer; font-size: 12px; }
  .likert button.selected { background: #3b82f6; border-color: #3b82f6; color: #fff; }
  .health { display: flex; gap: 4px; }
  .health button { flex: 1; padding: 6px; background: #2d3340; color: #e2e8f0; border: 1px solid #3d4350;
                   border-radius: 4px; cursor: pointer; font-size: 12px; }
  .health button.selected.healthy { background: #16a34a; border-color: #16a34a; color: #fff; }
  .health button.selected.degraded { background: #eab308; border-color: #eab308; color: #1a1d27; }
  .health button.selected.pathological { background: #ef4444; border-color: #ef4444; color: #fff; }
  textarea { width: 100%; box-sizing: border-box; background: #0f1218; color: #e2e8f0;
             border: 1px solid #2d3340; padding: 8px; border-radius: 4px; font-size: 13px;
             font-family: inherit; min-height: 60px; }
  .actions { display: flex; gap: 8px; margin-top: 16px; }
  .actions button { padding: 8px 16px; background: #3b82f6; color: #fff; border: none;
                    border-radius: 4px; cursor: pointer; font-size: 13px; }
  .actions button.secondary { background: #2d3340; }
  .actions button:disabled { opacity: 0.5; cursor: not-allowed; }
  .toast { position: fixed; bottom: 20px; right: 20px; background: #16a34a; color: #fff;
           padding: 10px 16px; border-radius: 4px; font-size: 13px; opacity: 0; transition: opacity 0.2s; }
  .toast.show { opacity: 1; }
  .toast.error { background: #ef4444; }
  .dim-summary { font-size: 12px; color: #94a3b8; margin-top: 2px; }
  .dim-block { background: #14171f; border: 1px solid #2d3340; border-radius: 6px;
        padding: 12px 14px; margin-bottom: 14px; }
  .dim-block .dim-label { font-size: 14px; font-weight: 700; color: #e2e8f0;
        text-transform: none; margin-bottom: 2px; }
  .dim-block .pol { font-size: 11px; font-weight: 400; color: #94a3b8; }
  .dim-note { font-size: 12px; color: #fbbf24; background: #2a2410;
        border: 1px solid #4d3f12; padding: 6px 8px; border-radius: 4px; margin: 6px 0; }
  .opts { display: flex; flex-direction: column; gap: 4px; margin: 8px 0; }
  .opt { display: flex; gap: 10px; align-items: baseline; text-align: left;
        padding: 7px 10px; background: #1f242f; color: #cbd5e1;
        border: 1px solid #3d4350; border-radius: 4px; cursor: pointer;
        font-size: 12.5px; line-height: 1.3; }
  .opt:hover { background: #262c39; }
  .opt b { min-width: 70px; color: #e2e8f0; flex-shrink: 0; }
  .opt.selected { background: #1e3a5f; border-color: #3b82f6; color: #fff; }
  .opt.selected b { color: #93c5fd; }
  .opt.selected.healthy { background: #14361f; border-color: #16a34a; }
  .opt.selected.degraded { background: #3a3210; border-color: #eab308; }
  .opt.selected.pathological { background: #3a1414; border-color: #ef4444; }
  .dim-controls { display: flex; flex-direction: column; gap: 6px; }
  .value-row { display: flex; gap: 4px; align-items: center; }
  .value-row .vals { display: flex; gap: 4px; flex: 1; }
  .value-row .vals button { flex: 1; padding: 6px; background: #2d3340; color: #e2e8f0;
        border: 1px solid #3d4350; border-radius: 4px; cursor: pointer; font-size: 12px; }
  .value-row .vals button.selected { background: #3b82f6; border-color: #3b82f6; color: #fff; }
  .value-row .vals button.selected.healthy { background: #16a34a; border-color: #16a34a; }
  .value-row .vals button.selected.degraded { background: #eab308; border-color: #eab308; color: #1a1d27; }
  .value-row .vals button.selected.pathological { background: #ef4444; border-color: #ef4444; }
  .cant-tell { padding: 6px 10px; background: #2d3340; color: #94a3b8;
        border: 1px solid #3d4350; border-radius: 4px; cursor: pointer; font-size: 12px; }
  .cant-tell.selected { background: #6b7280; color: #fff; border-color: #6b7280; }
  .conf-row { display: flex; gap: 4px; align-items: center; font-size: 11px; color: #94a3b8; }
  .conf-row span { width: 70px; }
  .conf-row button { padding: 3px 10px; background: #1f242f; color: #94a3b8;
        border: 1px solid #3d4350; border-radius: 4px; cursor: pointer; font-size: 11px; }
  .conf-row button.selected { background: #0ea5e9; color: #fff; border-color: #0ea5e9; }
  .conf-row.disabled { opacity: 0.35; pointer-events: none; }
</style>
</head>
<body>
<header>
  <h1>CAFT validation rater</h1>
  <div class="meta" id="meta">loading...</div>
</header>
<main>
  <aside class="sidebar" id="sidebar"></aside>
  <section class="pane" id="pane">
    <div class="placeholder">Pick a session from the left.</div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
const DIMS = ["stuck_in_loop","goal_drifted","coherent_progress","user_satisfied","overall_health"];
const LIKERT_DIMS = ["stuck_in_loop","goal_drifted","coherent_progress","user_satisfied"];
const HEALTH = ["healthy","degraded","pathological"];
// Anchors come from the server (single source of truth shared with the
// Ollama prompt and the CAFT rule-mapping). Populated in loadSessions().
let ANCHORS = {}, NOTES = {}, HEALTH_ANCHORS = {};

let CURRENT_SESSION = null;
let CURRENT_RATINGS = {};
let META = {};

function toast(msg, isError) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.toggle('error', !!isError);
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}

async function loadSessions() {
  const r = await fetch('/api/sessions');
  const data = await r.json();
  META = data;
  ANCHORS = data.scale_anchors || {};
  NOTES = data.scale_notes || {};
  HEALTH_ANCHORS = data.health_anchors || {};
  document.getElementById('meta').textContent =
    `human=${data.human_id} | ollama=${data.ollama_model}`;
  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  data.sessions.forEach(s => {
    const div = document.createElement('div');
    div.className = 'session-row';
    div.dataset.id = s.session_id;
    let badges = '';
    s.rated_by.forEach(r => {
      const cls = r.startsWith('human') ? 'human' : r.startsWith('ollama') ? 'ollama' : 'caft';
      badges += `<span class="badge ${cls}">${r.split(':')[0]}</span>`;
    });
    div.innerHTML = `
      <div class="id">${s.session_id.slice(0, 18)}...</div>
      <div class="meta">${s.size_mb} MB</div>
      <div class="badges">${badges}</div>`;
    div.onclick = () => openSession(s.session_id);
    sb.appendChild(div);
  });
}

async function openSession(id) {
  CURRENT_SESSION = id;
  CURRENT_RATINGS = {};
  document.querySelectorAll('.session-row').forEach(el => {
    el.classList.toggle('active', el.dataset.id === id);
  });
  const r = await fetch('/api/session/' + encodeURIComponent(id));
  const data = await r.json();
  renderPane(id, data);
}

function renderPane(id, data) {
  const pane = document.getElementById('pane');
  const ratingsByRater = {};
  (data.ratings || []).forEach(row => {
    const k = `${row.rater_type}:${row.rater_id}`;
    ratingsByRater[k] = ratingsByRater[k] || {};
    ratingsByRater[k][row.dimension] = row;
  });
  let table = '<table class="ratings"><thead><tr><th>Rater</th>';
  DIMS.forEach(d => table += `<th>${d}</th>`);
  table += '</tr></thead><tbody>';
  Object.keys(ratingsByRater).sort().forEach(rater => {
    table += `<tr><td>${rater}</td>`;
    DIMS.forEach(d => {
      const row = ratingsByRater[rater][d];
      let cell = '—';
      if (row) {
        if (row.value === null || row.value === undefined) cell = '<i>abstain</i>';
        else cell = row.value + (row.confidence ? ` <small>(${row.confidence})</small>` : '');
      }
      table += `<td>${cell}</td>`;
    });
    table += '</tr>';
  });
  table += '</tbody></table>';

  let dimRows = '';
  DIMS.forEach(d => {
    const isLikert = LIKERT_DIMS.includes(d);
    let valButtons = '';
    if (isLikert) {
      [1,2,3,4,5].forEach(o => {
        const txt = (ANCHORS[d] && ANCHORS[d][o]) ? ANCHORS[d][o] : '';
        valButtons += `<button class="opt" data-dim="${d}" data-val="${o}" onclick="setValue('${d}','${o}')"><b>${o}</b><span>${escapeHtml(txt)}</span></button>`;
      });
    } else {
      HEALTH.forEach(h => {
        const txt = HEALTH_ANCHORS[h] || '';
        valButtons += `<button class="opt" data-dim="${d}" data-val="${h}" onclick="setValue('${d}','${h}')"><b>${h}</b><span>${escapeHtml(txt)}</span></button>`;
      });
    }
    let confButtons = '';
    ['low','med','high'].forEach(c => {
      confButtons += `<button data-dim="${d}" data-conf="${c}" onclick="setConf('${d}','${c}')">${c}</button>`;
    });
    const note = NOTES[d] ? `<div class="dim-note">⚠ ${escapeHtml(NOTES[d])}</div>` : '';
    const pol = isLikert ? '<span class="pol">(1 = least · 5 = most of the named property)</span>' : '';
    dimRows += `
      <div class="dim-block">
        <div class="dim-label">${d} ${pol}</div>
        ${note}
        <div class="opts" id="vals-${d}">${valButtons}</div>
        <div class="value-row">
          <button class="cant-tell" id="ct-${d}" onclick="setAbstain('${d}')">Can't tell — evidence insufficient</button>
          <div class="conf-row" id="conf-${d}"><span>confidence:</span>${confButtons}</div>
        </div>
      </div>`;
  });

  pane.innerHTML = `
    <h2>${id}</h2>
    <pre class="digest">${escapeHtml(data.digest_text)}</pre>
    <h3>Ratings so far</h3>
    ${Object.keys(ratingsByRater).length ? table : '<p style="color:#94a3b8">No ratings yet — click Auto-rate.</p>'}
    <div class="form-section">
      <h3>Your rating</h3>
      ${dimRows}
      <div class="dim-row">
        <div class="dim-label">reasoning (optional)</div>
        <textarea id="reasoning" placeholder="Why did you rate it this way?"></textarea>
      </div>
      <div class="actions">
        <button class="secondary" onclick="autoRate()">Auto-rate (CAFT + Ollama)</button>
        <button onclick="submitHuman()">Save my rating</button>
        <button class="secondary" onclick="window.open('/api/report','_blank')">View report</button>
      </div>
    </div>`;
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function _ensure(dim) {
  if (!CURRENT_RATINGS[dim]) CURRENT_RATINGS[dim] = {value: undefined, confidence: ''};
  return CURRENT_RATINGS[dim];
}

window.setValue = function(dim, val) {
  const r = _ensure(dim);
  r.value = val;
  if (!r.confidence) r.confidence = 'high';  // sensible default once you commit
  const vals = document.getElementById('vals-' + dim);
  vals.querySelectorAll('button').forEach(b => {
    const m = String(b.dataset.val) === String(val);
    b.classList.toggle('selected', m);
    if (m && !LIKERT_DIMS.includes(dim)) {
      b.classList.remove('healthy','degraded','pathological');
      b.classList.add(val);
    }
  });
  document.getElementById('ct-' + dim).classList.remove('selected');
  const cr = document.getElementById('conf-' + dim);
  cr.classList.remove('disabled');
  _paintConf(dim);
};

window.setAbstain = function(dim) {
  const r = _ensure(dim);
  r.value = 'abstain';
  r.confidence = '';
  const vals = document.getElementById('vals-' + dim);
  vals.querySelectorAll('button').forEach(b => b.classList.remove('selected'));
  document.getElementById('ct-' + dim).classList.add('selected');
  document.getElementById('conf-' + dim).classList.add('disabled');
};

window.setConf = function(dim, c) {
  const r = _ensure(dim);
  if (r.value === undefined || r.value === 'abstain') return;
  r.confidence = c;
  _paintConf(dim);
};

function _paintConf(dim) {
  const r = _ensure(dim);
  document.getElementById('conf-' + dim).querySelectorAll('button').forEach(b => {
    b.classList.toggle('selected', b.dataset.conf === r.confidence);
  });
}

async function autoRate() {
  if (!CURRENT_SESSION) return;
  toast('Running CAFT + Ollama (this may take 10-30s)...');
  const r = await fetch('/api/session/' + encodeURIComponent(CURRENT_SESSION) + '/auto_rate', {
    method: 'POST'
  });
  const data = await r.json();
  if (data.ok) {
    toast(`Added ${data.added} ratings${data.ollama_ran ? ' (incl. Ollama)' : ' (Ollama skipped)'}`);
    if (data.ollama_error) toast('Ollama error: ' + data.ollama_error, true);
    openSession(CURRENT_SESSION);
    loadSessions();
  } else {
    toast(data.error || 'Auto-rate failed', true);
  }
}

async function submitHuman() {
  if (!CURRENT_SESSION) return;
  const missing = DIMS.filter(d => {
    const r = CURRENT_RATINGS[d];
    return !r || r.value === undefined;
  });
  if (missing.length) {
    toast('Set a value or "Can\\'t tell" for: ' + missing.join(', '), true);
    return;
  }
  const ratings = {};
  DIMS.forEach(d => {
    const r = CURRENT_RATINGS[d];
    ratings[d] = {
      value: r.value === 'abstain' ? null : r.value,
      confidence: r.value === 'abstain' ? '' : (r.confidence || 'high')
    };
  });
  const body = {ratings: ratings, reasoning: document.getElementById('reasoning').value};
  const r = await fetch('/api/session/' + encodeURIComponent(CURRENT_SESSION) + '/rate_human', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await r.json();
  if (data.ok) {
    toast('Saved your rating ✓');
    openSession(CURRENT_SESSION);
    loadSessions();
  } else {
    toast(data.error || 'Save failed', true);
  }
}

loadSessions();
</script>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser(description="CAFT validation rater UI")
    p.add_argument("--corpus", required=True,
                   help="Directory containing session JSONL files")
    p.add_argument("--ledger", required=True,
                   help="Path to JSONL ledger (created if missing)")
    p.add_argument("--human-id", default="default",
                   help="Identifier for the human rater (default: 'default')")
    p.add_argument("--ollama-model", default=DEFAULT_MODEL,
                   help=f"Ollama model name (default: {DEFAULT_MODEL})")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    global _corpus_root, _ledger, _human_id, _ollama_model
    _corpus_root = Path(args.corpus).resolve()
    if not _corpus_root.exists():
        raise SystemExit(f"corpus not found: {_corpus_root}")
    _ledger = Ledger(args.ledger)
    _human_id = args.human_id
    _ollama_model = args.ollama_model

    print(f"Corpus: {_corpus_root}")
    print(f"Ledger: {_ledger.path}")
    print(f"Sessions found: {len(_list_sessions())}")
    print(f"Ollama: {_ollama_model} at {('UP' if is_ollama_available() else 'DOWN')}")
    print(f"Open: http://{args.host}:{args.port}/")

    app = _make_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
