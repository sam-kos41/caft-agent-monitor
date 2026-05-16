# Demo — human-factors constructs over a real agent trajectory

A static, self-contained visual: **one real SWE-agent trajectory**,
rendered step-by-step, with each construct's per-step signal overlaid.
It makes the constructs *concrete* — you can see, on actual agent
interaction data, what "perception", "workload", "error recovery", and
"thought-action coherence" each measure, against the
information-theoretic action-sequence "ribbon" that turned out to be
load-bearing.

## Run it

Open `index.html` in a browser. No build, no server, no dependencies —
ready for a personal site or GitHub Pages. Hover any step for the
verbatim reasoning + what each construct saw there.

## What you're looking at

Trajectory = frozen-sample **row 237** (`beetbox__beets-3868`,
swe-agent-llama-70b, resolved). Selected by position because
`instance_id` is non-unique across model scales. 18 action steps.
Lanes:

- **Action ribbon** — the action type per step; this *sequence shape*
  is exactly what information-theoretic structure scores.
- **Workload** — reasoning length per step (Sweller / Wickens analog).
- **Perception** — for edits: had the agent read that file first?
  (Endsley Level-1 analog).
- **Error / recovery** — did the step error, and did the agent change
  strategy after? (Reason / Hollnagel analog).
- **Thought–Action** — did the stated intent match the action taken?
  (agent-native).

The **action ribbon is rendered as the dominant lane** — it is the
load-bearing measurement, and each cell is tagged `● new` / `↻ rep`
to show the sequence's own redundancy (the visible signature of what
compression / MI score; illustrative, not a per-step metric).

Below the trajectory: a **three-step zoom** (executes-well /
heaviest-or-errored / struggling, chosen from this trajectory) so the
lenses are visibly varying, and the **population finding table**
(N = 2,000) — each construct alone vs. beyond IT vs. correlation with
IT. The finding lives on the page itself, not only behind a link.

Every per-step value is computed with the **same frozen logic the
committed extractors use** (imported, not re-derived); the table
numbers are the validated population results (see snapshot); the data
is baked into the page — no analysis runs in the browser. Regenerate
with `python demo/build_trajectory_demo.py`.

## Read this — it is part of the result

This is **one illustrative trajectory**. The validated finding is
**population-level** (2,000 trajectories): all four constructs predict
task outcome on their own, yet none add predictive value beyond IT
action-sequence structure — and that survives a graded outcome. A
single pretty trajectory is **not** evidence for per-instance
detection; that regime was tested and failed (the CAFT autopsy).
Scope: one corpus (nebius SWE-agent), one agent family (Llama,
reactive ReAct), one outcome family. "Agents are universally flat" is
not claimed.

Full narrative & evidence ledger:
[`../docs/PROJECT_SNAPSHOT.html`](../docs/PROJECT_SNAPSHOT.html).
Governing doc and per-leg pre-registrations under `../docs/`.
