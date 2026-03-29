# Experiment A-BadPrompt: Qualitative Observations

**Date:** 2026-03-29
**Anomaly type:** thrash (Agent B builds frontend without reading backend code)
**Session ID (Agent B):** c783b5fa

## Agent B Timeline (thrash-injected frontend agent)

### Phase 1: Confident Blind Building (~steps 1-60)
- Agent B wrote a complete React frontend from scratch
- Invented API endpoints, WebSocket message formats, component structure
- Used guessed message types: `player_joined`, `game_started`, `new_question`, `submit_answer`
- Built WebSocket hook, Lobby, GameRoom, Scoreboard components
- Build succeeded (no syntax errors)
- **CAFT profile: Normal.** Steady writing, moderate entropy, decent MI.

### Phase 2: Discovery / Distributional Shift (~steps 60-100)
- Agent B tried to connect frontend to running backend
- `curl http://localhost:8000/` confirmed backend was running
- Probed `/docs` and `/openapi.json` to discover real API schema
- Discovered REST endpoints matched its guesses (got lucky on structure)
- Tested WebSocket: found path `/ws/{room_code}/{player_name}` worked
- **Key discoveries (mismatches):**
  - `player_connected` not `player_joined`
  - `game_start` not `game_started`
  - `question` not `new_question` (different field structure)
  - `{type: "answer", choice: index}` not `{type: "submit_answer", answer: index}`
  - `round_result` with results array, not individual `answer_result`
  - `player_answered` broadcast (not anticipated)
- **CAFT profile: KL divergence spike.** Tool distribution shifted from Write/Edit to Bash(curl)/Bash(python3). Distributional shift detected.

### Phase 3: Thrash / Mechanical Repetition (~steps 100-150)
- Agent B entered read-probe-edit-test-fail loop
- Ran multiple Python scripts to test WebSocket message flow
- Repeated timeout/retry pattern trying to capture `game_over` message format
- `sleep 10 && cat`, `sleep 20 && cat`, `sleep 60 && cat` — polling for async results
- Could not capture game completion message, decided to assume format and move on
- **CAFT detected: 32 mechanical_repetition anomalies, 1 distributional_anomaly**
- **Session metrics at this point:** MI=0.27b, coherence=13.7%, KL=0.690

### Phase 4: Recovery (~steps 150-163)
- Agent B rewrote GameRoom.js with 89 lines added, 55 removed
- Mapped all discovered message types to frontend handlers
- Build succeeded
- Started dev server, tested full flow (REST room creation + WebSocket connection)
- **Result: Working frontend that connects to backend**
- Confirmed: room creation, joining, WebSocket connections all functional

## Agent A (backend) — Healthy
- Built FastAPI backend normally
- No anomalies detected
- Healthy profile throughout

## Agent C (QA) — Healthy with secondary signal
- 49/49 tests passing
- Found 8 real bugs in the codebase
- Noted Agent B's frontend was "CRA boilerplate with no game UI" for Playwright tests
  (because the bad-prompt frontend was still being built/fixed when Agent C tested)
- **Secondary signal:** Agent C's detection of the broken frontend is qualitative
  evidence that the thrash injection created a real, observable failure

## CAFT Detection Assessment

### What CAFT caught:
- Premature termination (Critical, step 26): Agent delivering without verification
- Context loss (Warning, step 27): Re-reading same resource with 10 ops between
- 32 mechanical_repetition signatures in the thrash session
- 1 distributional_anomaly
- KL divergence spike to 0.690 (vs 0.000 for all healthy sessions)
- MI collapse to 0.27b (vs 1.67b healthy average)
- Coherence collapse to 13.7% (vs 81.5% healthy average)

### What CAFT did NOT tell us:
- Could not distinguish "debugging phase" from "genuinely confused" (global baseline)
- No phase-conditional detection (would require harness infrastructure)
- No retrospective correlation (no evaluator QA grades to cross-reference)
- Anomalies were detected but without temporal context of WHY the shift happened

### Key finding:
Agent B **recovered** despite the bad prompt. The thrash was real (32 anomalies),
but the agent eventually figured out the real API and rewrote its frontend.
This means the anomaly pattern is: healthy -> shift -> thrash -> recovery,
not permanent degradation. CAFT correctly detected the middle phases but
has no mechanism to detect the recovery (return to healthy baseline).
