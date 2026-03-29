"""
Real Claude Code agents for the CAFT harness.

These wrap `claude -p` (non-interactive print mode) to implement the
planner / generator / evaluator protocol that HarnessOrchestrator expects.

The generator runs with full tool access so it actually writes files.
The planner and evaluator only need text output.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional


def _run_claude(prompt: str, cwd: str, allow_tools: bool = False,
                timeout: int = 300, label: str = "claude",
                verbose: bool = False) -> str:
    """Run claude -p and return the text output.

    Args:
        prompt: The prompt to send.
        cwd: Working directory for the claude process.
        allow_tools: If True, allow file/shell tools so the agent can write code.
        timeout: Max seconds to wait.
        label: Label for verbose output (e.g. "generator", "evaluator").
        verbose: If True, stream output lines to terminal in real time.
    """
    cmd = ["claude", "-p", prompt]

    if allow_tools:
        cmd.extend(["--allowedTools", "Read Edit Write Bash Glob Grep"])

    try:
        if verbose:
            # Stream output in real time so user can see what's happening
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
            output_lines = []
            import select
            import threading

            def _read_stream(stream, lines, prefix):
                for line in stream:
                    line = line.rstrip()
                    if line:
                        lines.append(line)
                        print(f"    [{label}] {line}", flush=True)

            stdout_thread = threading.Thread(
                target=_read_stream, args=(proc.stdout, output_lines, label))
            stdout_thread.daemon = True
            stdout_thread.start()

            # Wait for completion with timeout
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"    [!] {label} timed out after {timeout}s")
                proc.kill()
                proc.wait()
                return "\n".join(output_lines) or "(timed out)"

            stdout_thread.join(timeout=5)
            return "\n".join(output_lines)
        else:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout.strip()
            if result.returncode != 0 and not output:
                output = result.stderr.strip() or "(claude returned no output)"
            return output
    except subprocess.TimeoutExpired:
        print(f"    [!] {label} timed out after {timeout}s — returning partial result")
        subprocess.run(f"pkill -f 'claude.*-p' 2>/dev/null", shell=True, check=False)
        return "(timed out)"


def _extract_json(text: str) -> Optional[dict | list]:
    """Try to extract JSON from Claude's output (may be wrapped in markdown)."""
    # Try the whole thing first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Look for ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Look for [ ... ] or { ... } blocks
    for pattern in [r"\[.*\]", r"\{.*\}"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

    return None


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------

class ClaudePlanner:
    """Breaks a goal into sprint specifications using Claude."""

    def __init__(self, cwd: str, max_sprints: int = 3, timeout: int = 120):
        self.cwd = cwd
        self.max_sprints = max_sprints
        self.timeout = timeout

    def __call__(self, goal: str, context: dict) -> list[dict]:
        prompt = (
            f"You are a software project planner. Break this goal into "
            f"{self.max_sprints} sprints (or fewer if the task is small).\n\n"
            f"Goal: {goal}\n\n"
            f"Output ONLY a JSON array. Each element must have exactly these keys:\n"
            f'  "goal": string (what this sprint accomplishes)\n'
            f'  "deliverables": [string, ...] (file names or components to build)\n'
            f'  "success_criteria": [string, ...] (testable conditions for done)\n\n'
            f"Example:\n"
            f'[{{"goal": "Build REST API", "deliverables": ["server.py", "models.py"], '
            f'"success_criteria": ["all endpoints return 200", "SQLite schema created"]}}]\n\n'
            f"Output JSON only, no explanation."
        )

        print(f"    [planner] Asking Claude to plan: {goal[:60]}...")
        output = _run_claude(prompt, self.cwd, allow_tools=False,
                             timeout=self.timeout, label="planner", verbose=True)
        print(f"    [planner] Done ({len(output)} chars)")

        parsed = _extract_json(output)
        if isinstance(parsed, list) and len(parsed) > 0:
            # Validate each sprint has required keys
            sprints = []
            for i, spec in enumerate(parsed[:self.max_sprints]):
                sprints.append({
                    "goal": spec.get("goal", f"Sprint {i+1}"),
                    "deliverables": spec.get("deliverables", [f"deliverable_{i}"]),
                    "success_criteria": spec.get("success_criteria", [f"criterion_{i}"]),
                })
            return sprints

        # Fallback: single sprint with the whole goal
        print(f"    [planner] Could not parse JSON, using single-sprint fallback")
        return [{
            "goal": goal,
            "deliverables": ["backend", "frontend", "tests"],
            "success_criteria": [
                "server starts without errors",
                "frontend connects to backend",
                "tests pass",
            ],
        }]


# ---------------------------------------------------------------------------
# Generator Agent
# ---------------------------------------------------------------------------

class ClaudeGenerator:
    """Generates code using Claude with full tool access.

    If anomaly_instructions is set, it's prepended to the generator prompt
    to inject pathological behavior (e.g. "don't read backend code").
    """

    def __init__(self, cwd: str, timeout: int = 600,
                 anomaly_instructions: Optional[str] = None):
        self.cwd = cwd
        self.timeout = timeout
        self.anomaly_instructions = anomaly_instructions

    def __call__(self, contract, context: dict, feedback=None) -> dict:
        # Build the prompt
        parts = []

        if self.anomaly_instructions:
            parts.append(f"IMPORTANT INSTRUCTIONS: {self.anomaly_instructions}")
            parts.append("")

        parts.extend([
            f"You are a code generator working on sprint {contract.sprint_number}.",
            f"\nGoal: {contract.goal}",
            f"\nDeliverables to create: {', '.join(contract.deliverables)}",
            f"\nSuccess criteria: {', '.join(contract.success_criteria)}",
        ])

        if feedback:
            parts.append(f"\n\nPREVIOUS ATTEMPT FAILED (iteration {feedback.iteration}).")
            parts.append(f"Evaluator critique: {feedback.critique}")
            if feedback.suggestions:
                parts.append(f"Suggestions: {', '.join(feedback.suggestions)}")
            parts.append("\nFix the issues and try again.")
        else:
            parts.append(
                "\n\nWrite the code now. Create all necessary files. "
                "Use best practices. Make sure the code actually works."
            )

        prompt = "\n".join(parts)
        iteration = feedback.iteration + 1 if feedback else 1

        print(f"    [generator] Sprint {contract.sprint_number}, iteration {iteration}...")
        print(f"    [generator] Goal: {contract.goal[:80]}")
        print(f"    [generator] Deliverables: {', '.join(contract.deliverables)}")
        if self.anomaly_instructions:
            print(f"    [generator] *** ANOMALY ACTIVE: building blind ***")
        print()
        start = time.time()
        output = _run_claude(prompt, self.cwd, allow_tools=True,
                             timeout=self.timeout, label="generator", verbose=True)
        elapsed = time.time() - start
        print()
        print(f"    [generator] Done in {elapsed:.0f}s ({len(output)} chars)")

        # Discover what files were created/modified (exclude node_modules, etc.)
        files_changed = []
        try:
            # Stage everything first so diff shows all changes
            subprocess.run(
                ["git", "add", "-A"], cwd=self.cwd,
                capture_output=True, timeout=10,
            )
            # Diff against last commit
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.cwd, capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                files_changed = [
                    f for f in result.stdout.strip().split("\n")
                    if not f.startswith("node_modules/")
                    and not f.startswith(".next/")
                    and not f.startswith("__pycache__/")
                    and not f.endswith(".pyc")
                ]
            # Commit so the next sprint has a clean baseline
            if files_changed:
                subprocess.run(
                    ["git", "commit", "-m",
                     f"Sprint {contract.sprint_number} iteration {iteration}"],
                    cwd=self.cwd, capture_output=True, timeout=15,
                )
        except Exception:
            pass

        return {
            "files": files_changed,
            "output": output[:2000],  # Truncate for storage
            "iteration": iteration,
            "duration_sec": elapsed,
        }


# ---------------------------------------------------------------------------
# Evaluator Agent
# ---------------------------------------------------------------------------

class ClaudeEvaluator:
    """Reviews generated code and scores it against the contract."""

    def __init__(self, cwd: str, timeout: int = 180):
        self.cwd = cwd
        self.timeout = timeout

    def __call__(self, contract, artifacts: dict, context: dict):
        # Import here to avoid circular imports at module level
        from agentdiag.harness import EvaluationGrade

        files = artifacts.get("files", [])
        criteria = contract.success_criteria

        # Build a prompt that asks for structured scoring
        file_list = "\n".join(f"  - {f}" for f in files) if files else "  (no files detected)"

        prompt = (
            f"You are a code reviewer evaluating sprint {contract.sprint_number}.\n\n"
            f"Goal: {contract.goal}\n"
            f"Deliverables expected: {', '.join(contract.deliverables)}\n"
            f"Files produced:\n{file_list}\n\n"
            f"Read the code files listed above and evaluate against these criteria.\n"
            f"For each criterion, score 0.0 to 1.0:\n"
        )
        for c in criteria:
            prompt += f'  - "{c}"\n'

        prompt += (
            f"\nOutput ONLY a JSON object with these exact keys:\n"
            f'  "criteria_scores": {{"criterion_name": score, ...}}\n'
            f'  "overall_score": float (average of criteria scores)\n'
            f'  "passed": boolean (true if overall >= 0.7)\n'
            f'  "critique": string (what needs fixing, empty if passed)\n'
            f'  "suggestions": [string, ...] (specific improvements)\n\n'
            f"Be honest and strict. Output JSON only."
        )

        print(f"    [evaluator] Reviewing sprint {contract.sprint_number} "
              f"({len(files)} files, {len(criteria)} criteria)...")
        print()
        output = _run_claude(prompt, self.cwd, allow_tools=True,
                             timeout=self.timeout, label="evaluator", verbose=True)
        print()
        print(f"    [evaluator] Done ({len(output)} chars)")

        parsed = _extract_json(output)
        if isinstance(parsed, dict):
            criteria_scores = parsed.get("criteria_scores", {})
            # Normalize: ensure all contract criteria have a score
            for c in criteria:
                if c not in criteria_scores:
                    criteria_scores[c] = 0.5  # Default for missing
            overall = parsed.get("overall_score",
                                 sum(criteria_scores.values()) / max(len(criteria_scores), 1))
            return EvaluationGrade(
                sprint_number=contract.sprint_number,
                overall_score=round(float(overall), 3),
                criteria_scores={k: round(float(v), 3) for k, v in criteria_scores.items()},
                passed=float(overall) >= 0.7,
                critique=parsed.get("critique", ""),
                suggestions=parsed.get("suggestions", []),
            )

        # Fallback: couldn't parse, assume moderate score
        print(f"    [evaluator] Could not parse JSON, using fallback score")
        fallback_scores = {c: 0.6 for c in criteria}
        return EvaluationGrade(
            sprint_number=contract.sprint_number,
            overall_score=0.6,
            criteria_scores=fallback_scores,
            passed=False,
            critique="Evaluator could not parse its own output. Manual review needed.",
            suggestions=["Re-run evaluation"],
        )


# ---------------------------------------------------------------------------
# Contract Negotiator
# ---------------------------------------------------------------------------

class ClaudeNegotiator:
    """Reviews and optionally amends sprint contracts."""

    def __init__(self, cwd: str, timeout: int = 60):
        self.cwd = cwd
        self.timeout = timeout

    def __call__(self, contract, context: dict):
        criteria_list = "\n".join(f"  - {c}" for c in contract.success_criteria)
        prompt = (
            f"You are a QA lead reviewing a sprint contract.\n\n"
            f"Sprint {contract.sprint_number}: {contract.goal}\n"
            f"Deliverables: {', '.join(contract.deliverables)}\n"
            f"Success criteria:\n{criteria_list}\n\n"
            f"Are the success criteria sufficient? If not, add 1-2 more.\n"
            f"Output ONLY a JSON object:\n"
            f'  "amendments": string (what you changed, empty if nothing)\n'
            f'  "additional_criteria": [string, ...] (new criteria to add, or empty [])\n\n'
            f"Output JSON only."
        )

        print(f"    [negotiator] Reviewing contract for sprint {contract.sprint_number}...")
        print(f"    [negotiator] Criteria: {', '.join(contract.success_criteria[:3])}...")
        output = _run_claude(prompt, self.cwd, allow_tools=False, timeout=self.timeout)

        parsed = _extract_json(output)
        if isinstance(parsed, dict):
            additional = parsed.get("additional_criteria", [])
            amendments = parsed.get("amendments", "")
            if additional:
                contract.success_criteria.extend(additional)
                contract.evaluator_amendments = amendments or "Added criteria from QA review"
                contract.status = "amended"
                print(f"    [negotiator] Added {len(additional)} criteria")
            else:
                print(f"    [negotiator] No changes needed")
        return contract
