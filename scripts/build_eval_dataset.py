#!/usr/bin/env python3
"""Build the 4-bucket evaluation dataset from local Claude Code sessions.

Discovers all Claude Code session logs, assigns them to evaluation buckets
using a deterministic hash-based split, creates symlinks in the data/
directories, and generates a CSV manifest.

Bucket allocation (by session_id hash):
  A  claude_debug  10%   Parser debugging — you may freely inspect these
  B  claude_val    20%   Detector calibration / threshold tuning
  C  claude_test   70%   Blind final evaluation — DO NOT PEEK

External buckets (mast/, agentrx/) are created empty with download
instructions in their README files.

Usage:
    python -m agentdiag.scripts.build_eval_dataset
    python scripts/build_eval_dataset.py
    python scripts/build_eval_dataset.py --traces ~/.claude/projects --min-lines 10
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hash-based split boundaries (deterministic, reproducible)
# bucket = int(sha256(session_id).hexdigest()[:8], 16) % 100
#   0-9   -> debug  (10%)
#   10-29 -> val    (20%)
#   30-99 -> test   (70%)
SPLIT_BOUNDS = {"claude_debug": (0, 10), "claude_val": (10, 30), "claude_test": (30, 100)}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bucket_for_session(session_id: str) -> str:
    """Deterministic bucket assignment based on SHA-256 of session_id."""
    h = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) % 100
    for bucket, (lo, hi) in SPLIT_BOUNDS.items():
        if lo <= h < hi:
            return bucket
    return "claude_test"  # fallback


def discover_sessions(root: Path, min_lines: int = 10) -> list[dict]:
    """Walk ~/.claude/projects and find top-level session JSONL files."""
    root = Path(root).expanduser()
    sessions = []

    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name

        for jsonl_file in sorted(project_dir.glob("*.jsonl")):
            # Skip subagent files
            if "subagents" in str(jsonl_file):
                continue

            line_count = sum(1 for _ in open(jsonl_file, "r", errors="replace"))
            if line_count < min_lines:
                continue

            session_id = jsonl_file.stem

            # Extract first/last timestamps
            first_ts, last_ts = None, None
            try:
                with open(jsonl_file, "r", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            ts = obj.get("timestamp")
                            if ts:
                                if first_ts is None:
                                    first_ts = ts
                                last_ts = ts
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass

            sessions.append({
                "session_id": session_id,
                "source_path": str(jsonl_file),
                "project": project_name,
                "line_count": line_count,
                "first_timestamp": first_ts or "",
                "last_timestamp": last_ts or "",
            })

    return sessions


def compute_session_stats(path: str) -> dict:
    """Quick stats from a session JSONL without full parsing.

    Claude Code JSONL format nests content under obj["message"]["content"],
    not at the top level.
    """
    stats = {
        "event_types": {},
        "tool_names": {},
        "error_count": 0,
        "has_tool_use": False,
    }
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Track event type (top-level "type" field)
                etype = obj.get("type", "")
                stats["event_types"][etype] = stats["event_types"].get(etype, 0) + 1

                # Claude Code nests content under message.content
                message = obj.get("message", {})
                content = message.get("content", obj.get("content", []))
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_use":
                                name = block.get("name", "unknown")
                                stats["tool_names"][name] = stats["tool_names"].get(name, 0) + 1
                                stats["has_tool_use"] = True
                            if block.get("type") == "tool_result" and block.get("is_error"):
                                stats["error_count"] += 1
    except Exception:
        pass

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(
    traces_root: str = "~/.claude/projects",
    min_lines: int = 10,
    copy_mode: str = "symlink",  # "symlink" or "copy"
) -> None:
    """Build the 4-bucket evaluation dataset.

    Args:
        traces_root: Where Claude Code stores session logs.
        min_lines: Minimum lines to include a session.
        copy_mode: "symlink" creates symlinks, "copy" copies files.
    """
    traces_root = Path(traces_root).expanduser()
    print(f"Discovering sessions in {traces_root}...")
    sessions = discover_sessions(traces_root, min_lines=min_lines)
    print(f"  Found {len(sessions)} sessions (>= {min_lines} lines)")

    if not sessions:
        print("No sessions found. Check --traces path.")
        sys.exit(1)

    # Assign buckets
    for s in sessions:
        s["bucket"] = bucket_for_session(s["session_id"])

    # Count per bucket
    counts = {}
    for s in sessions:
        counts[s["bucket"]] = counts.get(s["bucket"], 0) + 1
    print(f"\n  Bucket allocation:")
    for bucket in ["claude_debug", "claude_val", "claude_test"]:
        print(f"    {bucket:<16} {counts.get(bucket, 0):>4} sessions")

    # Create symlinks/copies
    print(f"\n  Populating data/ directories ({copy_mode} mode)...")
    for s in sessions:
        bucket_dir = DATA_DIR / s["bucket"]
        bucket_dir.mkdir(parents=True, exist_ok=True)
        dest = bucket_dir / f"{s['session_id']}.jsonl"

        # Remove existing link/file
        if dest.exists() or dest.is_symlink():
            dest.unlink()

        src = Path(s["source_path"])
        if copy_mode == "symlink":
            dest.symlink_to(src)
        else:
            shutil.copy2(src, dest)

        s["data_path"] = str(dest)

    # Compute quick stats for manifest
    print("  Computing session stats for manifest...")
    manifest_rows = []
    for s in sessions:
        stats = compute_session_stats(s["source_path"])
        tools_str = ",".join(sorted(stats["tool_names"].keys()))
        manifest_rows.append({
            "session_id": s["session_id"],
            "bucket": s["bucket"],
            "project": s["project"],
            "line_count": s["line_count"],
            "first_timestamp": s["first_timestamp"],
            "last_timestamp": s["last_timestamp"],
            "has_tool_use": stats["has_tool_use"],
            "error_count": stats["error_count"],
            "tools_used": tools_str,
            "label_status": "unlabeled",
            "caft_labels": "",
            "notes": "",
        })

    # Sort by bucket then session_id for reproducibility
    manifest_rows.sort(key=lambda r: (r["bucket"], r["session_id"]))

    # Write manifest CSV
    fieldnames = [
        "session_id", "bucket", "project", "line_count",
        "first_timestamp", "last_timestamp",
        "has_tool_use", "error_count", "tools_used",
        "label_status", "caft_labels", "notes",
    ]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n  Manifest written to {MANIFEST_PATH}")
    print(f"  {len(manifest_rows)} rows")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION DATASET BUILT")
    print(f"{'=' * 60}")
    print(f"  data/claude_debug/   {counts.get('claude_debug', 0):>3} traces  (Bucket A: parser debugging)")
    print(f"  data/claude_val/     {counts.get('claude_val', 0):>3} traces  (Bucket B: detector calibration)")
    print(f"  data/claude_test/    {counts.get('claude_test', 0):>3} traces  (Bucket C: blind evaluation)")
    print(f"  data/mast/             0 traces  (Bucket D: download from HuggingFace)")
    print(f"  data/agentrx/          0 traces  (Bucket D: download from GitHub)")
    print(f"  data/manifest.csv  {len(manifest_rows):>5} rows")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build 4-bucket evaluation dataset")
    parser.add_argument(
        "--traces", default="~/.claude/projects",
        help="Root directory with Claude Code session logs",
    )
    parser.add_argument(
        "--min-lines", type=int, default=10,
        help="Minimum lines to include a session (default: 10)",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlinking (uses more disk)",
    )
    args = parser.parse_args()

    build_dataset(
        traces_root=args.traces,
        min_lines=args.min_lines,
        copy_mode="copy" if args.copy else "symlink",
    )
