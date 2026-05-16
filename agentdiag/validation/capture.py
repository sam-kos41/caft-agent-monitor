"""Trace capture / anti-rotation archiver.

THE most load-bearing piece of the pivot. Claude Code keeps only a
rolling ~3-week window of session JSONL; the original CAFT corpus was
~618 sessions and 617 of the raw traces are already gone. Any
population-level measurement program needs raw event streams, which
this preserves before they rotate away.

Design constraints:
  * stdlib only — must run from cron / a Stop hook with zero deps.
  * NEVER deletes anything. Source-of-truth is append-only safe.
  * Idempotent. Safe to run every hour forever.
  * Growth-aware: sessions grow in place; archive the LARGEST/newest
    state seen (a later run with a bigger file overwrites the archive
    copy; a smaller/older one never does).
  * Subagent traces (.../subagents/*.jsonl) are captured too.
  * Writes a manifest so corpus growth is itself measurable.

Usage:
    python -m agentdiag.validation.capture                 # one-shot
    python -m agentdiag.validation.capture --dest /vol/x   # custom dest

Keep it running (pick one):
    # crontab -e   — every 2 hours
    0 */2 * * * /usr/bin/env python3 -m agentdiag.validation.capture --quiet

    # or a Claude Code Stop hook in settings.json:
    {"hooks": {"Stop": [{"hooks": [{"type": "command",
      "command": "python3 -m agentdiag.validation.capture --quiet"}]}]}}
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SRC = Path.home() / ".claude" / "projects"
DEFAULT_DEST = Path.home() / "caft_trace_archive"


def _archive_name(src: Path, src_root: Path) -> str:
    """Stable, collision-free flat name: <project>__<...path...>.jsonl."""
    rel = src.relative_to(src_root)
    # rel is like <project_slug>/<session>.jsonl
    #          or <project_slug>/<session>/subagents/<agent>.jsonl
    return "__".join(rel.parts)


def archive(src_root: Path = DEFAULT_SRC,
            dest: Path = DEFAULT_DEST,
            quiet: bool = False) -> dict:
    src_root = Path(src_root).expanduser()
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.jsonl"

    def _n_archived() -> int:
        # manifest.jsonl lives in dest and matches *.jsonl — never count it
        return sum(1 for p in dest.glob("*.jsonl")
                   if p.name != "manifest.jsonl")

    scanned = copied_new = copied_grew = unchanged = skipped_smaller = 0
    total_bytes = 0
    rows = []

    if not src_root.exists():
        if not quiet:
            print(f"[capture] source not found: {src_root}")
        return {"error": "src_not_found", "src": str(src_root)}

    for src in sorted(src_root.rglob("*.jsonl")):
        if not src.is_file():
            continue
        scanned += 1
        try:
            s_size = src.stat().st_size
        except OSError:
            continue
        name = _archive_name(src, src_root)
        target = dest / name
        action = "unchanged"
        if not target.exists():
            shutil.copy2(src, target)
            copied_new += 1
            action = "new"
        else:
            t_size = target.stat().st_size
            if s_size > t_size:
                shutil.copy2(src, target)         # session grew
                copied_grew += 1
                action = "grew"
            elif s_size < t_size:
                skipped_smaller += 1              # never shrink the archive
                action = "skipped_smaller"
            else:
                unchanged += 1
        total_bytes += target.stat().st_size if target.exists() else 0
        rows.append({
            "archived_name": name,
            "src": str(src),
            "bytes": s_size,
            "action": action,
        })

    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "scanned": scanned,
            "new": copied_new,
            "grew": copied_grew,
            "unchanged": unchanged,
            "skipped_smaller": skipped_smaller,
            "archive_files": _n_archived(),
            "archive_bytes": total_bytes,
        }) + "\n")

    summary = {
        "src": str(src_root),
        "dest": str(dest),
        "scanned": scanned,
        "new": copied_new,
        "grew": copied_grew,
        "unchanged": unchanged,
        "skipped_smaller": skipped_smaller,
        "archive_files": _n_archived(),
        "archive_mb": round(total_bytes / 1048576, 1),
    }
    if not quiet:
        print(f"[capture] {src_root} -> {dest}")
        print(f"[capture] scanned={scanned} new={copied_new} "
              f"grew={copied_grew} unchanged={unchanged} "
              f"skipped_smaller={skipped_smaller}")
        print(f"[capture] archive now: {summary['archive_files']} files, "
              f"{summary['archive_mb']} MB (cumulative, rotation-proof)")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Claude Code trace archiver")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    archive(Path(args.src), Path(args.dest), quiet=args.quiet)


if __name__ == "__main__":
    main()
