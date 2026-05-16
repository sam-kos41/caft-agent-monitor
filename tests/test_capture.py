"""Tests for the anti-rotation trace archiver.

The whole point is durability + idempotency + growth-awareness, so
those are pinned exactly on a synthetic ~/.claude-like tree.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentdiag.validation.capture import archive, _archive_name


def _mk(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_first_capture_copies_all(tmp_path):
    src = tmp_path / "projects"
    dest = tmp_path / "arch"
    _mk(src / "-proj-a" / "sess1.jsonl", '{"a":1}\n')
    _mk(src / "-proj-a" / "sess1" / "subagents" / "agent-x.jsonl", '{"b":2}\n')
    _mk(src / "-proj-b" / "sess2.jsonl", '{"c":3}\n')
    s = archive(src, dest, quiet=True)
    assert s["scanned"] == 3
    assert s["new"] == 3
    assert s["archive_files"] == 3
    # subagent flattened into a collision-free name
    assert (dest / "-proj-a__sess1__subagents__agent-x.jsonl").exists()


def test_idempotent_second_run(tmp_path):
    src = tmp_path / "projects"
    dest = tmp_path / "arch"
    _mk(src / "-p" / "s.jsonl", '{"x":1}\n')
    archive(src, dest, quiet=True)
    s2 = archive(src, dest, quiet=True)
    assert s2["new"] == 0
    assert s2["unchanged"] == 1
    assert s2["archive_files"] == 1


def test_growth_is_captured(tmp_path):
    src = tmp_path / "projects"
    dest = tmp_path / "arch"
    f = src / "-p" / "s.jsonl"
    _mk(f, '{"x":1}\n')
    archive(src, dest, quiet=True)
    f.write_text('{"x":1}\n{"x":2}\n{"x":3}\n')   # session grew
    s2 = archive(src, dest, quiet=True)
    assert s2["grew"] == 1
    assert (dest / "-p__s.jsonl").read_text().count("\n") == 3


def test_archive_never_shrinks(tmp_path):
    """A rotated/truncated source must NOT overwrite a fuller archive copy."""
    src = tmp_path / "projects"
    dest = tmp_path / "arch"
    f = src / "-p" / "s.jsonl"
    _mk(f, '{"x":1}\n{"x":2}\n{"x":3}\n')
    archive(src, dest, quiet=True)
    f.write_text('{"x":1}\n')                      # source shrank
    s2 = archive(src, dest, quiet=True)
    assert s2["skipped_smaller"] == 1
    assert (dest / "-p__s.jsonl").read_text().count("\n") == 3  # preserved


def test_manifest_grows_each_run(tmp_path):
    src = tmp_path / "projects"
    dest = tmp_path / "arch"
    _mk(src / "-p" / "s.jsonl", '{"x":1}\n')
    archive(src, dest, quiet=True)
    archive(src, dest, quiet=True)
    lines = (dest / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["scanned"] == 1


def test_missing_source_is_safe(tmp_path):
    s = archive(tmp_path / "nope", tmp_path / "arch", quiet=True)
    assert s.get("error") == "src_not_found"


def test_archive_name_flattens_path(tmp_path):
    root = tmp_path / "projects"
    p = root / "-proj" / "sess" / "subagents" / "a.jsonl"
    assert _archive_name(p, root) == "-proj__sess__subagents__a.jsonl"
