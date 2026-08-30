from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/hash_tree_manifest.py"
SPEC = importlib.util.spec_from_file_location("hash_tree_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_tree_manifest_is_content_and_path_deterministic(tmp_path, monkeypatch):
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.bin").write_bytes(b"b")
    (tmp_path / "nested/a.bin").write_bytes(b"a")
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": "abc123\n"})(),
    )
    first = MODULE.tree_manifest(tmp_path, "fixture")
    second = MODULE.tree_manifest(tmp_path, "fixture")
    assert first["tree_manifest_sha256"] == second["tree_manifest_sha256"]
    assert first["file_count"] == 2
    assert [row["path"] for row in first["files"]] == ["b.bin", "nested/a.bin"]


def test_tree_manifest_rejects_internal_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("x")
    (tmp_path / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.tree_manifest(tmp_path, "fixture")
