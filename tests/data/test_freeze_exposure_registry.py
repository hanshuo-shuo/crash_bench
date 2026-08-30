from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/freeze_exposure_registry.py"
SPEC = importlib.util.spec_from_file_location("freeze_exposure_registry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_registry_unions_json_jsonl_and_csv(tmp_path, monkeypatch):
    hashes = [char * 64 for char in "abc"]
    (tmp_path / "results").mkdir()
    (tmp_path / "results/a.json").write_text(json.dumps({"source_state_sha256": hashes[0]}))
    (tmp_path / "results/b.jsonl").write_text(json.dumps({"source_sha256": hashes[1]}) + "\n")
    (tmp_path / "results/c.csv").write_text(f"source_state_sha256,reset_seed\n{hashes[2]},9\n")
    monkeypatch.setattr(MODULE, "git_tracked_files", lambda _: set())
    payload = MODULE.build_registry(
        tmp_path,
        {"scan_roots": ["results"], "artifact_roots": [], "pool_blacklists": []},
        include_local=True,
    )
    assert [row["source_state_sha256"] for row in payload["sources"]] == hashes
    assert payload["gate"]["status"] == "GO"
    assert payload["counts"]["source_union_count"] == 3


def test_parse_failure_is_fail_closed(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "results/bad.json").write_text('{"source_state_sha256":')
    monkeypatch.setattr(MODULE, "git_tracked_files", lambda _: set())
    payload = MODULE.build_registry(
        tmp_path,
        {"scan_roots": ["results"], "artifact_roots": [], "pool_blacklists": []},
        include_local=True,
    )
    assert payload["gate"]["status"] == "NO_GO"
    assert payload["gate"]["unresolved_unblacklisted_count"] == 1


def test_registry_excludes_its_own_generated_output(tmp_path, monkeypatch):
    output = tmp_path / "results/expansion/governance/exposure_registry.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({"source_state_sha256": "d" * 64}))
    monkeypatch.setattr(MODULE, "git_tracked_files", lambda _: set())
    payload = MODULE.build_registry(
        tmp_path,
        {"scan_roots": ["results"], "artifact_roots": [], "pool_blacklists": []},
        include_local=True,
    )
    assert payload["sources"] == []
    assert payload["inputs"] == []
