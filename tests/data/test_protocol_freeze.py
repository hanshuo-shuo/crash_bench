from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/freeze_protocol.py"
SPEC = importlib.util.spec_from_file_location("freeze_protocol", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_repository_protocol_freeze_is_deterministic(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / path
        for path in (
            "configs/expansion/benchmark_v1.yaml",
            "configs/expansion/source_sampling_v1.yaml",
            "configs/expansion/splits_v1.yaml",
            "configs/expansion/utility_v1.yaml",
            "configs/expansion/deployable_options_v1.yaml",
            "configs/expansion/diagnostic_oracle_options_v1.yaml",
            "configs/expansion/mechanism_screens_v1.yaml",
        )
    ]
    monkeypatch.setattr(MODULE, "resolve_git_head", lambda _: "commit")
    first = MODULE.build_protocol(paths, repo_root=root)
    second = MODULE.build_protocol(reversed(paths), repo_root=root)
    assert first == second
    assert len(first["protocol_sha256"]) == 64
    assert first["test_authorized"] is False
