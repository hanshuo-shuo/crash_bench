#!/usr/bin/env python3
"""Freeze the composite D4 protocol hash from all prospective config inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.expansion.hash_tree_manifest import resolve_git_head


def build_protocol(paths: Iterable[Path], *, repo_root: Path) -> dict[str, Any]:
    rows = []
    payloads = {}
    for path in sorted(paths, key=lambda value: value.as_posix()):
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        raw = path.read_bytes()
        rows.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
        payloads[relative] = json.loads(raw)
    required_names = {
        "configs/expansion/benchmark_v1.yaml",
        "configs/expansion/source_sampling_v1.yaml",
        "configs/expansion/splits_v1.yaml",
        "configs/expansion/utility_v1.yaml",
        "configs/expansion/deployable_options_v1.yaml",
        "configs/expansion/diagnostic_oracle_options_v1.yaml",
        "configs/expansion/mechanism_screens_v1.yaml",
    }
    if set(payloads) != required_names:
        raise ValueError(f"protocol config set mismatch: {sorted(set(payloads) ^ required_names)}")
    benchmark = payloads["configs/expansion/benchmark_v1.yaml"]
    utility = payloads["configs/expansion/utility_v1.yaml"]
    if benchmark.get("test_policy", {}).get("test_authorized") is not False:
        raise ValueError("D4 protocol cannot freeze with test authorized")
    if benchmark.get("test_policy", {}).get("test_outcomes_may_be_read") is not False:
        raise ValueError("D4 protocol cannot freeze after test outcome access")
    normalization = utility.get("normalization", {})
    for key in (
        "option_duration_steps", "path_length_m", "force_exposure_ns", "latency_ms"
    ):
        if float(normalization.get(key, 0)) <= 0:
            raise ValueError(f"utility physical budget is not frozen: {key}")
    if "test" in normalization.get("source", "").lower() and "no formal/test" not in normalization.get("source", "").lower():
        raise ValueError("utility normalization source appears test-derived")
    canonical_inputs = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    protocol_sha = hashlib.sha256(canonical_inputs).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d4_protocol_freeze",
        "protocol_sha256": protocol_sha,
        "git_commit": resolve_git_head(repo_root),
        "inputs": rows,
        "primary_policy": benchmark["primary_policy"],
        "active_scope": benchmark["active_scoped_pilot"],
        "test_authorized": False,
        "test_outcomes_read": 0,
    }
    payload["freeze_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite protocol freeze: {args.output}")
    payload = build_protocol(args.config, repo_root=args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "protocol_sha256": payload["protocol_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
