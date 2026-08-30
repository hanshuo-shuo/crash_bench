#!/usr/bin/env python3
"""Freeze the additive union of all locally discoverable exposed source lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_KEYS = {
    "source_state_sha256",
    "source_sha256",
    "init_state_sha256",
    "initial_state_sha256",
}
LINEAGE_HINTS = SOURCE_KEYS | {
    "source_trace_manifest_sha256",
    "source_manifest_sha256",
    "reset_seed",
    "generator_seed",
    "scene_fingerprint",
    "physical_source_id",
    "upstream_source_key",
}
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml"}
SELF_EXCLUDED_PATHS = {"results/expansion/governance/exposure_registry.json"}
IDENTIFIER_KEYS = {
    "upstream_source_key": "upstream_source_key",
    "physical_source_id": "physical_source_id",
    "scene_fingerprint": "scene_fingerprint",
    "reset_seed": "reset_seed",
    "generator_seed": "generator_seed",
    "source_manifest_sha256": "source_manifest_sha256",
    "source_trace_manifest_sha256": "source_manifest_sha256",
    "nominal_source_trace_manifest_sha256": "source_manifest_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.lower()
    if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
        return value
    return None


def is_source_hash_key(key: str) -> bool:
    lowered = key.lower()
    if "manifest" in lowered or "fingerprint" in lowered:
        return False
    return key in SOURCE_KEYS or (
        ("source" in lowered or "init_state" in lowered or "initial_state" in lowered)
        and "sha256" in lowered
    )


def hashes_in_value(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {item for child in value for item in hashes_in_value(child)}
    source_hash = canonical_sha(value)
    return {source_hash} if source_hash else set()


def walk_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child)


def parse_records(path: Path) -> tuple[str, list[Any]]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            return "json_compatible_yaml", [json.loads(path.read_text())]
        except json.JSONDecodeError:
            import yaml

            return "yaml", [yaml.safe_load(path.read_text())]
    if suffix == ".json":
        return "json", [json.loads(path.read_text())]
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return "jsonl", rows
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return "csv", list(csv.DictReader(handle))
    raise ValueError(f"unsupported suffix: {suffix}")


def artifact_role(relative_path: str, roots: list[dict[str, Any]]) -> str:
    matches = [
        row for row in roots
        if relative_path == row["path"] or relative_path.startswith(row["path"].rstrip("/") + "/")
    ]
    if matches:
        return max(matches, key=lambda row: len(row["path"]))["artifact_role"]
    return "EXPOSED_LEGACY_SOURCE_LINEAGE"


def git_tracked_files(repo_root: Path) -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo_root, check=True, capture_output=True
    )
    return {
        (repo_root / item.decode()).resolve()
        for item in result.stdout.split(b"\0")
        if item
    }


def candidate_files(repo_root: Path, scan_roots: list[str], include_local: bool) -> list[Path]:
    tracked = git_tracked_files(repo_root)
    paths: set[Path] = set()
    for root_name in scan_roots:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            resolved = path.resolve()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                if path.relative_to(repo_root).as_posix() in SELF_EXCLUDED_PATHS:
                    continue
                if include_local or resolved in tracked:
                    paths.add(resolved)
    return sorted(paths)


def build_registry(repo_root: Path, ledger: dict[str, Any], include_local: bool) -> dict[str, Any]:
    artifact_roots = ledger.get("artifact_roots", [])
    source_evidence: dict[str, set[str]] = defaultdict(set)
    source_roles: dict[str, set[str]] = defaultdict(set)
    source_metadata: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    identifier_evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    identifier_roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    inputs: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []

    for path in candidate_files(repo_root, ledger.get("scan_roots", []), include_local):
        relative = path.relative_to(repo_root).as_posix()
        try:
            raw_text = path.read_text(errors="replace")
        except OSError as exc:
            unresolved.append({"path": relative, "reason": f"read_error:{exc}"})
            continue
        if not any(hint in raw_text for hint in LINEAGE_HINTS):
            continue
        try:
            schema, records = parse_records(path)
        except Exception as exc:  # fail closed and preserve exact file in blocker queue
            unresolved.append({"path": relative, "reason": f"parse_error:{type(exc).__name__}"})
            continue
        found: set[str] = set()
        current_role = artifact_role(relative, artifact_roots)
        for record in records:
            flattened = list(walk_values(record))
            for key, value in flattened:
                identifier_type = IDENTIFIER_KEYS.get(key)
                if identifier_type is None or value in (None, "") or isinstance(value, (dict, list)):
                    continue
                normalized = str(value).lower() if identifier_type == "source_manifest_sha256" else str(value)
                if identifier_type == "source_manifest_sha256" and canonical_sha(normalized) is None:
                    continue
                identifier_evidence[(identifier_type, normalized)].add(relative)
                identifier_roles[(identifier_type, normalized)].add(current_role)
            row_sources = {
                source_hash
                for key, value in flattened
                if is_source_hash_key(key)
                for source_hash in hashes_in_value(value)
            }
            found.update(row_sources)
            for source_hash in row_sources:
                source_evidence[source_hash].add(relative)
                source_roles[source_hash].add(current_role)
                for key, value in flattened:
                    if key in {"physical_source_id", "scene_fingerprint", "reset_seed"} and value not in (None, ""):
                        source_metadata[source_hash][key].add(str(value))
        inputs.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "parser": schema,
                "row_count": len(records),
                "source_hash_count": len(found),
            }
        )

    missing_roots = []
    for row in artifact_roots:
        if not (repo_root / row["path"]).exists() and not row.get("allow_missing", False):
            missing_roots.append(row["path"])

    sources = []
    for source_hash in sorted(source_evidence):
        row: dict[str, Any] = {
            "source_state_sha256": source_hash,
            "artifact_roles": sorted(source_roles[source_hash]),
            "evidence": sorted(source_evidence[source_hash]),
            "test_eligible": False,
        }
        for key, values in source_metadata[source_hash].items():
            if len(values) == 1:
                value = next(iter(values))
                row[key] = int(value) if key == "reset_seed" and value.isdigit() else value
            else:
                row[f"observed_{key}s"] = sorted(values)
        sources.append(row)

    gate_pass = not unresolved and not missing_roots
    identifiers = [
        {
            "identifier_type": identifier_type,
            "value": value,
            "artifact_roles": sorted(identifier_roles[(identifier_type, value)]),
            "evidence": sorted(evidence),
            "test_eligible": False,
        }
        for (identifier_type, value), evidence in sorted(identifier_evidence.items())
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_exposure_registry",
        "command": "python scripts/expansion/freeze_exposure_registry.py --repo-root . --include-tracked-and-local-results --remote-lineage-ledger configs/expansion/exposed_sources_v1.yaml --output results/expansion/governance/exposure_registry.json --fail-on-unresolved-unblacklisted",
        "sources": sources,
        "identifiers": identifiers,
        "inputs": inputs,
        "pool_blacklists": ledger.get("pool_blacklists", []),
        "unresolved": unresolved,
        "missing_required_roots": missing_roots,
        "counts": {
            "source_union_count": len(sources),
            "identifier_union_count": len(identifiers),
            "input_count": len(inputs),
            "pool_blacklist_count": len(ledger.get("pool_blacklists", [])),
        },
        "gate": {
            "status": "GO" if gate_pass else "NO_GO",
            "unresolved_unblacklisted_count": len(unresolved),
            "missing_required_root_count": len(missing_roots),
            "next_action": "FREEZE_D0_PROTOCOL" if gate_pass else "RESOLVE_OR_BLACKLIST_LINEAGE",
        },
    }
    digest_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["registry_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--include-tracked-and-local-results", action="store_true")
    parser.add_argument("--remote-lineage-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-unresolved-unblacklisted", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    ledger_path = args.remote_lineage_ledger
    if not ledger_path.is_absolute():
        ledger_path = repo_root / ledger_path
    ledger = json.loads(ledger_path.read_text())
    payload = build_registry(repo_root, ledger, args.include_tracked_and_local_results)
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), **payload["counts"], **payload["gate"]}, sort_keys=True))
    if args.fail_on_unresolved_unblacklisted and payload["gate"]["status"] != "GO":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
