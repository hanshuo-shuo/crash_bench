#!/usr/bin/env python3
"""Read-only, append-only inventory of historical E14 glass artifacts.

This tool never imports LIBERO/OpenVLA and never writes below ``--source-root``.
It records what evidence exists and whether a historical attempt is a candidate
for *recollection* at a new exact H.  It never promotes an old pair to schema v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.glass_recovery_data import canonical_sha256


AUDIT_SCHEMA_VERSION = 1
CANONICAL_AUDIT_NAME = "core_salvage_audit.jsonl"
LEDGER_NAMES = ("attempts.jsonl", "attempt_ledger.jsonl", "collection_attempts.jsonl")
EVIDENCE_NAMES = (
    "pair.json",
    "precrash_onpath_state.npy",
    "matched_robot_state.npy",
    "controller_state.npz",
    "nominal_catastrophe.npz",
    "nominal_replay.json",
    "oracle_recovery.npz",
    "oracle_search.json",
    "off_path_control.npz",
    "offpath_probe.json",
    "careful_gate.json",
    "careful_annotation.json",
    "blocked_safe_abort.npz",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append({**value, "_ledger_path": path, "_ledger_line": line_number})
    return rows


def _record_map(pair_payload: Mapping[str, Any]) -> dict[str, dict]:
    return {
        str(row.get("trajectory_kind")): dict(row)
        for row in pair_payload.get("records", [])
        if isinstance(row, Mapping) and row.get("trajectory_kind")
    }


def _resolve_array_path(source_root: Path, pair_root: Path, declared: str) -> Path | None:
    candidates = (
        pair_root / Path(declared).name,
        source_root / declared,
        source_root / "dataset" / declared,
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _evidence_files(source_root: Path, pair_root: Path, records: Mapping[str, dict]) -> dict:
    files: dict[str, dict] = {}
    candidates = [pair_root / name for name in EVIDENCE_NAMES]
    for record in records.values():
        declared = record.get("arrays_path")
        if declared:
            resolved = _resolve_array_path(source_root, pair_root, str(declared))
            if resolved is not None:
                candidates.append(resolved)
    for path in sorted(set(candidates)):
        if not path.is_file():
            continue
        relative = path.resolve().relative_to(source_root).as_posix()
        files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return files


def _safe_json(path: Path) -> dict | None:
    return _read_json(path) if path.is_file() else None


def _attempt_key(row: Mapping[str, Any]) -> str | None:
    value = row.get("attempt_key")
    if value:
        return str(value)
    identity = row.get("attempt_identity")
    if isinstance(identity, Mapping) and identity.get("attempt_key"):
        return str(identity["attempt_key"])
    return None


def _pair_attempt_key(pair: Mapping[str, Any], records: Mapping[str, dict]) -> str | None:
    key = _attempt_key(pair)
    if key:
        return key
    values = {
        str(row.get("metadata", {}).get("attempt_key"))
        for row in records.values()
        if row.get("metadata", {}).get("attempt_key")
    }
    return next(iter(values)) if len(values) == 1 else None


def discover_attempts(source_root: Path) -> list[dict]:
    """Return terminal ledger attempts plus pair-only historical attempts."""

    attempts: dict[str, dict] = {}
    for name in LEDGER_NAMES:
        for ledger in sorted(source_root.rglob(name)):
            latest: dict[str, dict] = {}
            for row in _jsonl_rows(ledger):
                explicit = _attempt_key(row)
                identity = explicit or canonical_sha256({
                    "ledger": ledger.relative_to(source_root).as_posix(),
                    "line": row["_ledger_line"],
                    "placement_id": row.get("placement_id"),
                    "rollout_seed": row.get("rollout_seed"),
                    "event": row.get("event"),
                })
                latest[identity] = row
            for identity, row in latest.items():
                if row.get("event") not in {
                    "accepted", "rejected", "failed", "skipped_deterministic_rejection"
                }:
                    continue
                attempts[identity] = {
                    "attempt_id": identity,
                    "identity_quality": "explicit_attempt_key" if _attempt_key(row) else "ledger_line_synthetic",
                    "ledger": row,
                    "pair_path": None,
                }

    for pair_path in sorted(source_root.rglob("pair.json")):
        pair = _read_json(pair_path)
        records = _record_map(pair)
        explicit = _pair_attempt_key(pair, records)
        identity = explicit or canonical_sha256({
            "pair_path": pair_path.relative_to(source_root).as_posix(),
            "pair_sha256": file_sha256(pair_path),
        })
        attempt = attempts.setdefault(identity, {
            "attempt_id": identity,
            "identity_quality": "explicit_attempt_key" if explicit else "pair_path_synthetic",
            "ledger": None,
            "pair_path": None,
        })
        if attempt["pair_path"] is not None and attempt["pair_path"] != pair_path:
            attempt["duplicate_pair_paths"] = [attempt["pair_path"], pair_path]
        attempt["pair_path"] = pair_path
    return [attempts[key] for key in sorted(attempts)]


def _build_audit_row(
    source_root: Path,
    attempt: Mapping[str, Any],
    *,
    target_h: int,
    expected_run_commit: str,
    expected_checkpoint_revision: str,
    placement_retry_counts: Mapping[str, int],
) -> dict:
    ledger = attempt.get("ledger") or {}
    pair_path = attempt.get("pair_path")
    pair = _read_json(pair_path) if isinstance(pair_path, Path) else {}
    records = _record_map(pair)
    nominal = records.get("nominal_catastrophe", {})
    oracle = records.get("oracle_recovery", {})
    offpath = records.get("off_path_control", {})
    nominal_meta = nominal.get("metadata", {})
    oracle_meta = oracle.get("metadata", {})
    pair_root = pair_path.parent if isinstance(pair_path, Path) else source_root
    files = _evidence_files(source_root, pair_root, records)
    placement_id = str(
        ledger.get("placement_id")
        or nominal.get("placement_id")
        or nominal.get("pair_id")
        or "unknown"
    )
    nominal_replay = _safe_json(pair_root / "nominal_replay.json")
    oracle_search = _safe_json(pair_root / "oracle_search.json")
    offpath_probe = _safe_json(pair_root / "offpath_probe.json")
    careful = (
        nominal_meta.get("careful_comparator")
        or _safe_json(pair_root / "careful_annotation.json")
        or _safe_json(pair_root / "careful_gate.json")
    )
    suffix_length = nominal.get("n_steps")
    if suffix_length is None and nominal_replay:
        suffix_length = nominal_replay.get("actions")
    blockers = []
    required_exact = (
        pair_root / "precrash_onpath_state.npy",
        pair_root / "controller_state.npz",
    )
    if not all(path.is_file() for path in required_exact):
        blockers.append("missing_exact_state_or_controller")
    if not nominal:
        blockers.append("missing_nominal_record")
    if not oracle:
        blockers.append("missing_oracle_record")
    if not offpath:
        blockers.append("missing_off_path_record")
    if not isinstance(suffix_length, int) or suffix_length < target_h:
        blockers.append("old_suffix_shorter_than_target_h")
    action_replay = nominal_meta.get("action_replay_evidence", {})
    static_replay_verified = bool(
        action_replay.get("verified") is True
        or (nominal_replay and nominal_replay.get("verified") is True)
    )
    if not static_replay_verified:
        blockers.append("nominal_replay_not_statically_verified")
    verification = oracle_meta.get("oracle_verification", {})
    independent_oracle = bool(
        verification.get("independent_recapture") is True
        and verification.get("recapture_success") is True
    )
    if not independent_oracle:
        blockers.append("oracle_independent_recapture_missing")
    offpath_success = bool(
        offpath.get("succeeded") is True
        or (offpath_probe and offpath_probe.get("succeeded") is True)
    )
    if not offpath_success:
        blockers.append("off_path_task_success_missing")

    code_commit = str(
        ledger.get("code_commit")
        or pair.get("attempt_identity", {}).get("code_commit")
        or "unknown"
    )
    checkpoint_revision = str(
        ledger.get("checkpoint_revision")
        or pair.get("attempt_identity", {}).get("checkpoint_revision")
        or "unknown"
    )
    provenance_matches = {
        "run_commit": code_commit == expected_run_commit,
        "checkpoint_revision": checkpoint_revision == expected_checkpoint_revision,
    }
    row = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "kind": "glass_core_salvage_attempt_inventory",
        "attempt_id": attempt["attempt_id"],
        "attempt_identity_quality": attempt["identity_quality"],
        "placement_id": placement_id,
        "split": ledger.get("split") or nominal.get("split"),
        "terminal_event": ledger.get("event") or ("accepted_pair_artifact" if pair else "unknown"),
        "rejection_reason": ledger.get("reason"),
        "source_root": str(source_root),
        "source_root_read_only": True,
        "pair_path": (
            pair_path.relative_to(source_root).as_posix()
            if isinstance(pair_path, Path) else None
        ),
        "provenance": {
            "code_commit": code_commit,
            "checkpoint_revision": checkpoint_revision,
            "rollout_seed": ledger.get("rollout_seed")
            or pair.get("attempt_identity", {}).get("rollout_seed"),
            "protocol_sha256": ledger.get("protocol_sha256")
            or pair.get("attempt_identity", {}).get("protocol_sha256"),
            "expected_identity_matches": provenance_matches,
        },
        "hashes": {
            "source_state_sha256": nominal.get("source_state_sha256"),
            "nominal_scene_sha256": nominal.get("scene_sha256"),
            "oracle_scene_sha256": oracle.get("scene_sha256"),
            "off_path_scene_sha256": offpath.get("scene_sha256"),
            "nominal_start_state_sha256": nominal.get("branch_start_state_sha256"),
            "oracle_start_state_sha256": oracle.get("branch_start_state_sha256"),
            "controller_state_sha256": nominal_meta.get("controller_state_sha256"),
        },
        "files": files,
        "old_suffix_length_actions": suffix_length,
        "nominal_replay": {
            "artifact_present": nominal_replay is not None,
            "verified": static_replay_verified,
            "metadata": action_replay or nominal_replay,
        },
        "oracle": {
            "search_artifact_present": oracle_search is not None,
            "capture_artifact_present": bool(oracle),
            "independent_recapture_verified": independent_oracle,
            "verification": verification,
        },
        "off_path": {
            "probe_artifact_present": offpath_probe is not None,
            "task_succeeded": offpath_success,
            "outcome": offpath.get("outcome"),
        },
        "careful": careful,
        "retry_integrity": {
            "attempts_for_placement": int(placement_retry_counts.get(placement_id, 1)),
            "duplicate_pair_paths_for_attempt": [
                path.relative_to(source_root).as_posix()
                for path in attempt.get("duplicate_pair_paths", [])
            ],
            "possible_overwrite": bool(
                placement_retry_counts.get(placement_id, 1) > 1
                and attempt["identity_quality"] != "explicit_attempt_key"
            ),
        },
        "realignment": {
            "target_h_actions": target_h,
            "advance_captured_actions": (
                int(suffix_length) - target_h
                if isinstance(suffix_length, int) and suffix_length >= target_h
                else None
            ),
            "static_candidate_for_recollection": not blockers,
            "direct_v2_promotion_allowed": False,
            "required_next_steps": [
                "restore historical exact simulator/controller state",
                "advance captured nominal actions to exact target H",
                "verify catastrophe on action H with repaired predicates",
                "rerun and independently recapture the oracle",
                "recollect matched off-path task-success control",
                "write fresh schema-v2 records under a new attempt identity",
            ],
            "blockers": blockers,
        },
    }
    row["source_evidence_sha256"] = canonical_sha256(row)
    return row


def _existing_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for value in _jsonl_rows(path):
        value.pop("_ledger_path", None)
        value.pop("_ledger_line", None)
        attempt_id = str(value.get("attempt_id", ""))
        if not attempt_id or attempt_id in rows:
            raise ValueError(f"existing audit has empty/duplicate attempt_id {attempt_id!r}")
        rows[attempt_id] = value
    return rows


def run_audit(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    summary_out = Path(args.summary_out).resolve()
    if not args.read_only:
        raise SystemExit("--read-only is required")
    if not source_root.is_dir():
        raise SystemExit(f"source root does not exist: {source_root}")
    for target in (output, summary_out):
        try:
            target.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise SystemExit("audit outputs must be outside the historical source root")

    source_before = {
        path.relative_to(source_root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in source_root.rglob("*") if path.is_file()
    }
    attempts = discover_attempts(source_root)
    placement_counts = Counter()
    for attempt in attempts:
        ledger = attempt.get("ledger") or {}
        pair = _read_json(attempt["pair_path"]) if attempt.get("pair_path") else {}
        records = _record_map(pair)
        nominal = records.get("nominal_catastrophe", {})
        placement_counts[str(
            ledger.get("placement_id") or nominal.get("placement_id")
            or nominal.get("pair_id") or "unknown"
        )] += 1
    rows = [
        _build_audit_row(
            source_root, attempt, target_h=args.target_h,
            expected_run_commit=args.expected_run_commit,
            expected_checkpoint_revision=args.expected_checkpoint_revision,
            placement_retry_counts=placement_counts,
        )
        for attempt in attempts
    ]
    existing = _existing_rows(output)
    new_rows = []
    for row in rows:
        prior = existing.get(row["attempt_id"])
        if prior is not None:
            if prior.get("source_evidence_sha256") != row["source_evidence_sha256"]:
                raise RuntimeError(
                    f"historical evidence drift for audited attempt {row['attempt_id']}"
                )
            continue
        new_rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    if new_rows:
        with output.open("a") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    if summary_out.exists() and not args.overwrite_summary:
        raise SystemExit(f"refusing to overwrite {summary_out}; pass --overwrite-summary")
    all_rows = [*existing.values(), *new_rows]
    summary = {
        "schema_version": 1,
        "kind": "glass_core_h_realignment_inventory_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_root_read_only": True,
        "audit_jsonl": str(output),
        "target_h_actions": args.target_h,
        "attempts": len(all_rows),
        "newly_appended_attempts": len(new_rows),
        "unique_placements": len({row["placement_id"] for row in all_rows}),
        "terminal_events": dict(Counter(row["terminal_event"] for row in all_rows)),
        "rejection_reasons": dict(Counter(
            row["rejection_reason"] for row in all_rows if row["rejection_reason"]
        )),
        "static_recollection_candidates": sum(
            row["realignment"]["static_candidate_for_recollection"] for row in all_rows
        ),
        "possible_overwrites": sum(
            row["retry_integrity"]["possible_overwrite"] for row in all_rows
        ),
        "direct_v2_promotions": 0,
        "replayability_status": (
            "requires_real_LIBERO_replay_and_oracle_recollection"
            if all_rows else "no_attempts_discovered"
        ),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    source_after = {
        path.relative_to(source_root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in source_root.rglob("*") if path.is_file()
    }
    if source_after != source_before:
        raise RuntimeError("historical source root changed during read-only audit")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--target-h", type=int, default=20)
    parser.add_argument("--expected-run-commit", required=True)
    parser.add_argument("--expected-checkpoint-revision", required=True)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--overwrite-summary", action="store_true")
    args = parser.parse_args()
    if args.target_h < 1:
        raise SystemExit("target H must be positive")
    summary = run_audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
