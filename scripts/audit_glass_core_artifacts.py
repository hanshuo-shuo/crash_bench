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
import re
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


def _rejection_reason_from_log(message: str) -> str:
    if (
        "nominal placement did not produce a catastrophe" in message
        or "captured nominal actions failed exact-state replay" in message
    ):
        return "no_base_crash"
    if "careful-prompt policy did not crash" in message:
        return "careful_did_not_crash"
    if "no safe task-completing oracle found" in message:
        return "oracle_task_failure"
    if "oracle" in message.lower() and "collision" in message.lower():
        return "oracle_collision"
    return "other"


def reconstruct_attempts_from_logs(
    log_specs: Iterable[str],
    historical_summary: Path,
) -> list[dict]:
    """Reconstruct unique E14 attempts from immutable job/ordinal log evidence."""

    payload = _read_json(historical_summary)
    provenance = payload.get("provenance", {})
    aggregate = payload.get("aggregate_attempt_accounting", {})
    code_commit = str(provenance.get("git_commit", "unknown"))
    checkpoint_revision = str(provenance.get("checkpoint_revision", "unknown"))
    placement_manifest_sha256 = str(
        provenance.get("placement_manifest_sha256", "unknown")
    )
    job_rows = provenance.get("slurm_jobs", [])
    jobs = {str(row.get("job_id")): row for row in job_rows}
    if not jobs or len(jobs) != len(job_rows):
        raise ValueError("historical summary has missing/duplicate Slurm job IDs")

    specs: dict[str, Path] = {}
    for spec in log_specs:
        job_id, separator, path_text = str(spec).partition("=")
        if not separator or not job_id.isdigit() or not path_text:
            raise ValueError("--attempt-log must have the form JOB_ID=PATH")
        if job_id in specs:
            raise ValueError(f"duplicate attempt log for Slurm job {job_id}")
        specs[job_id] = Path(path_text).resolve()
    if set(specs) != set(jobs):
        raise ValueError(
            "attempt logs must cover exactly the historical Slurm jobs: "
            f"logs={sorted(specs)} summary={sorted(jobs)}"
        )

    attempts: list[dict] = []
    collect_pattern = re.compile(
        r"^COLLECT\s+(\S+)\s+split=(\S+)\s+fraction=(\S+)\s*$"
    )
    terminal_pattern = re.compile(r"^(ACCEPT|REJECT)\s+([^:;]+)[:;](.*)$")
    for job_id, log_path in specs.items():
        if not log_path.is_file():
            raise FileNotFoundError(f"attempt log does not exist: {log_path}")
        lines = log_path.read_text(errors="replace").splitlines()
        log_sha256 = file_sha256(log_path)
        declared_commits = {
            match.group(1)
            for line in lines
            if (match := re.search(r"\bCODE_COMMIT=([0-9a-f]{40})\b", line))
        }
        if declared_commits and declared_commits != {code_commit}:
            raise ValueError(
                f"Slurm job {job_id} log commit {sorted(declared_commits)} "
                f"does not match historical summary {code_commit}"
            )

        pending: dict | None = None
        job_attempts: list[dict] = []
        for line_number, line in enumerate(lines, 1):
            collect_match = collect_pattern.match(line)
            if collect_match:
                if pending is not None:
                    raise ValueError(
                        f"{log_path}:{line_number} starts a new attempt before "
                        f"{pending['placement_id']} has a terminal outcome"
                    )
                pending = {
                    "placement_id": collect_match.group(1),
                    "split": collect_match.group(2),
                    "nominal_fraction": float(collect_match.group(3)),
                    "collect_line": line_number,
                }
                continue
            terminal_match = terminal_pattern.match(line)
            if terminal_match is None:
                continue
            if pending is None:
                raise ValueError(f"{log_path}:{line_number} has an orphan terminal event")
            event_name, placement_id, terminal_text = terminal_match.groups()
            if placement_id != pending["placement_id"]:
                raise ValueError(
                    f"{log_path}:{line_number} terminal placement {placement_id} "
                    f"does not match pending {pending['placement_id']}"
                )
            ordinal = len(job_attempts)
            event = "accepted" if event_name == "ACCEPT" else "rejected"
            reason = (
                None if event == "accepted"
                else _rejection_reason_from_log(terminal_text)
            )
            identity_payload = {
                "identity_kind": "e14_slurm_log_reconstruction_v1",
                "slurm_job_id": job_id,
                "attempt_ordinal": ordinal,
                "placement_id": placement_id,
                "split": pending["split"],
                "code_commit": code_commit,
                "checkpoint_revision": checkpoint_revision,
                "placement_manifest_sha256": placement_manifest_sha256,
            }
            attempt_id = canonical_sha256(identity_payload)
            job_attempts.append({
                "attempt_id": attempt_id,
                "identity_quality": "slurm_log_reconstructed",
                "ledger": {
                    "event": event,
                    "placement_id": placement_id,
                    "split": pending["split"],
                    "reason": reason,
                    "error": terminal_text.strip() if event == "rejected" else None,
                    "code_commit": code_commit,
                    "checkpoint_revision": checkpoint_revision,
                    "rollout_seed": 0,
                    "_ledger_path": log_path,
                    "_ledger_line": line_number,
                },
                "pair_path": None,
                "reconstruction": {
                    **identity_payload,
                    "attempt_id": attempt_id,
                    "log_path": str(log_path),
                    "log_sha256": log_sha256,
                    "collect_line": pending["collect_line"],
                    "terminal_line": line_number,
                    "terminal_event": event,
                    "terminal_text": terminal_text.strip(),
                    "nominal_fraction_logged": pending["nominal_fraction"],
                    "commit_evidence": (
                        "job_log" if declared_commits else "tracked_historical_summary"
                    ),
                },
            })
            pending = None
        if pending is not None:
            raise ValueError(
                f"{log_path} ends before {pending['placement_id']} has a terminal outcome"
            )

        expected_job = jobs[job_id]
        expected_count = int(expected_job.get("candidate_rollout_attempts", -1))
        if len(job_attempts) != expected_count:
            raise ValueError(
                f"Slurm job {job_id} reconstructed {len(job_attempts)} attempts, "
                f"expected {expected_count}"
            )
        actual_accepted = sorted(
            row["ledger"]["placement_id"] for row in job_attempts
            if row["ledger"]["event"] == "accepted"
        )
        expected_accepted = sorted(str(value) for value in (
            expected_job.get("newly_accepted") or []
        ))
        if actual_accepted != expected_accepted:
            raise ValueError(
                f"Slurm job {job_id} accepted IDs {actual_accepted} "
                f"do not match {expected_accepted}"
            )
        actual_rejections = dict(Counter(
            row["ledger"]["reason"] for row in job_attempts
            if row["ledger"]["event"] == "rejected"
        ))
        expected_rejections = {
            str(key): int(value) for key, value in
            expected_job.get("rejection_counts", {}).items()
            if int(value) != 0
        }
        if actual_rejections != expected_rejections:
            raise ValueError(
                f"Slurm job {job_id} rejection counts {actual_rejections} "
                f"do not match {expected_rejections}"
            )
        attempts.extend(job_attempts)

    reported = int(aggregate.get("candidate_rollout_attempts", -1))
    if len(attempts) != reported:
        raise ValueError(
            f"reconstructed {len(attempts)} total attempts, expected {reported}"
        )
    attempt_ids = {row["attempt_id"] for row in attempts}
    if len(attempt_ids) != len(attempts):
        raise ValueError("reconstructed attempt IDs are not unique")
    return attempts


def discover_attempts(
    source_root: Path,
    reconstructed_attempts: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict]:
    """Return terminal ledger attempts plus pair-only historical attempts."""

    attempts: dict[str, dict] = {
        str(row["attempt_id"]): dict(row) for row in (reconstructed_attempts or [])
    }
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

    # E14 predates the append-only attempt ledger.  Without complete job logs,
    # its final collection
    # summary is the only surviving terminal record for rejected placements,
    # so inventory those rows without pretending that the synthetic IDs recover
    # identities for retries from earlier jobs.
    for summary_path in (
        [] if reconstructed_attempts else sorted(source_root.rglob("collection_summary.json"))
    ):
        summary = _read_json(summary_path)
        metadata = summary.get("metadata", {})
        checkpoint = metadata.get("checkpoint_identity", {})
        checkpoint_revision = (
            checkpoint.get("resolved_revision")
            or checkpoint.get("model_config_commit_hash")
            or "unknown"
        )
        rejected = metadata.get("rejected", [])
        if not isinstance(rejected, list):
            raise ValueError(f"{summary_path} metadata.rejected must be a list")
        for index, rejected_row in enumerate(rejected):
            if not isinstance(rejected_row, Mapping):
                raise ValueError(
                    f"{summary_path} metadata.rejected[{index}] must be an object"
                )
            identity = canonical_sha256({
                "collection_summary": summary_path.relative_to(source_root).as_posix(),
                "rejected_index": index,
                "placement_id": rejected_row.get("placement_id"),
                "split": rejected_row.get("split"),
                "reason": rejected_row.get("reason"),
                "error": rejected_row.get("error"),
            })
            attempts.setdefault(identity, {
                "attempt_id": identity,
                "identity_quality": "collection_summary_index_synthetic",
                "ledger": {
                    "event": "rejected",
                    "placement_id": rejected_row.get("placement_id"),
                    "split": rejected_row.get("split"),
                    "reason": rejected_row.get("reason"),
                    "error": rejected_row.get("error"),
                    "code_commit": metadata.get("code_commit", "unknown"),
                    "checkpoint_revision": checkpoint_revision,
                    "_ledger_path": summary_path,
                    "_ledger_line": index + 1,
                },
                "pair_path": None,
            })

    for pair_path in sorted(source_root.rglob("pair.json")):
        pair = _read_json(pair_path)
        records = _record_map(pair)
        explicit = _pair_attempt_key(pair, records)
        accepted_matches = [
            row for row in attempts.values()
            if row.get("ledger", {}).get("event") == "accepted"
            and str(row.get("ledger", {}).get("placement_id")) == str(
                next(iter(records.values()), {}).get("placement_id")
                or next(iter(records.values()), {}).get("pair_id")
            )
        ]
        if explicit:
            identity = explicit
        elif len(accepted_matches) == 1:
            identity = str(accepted_matches[0]["attempt_id"])
        elif accepted_matches:
            raise ValueError(
                f"accepted pair {pair_path} matches multiple reconstructed attempts"
            )
        else:
            identity = canonical_sha256({
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
            "attempt_reconstruction": attempt.get("reconstruction"),
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
                and attempt["identity_quality"] not in {
                    "explicit_attempt_key", "slurm_log_reconstructed"
                }
            ),
            "raw_directory_reused_across_attempts": bool(
                placement_retry_counts.get(placement_id, 1) > 1
            ),
            "unique_attempt_identity_preserved": attempt["identity_quality"] in {
                "explicit_attempt_key", "slurm_log_reconstructed"
            },
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


def _historical_attempt_accounting(
    historical_summary: Path | None,
    rows: list[dict],
) -> dict | None:
    """Compare discoverable raw records with the immutable E14 accounting.

    This deliberately fails closed.  A stable synthetic ID for a surviving
    summary row does not recover the identity of an overwritten stochastic
    retry from an earlier job.
    """

    if historical_summary is None:
        return None
    payload = _read_json(historical_summary)
    aggregate = payload.get("aggregate_attempt_accounting", {})
    reported = int(aggregate.get("candidate_rollout_attempts", 0))
    reported_rejected = int(aggregate.get("rejected_attempts", 0))
    reported_accepted = int(aggregate.get("accepted_admissions", 0))
    discoverable = len(rows)
    missing = max(0, reported - discoverable)
    note = str(aggregate.get("accounting_note", ""))
    retries_reported = "retr" in note.lower() or any(
        "retr" in str(value).lower() for value in payload.get("limitations", [])
    )
    explicit = sum(
        row.get("attempt_identity_quality") == "explicit_attempt_key" for row in rows
    )
    reconstructed = sum(
        row.get("attempt_identity_quality") == "slurm_log_reconstructed"
        for row in rows
    )
    identity_covered = explicit + reconstructed
    unique_provenance = bool(
        reported > 0
        and discoverable == reported
        and identity_covered == reported
        and len({str(row.get("attempt_id")) for row in rows}) == reported
    )
    return {
        "historical_summary": str(historical_summary),
        "historical_summary_sha256": file_sha256(historical_summary),
        "reported_candidate_rollout_attempts": reported,
        "reported_rejected_attempts": reported_rejected,
        "reported_accepted_admissions": reported_accepted,
        "discoverable_terminal_attempt_records": discoverable,
        "explicit_attempt_keys": explicit,
        "slurm_log_reconstructed_attempt_keys": reconstructed,
        "unrecoverable_attempt_identity_lower_bound": missing,
        "stochastic_retries_reported": retries_reported,
        "attempts_have_unique_provenance": unique_provenance,
        "accounting_note": note,
    }


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
    historical_summary_arg = getattr(args, "historical_summary", None)
    historical_summary = (
        Path(historical_summary_arg).resolve() if historical_summary_arg else None
    )
    attempt_logs = list(getattr(args, "attempt_log", None) or [])
    if attempt_logs and historical_summary is None:
        raise SystemExit("--attempt-log requires --historical-summary")
    reconstructed_attempts = (
        reconstruct_attempts_from_logs(attempt_logs, historical_summary)
        if attempt_logs and historical_summary is not None else None
    )
    attempts = discover_attempts(source_root, reconstructed_attempts)
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
    historical_accounting = _historical_attempt_accounting(
        historical_summary, all_rows
    )
    no_go_reasons = []
    if (
        historical_accounting is not None
        and not historical_accounting["attempts_have_unique_provenance"]
    ):
        no_go_reasons.append(
            "historical_retry_attempts_lack_unique_preserved_provenance"
        )
        if historical_accounting["unrecoverable_attempt_identity_lower_bound"]:
            no_go_reasons.append("historical_attempt_terminal_records_incomplete")
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
        "raw_directory_reuses": sum(
            row["retry_integrity"]["raw_directory_reused_across_attempts"]
            for row in all_rows
        ),
        "attempt_identity_qualities": dict(Counter(
            row["attempt_identity_quality"] for row in all_rows
        )),
        "direct_v2_promotions": 0,
        "historical_attempt_accounting": historical_accounting,
        "pilot_a_decision": {
            "go": False if no_go_reasons else None,
            "decision": (
                "no_go_stop_before_gpu_realignment"
                if no_go_reasons else "inventory_complete_real_replay_required"
            ),
            "no_go_reasons": no_go_reasons,
            "gpu_realignment_submitted": False,
            "go_checks": {
                "attempts_have_unique_provenance": (
                    None if historical_accounting is None
                    else historical_accounting["attempts_have_unique_provenance"]
                ),
                "exact_h_nominal_suffix_replay_rate": None,
                "oracle_independent_replay_rate": None,
                "repaired_first_action_predicate_checks_pass": None,
            },
        },
        "replayability_status": (
            "blocked_by_historical_attempt_provenance_no_go"
            if no_go_reasons else (
                "requires_real_LIBERO_replay_and_oracle_recollection"
                if all_rows else "no_attempts_discovered"
            )
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
    parser.add_argument(
        "--historical-summary",
        help="immutable tracked E14 summary used to detect missing retry identities",
    )
    parser.add_argument(
        "--attempt-log",
        action="append",
        default=[],
        metavar="JOB_ID=PATH",
        help="complete E14 Slurm log; repeat once for every historical job",
    )
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--overwrite-summary", action="store_true")
    args = parser.parse_args()
    if args.target_h < 1:
        raise SystemExit("target H must be positive")
    summary = run_audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
