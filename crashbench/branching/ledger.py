"""Append-only logical branch attempt ledger with atomic acceptance."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


TERMINAL_STATUSES = frozenset(
    {"accepted", "deterministic_invalid", "operational_failed", "abandoned"}
)


@dataclass(frozen=True)
class LogicalAttemptKey:
    protocol_sha256: str
    git_commit: str
    checkpoint_sha256: str
    policy_id: str
    mechanism_id: str
    source_id: str
    condition_id: str
    severity_id: str
    anchor_id: str
    option_id: str
    declared_branch_seed: int | str

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PendingAttempt:
    logical_key_sha256: str
    retry_ordinal: int
    path: Path


class AttemptLedger:
    def __init__(self, root: str | Path, *, max_operational_retries: int = 2):
        self.root = Path(root)
        self.max_operational_retries = int(max_operational_retries)
        if self.max_operational_retries < 0:
            raise ValueError("max_operational_retries cannot be negative")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "pending").mkdir(exist_ok=True)
        (self.root / "terminal").mkdir(exist_ok=True)
        (self.root / "accepted").mkdir(exist_ok=True)
        self.events_path = self.root / "attempt_events.jsonl"
        self.lock_path = self.root / ".ledger.lock"

    def _locked_events(self):
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        handle = os.fdopen(descriptor, "r+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        events = []
        if self.events_path.is_file():
            events = [
                json.loads(line)
                for line in self.events_path.read_text().splitlines()
                if line.strip()
            ]
        return handle, events

    def _append_event(self, event: Mapping[str, Any]) -> None:
        descriptor = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(
                descriptor,
                (json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n").encode(),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def begin(self, key: LogicalAttemptKey, *, retry_ordinal: int) -> PendingAttempt:
        logical = key.sha256()
        ordinal = int(retry_ordinal)
        if ordinal < 0 or ordinal > self.max_operational_retries:
            raise ValueError(
                f"retry_ordinal {ordinal} exceeds allowed 0..{self.max_operational_retries}"
            )
        lock, events = self._locked_events()
        try:
            relevant = [row for row in events if row.get("logical_key_sha256") == logical]
            if any(row.get("status") == "accepted" for row in relevant):
                raise RuntimeError("logical branch is already accepted")
            if any(row.get("status") == "deterministic_invalid" for row in relevant):
                raise RuntimeError("deterministic invalidity cannot be retried")
            used = {int(row["retry_ordinal"]) for row in relevant}
            if ordinal in used:
                raise RuntimeError(f"retry ordinal already used: {ordinal}")
            if ordinal != len(used):
                raise RuntimeError(
                    f"retry ordinals must be contiguous; got {ordinal}, expected {len(used)}"
                )
            path = self.root / "pending" / f"{logical}.retry{ordinal}"
            path.mkdir(exist_ok=False)
            event = {
                "schema_version": 1,
                "event": "attempt_started",
                "logical_key_sha256": logical,
                "logical_key": asdict(key),
                "retry_ordinal": ordinal,
                "status": "pending",
                "path": path.relative_to(self.root).as_posix(),
            }
            self._append_event(event)
            return PendingAttempt(logical, ordinal, path)
        finally:
            lock.close()

    def finalize(
        self,
        pending: PendingAttempt,
        *,
        status: str,
        artifact_manifest_sha256: str | None = None,
        failure_reason: str | None = None,
    ) -> Path:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal attempt status: {status}")
        if status == "accepted" and not artifact_manifest_sha256:
            raise ValueError("accepted attempts require artifact_manifest_sha256")
        if status != "accepted" and not failure_reason:
            raise ValueError("non-accepted attempts require failure_reason")
        lock, events = self._locked_events()
        try:
            matching = [
                row
                for row in events
                if row.get("logical_key_sha256") == pending.logical_key_sha256
                and int(row.get("retry_ordinal", -1)) == pending.retry_ordinal
            ]
            if len(matching) != 1 or matching[0].get("status") != "pending":
                raise RuntimeError("pending attempt does not match exactly one started event")
            if not pending.path.is_dir():
                raise RuntimeError("pending attempt directory is missing")
            if status == "accepted":
                destination = self.root / "accepted" / pending.logical_key_sha256
                if destination.exists():
                    raise RuntimeError("conflicting accepted attempt already exists")
            else:
                destination = (
                    self.root
                    / "terminal"
                    / f"{pending.logical_key_sha256}.retry{pending.retry_ordinal}.{status}"
                )
            os.rename(pending.path, destination)
            event = {
                "schema_version": 1,
                "event": "attempt_finalized",
                "logical_key_sha256": pending.logical_key_sha256,
                "retry_ordinal": pending.retry_ordinal,
                "status": status,
                "path": destination.relative_to(self.root).as_posix(),
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "failure_reason": failure_reason,
            }
            self._append_event(event)
            return destination
        finally:
            lock.close()

    def accepted_count(self) -> int:
        return sum(1 for path in (self.root / "accepted").iterdir() if path.is_dir())
