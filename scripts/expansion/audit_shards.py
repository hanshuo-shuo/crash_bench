#!/usr/bin/env python3
"""After-any completeness audit for logical branch shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def audit_logical_shards(
    expected_keys: Iterable[str], events: Iterable[dict[str, Any]], *, max_operational_retries: int = 2
) -> dict[str, Any]:
    expected = tuple(sorted(set(expected_keys)))
    if not expected:
        raise ValueError("shard audit requires at least one expected logical key")
    if any(len(key) != 64 for key in expected):
        raise ValueError("expected logical keys must be SHA-256 strings")
    by_key: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        key = str(event.get("logical_key_sha256", ""))
        by_key.setdefault(key, []).append(event)
    unexpected = sorted(set(by_key) - set(expected))
    rows = []
    for key in expected:
        relevant = by_key.get(key, [])
        finalized = [row for row in relevant if row.get("event") == "attempt_finalized"]
        statuses = [str(row.get("status")) for row in finalized]
        accepted = statuses.count("accepted")
        deterministic_invalid = statuses.count("deterministic_invalid")
        if accepted > 1 or deterministic_invalid > 1 or (accepted and deterministic_invalid):
            state = "CONFLICTING_TERMINALS"
            complete = False
        elif accepted == 1:
            state = "ACCEPTED"
            complete = True
        elif deterministic_invalid == 1:
            state = "DETERMINISTIC_TERMINAL_INVALID"
            complete = True
        else:
            operational_ordinals = [
                int(row.get("retry_ordinal", -1))
                for row in finalized
                if row.get("status") in {"operational_failed", "abandoned"}
            ]
            pending = any(row.get("event") == "attempt_started" for row in relevant) and len(
                finalized
            ) < len([row for row in relevant if row.get("event") == "attempt_started"])
            if operational_ordinals and max(operational_ordinals) >= max_operational_retries:
                state = "OPERATIONAL_RETRIES_EXHAUSTED"
            elif pending:
                state = "PENDING_ATTEMPT"
            elif relevant:
                state = "RETRYABLE_INCOMPLETE"
            else:
                state = "MISSING_ATTEMPT"
            complete = False
        rows.append(
            {
                "logical_key_sha256": key,
                "state": state,
                "complete": complete,
                "attempt_started_count": sum(
                    row.get("event") == "attempt_started" for row in relevant
                ),
                "attempt_finalized_count": len(finalized),
            }
        )
    incomplete = [row["logical_key_sha256"] for row in rows if not row["complete"]]
    status = "GO" if not incomplete and not unexpected else "NO_GO"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_shard_completeness_audit",
        "expected_logical_shards": len(expected),
        "complete_logical_shards": len(expected) - len(incomplete),
        "incomplete_logical_keys": incomplete,
        "unexpected_logical_keys": unexpected,
        "rows": rows,
        "status": status,
        "next_action": "MERGE_ACCEPTED_SHARDS" if status == "GO" else "STOP_MERGE_AND_RESOLVE_OR_REPORT_MISSINGNESS",
    }
    payload["audit_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--max-operational-retries", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite shard audit: {args.output}")
    expected_payload = json.loads(args.expected.read_text())
    expected = expected_payload.get("logical_key_sha256", expected_payload.get("expected_logical_keys"))
    if not isinstance(expected, list):
        raise ValueError("expected manifest lacks logical key list")
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    payload = audit_logical_shards(
        expected, events, max_operational_retries=args.max_operational_retries
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "complete": payload["complete_logical_shards"]}, sort_keys=True))
    if payload["status"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
