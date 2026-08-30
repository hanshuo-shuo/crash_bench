#!/usr/bin/env python3
"""Freeze deterministic splits for selected formal staleness-pilot sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.data.source_registry import ExposureRegistry
from crashbench.data.splits import SourceRecord, freeze_split_manifest


def canonical_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-analysis", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite split manifest: {args.output}")
    analysis = json.loads(args.formal_analysis.read_text())
    if analysis.get("status") != "GO" or analysis.get("option_outcomes_opened") != 0:
        raise ValueError("formal nominal analysis is not GO and outcome-free")
    plan = json.loads(args.formal_plan.read_text())
    split_config = json.loads(args.split_config.read_text())
    allowed_attempts = frozenset(row["attempt_id"] for row in plan["attempts"])
    policy_identity = (
        "pi0_libero:15a9616a00943ada6c20a0f158e3adb39df2ccac:"
        "ea876fed5d324aa6a61e8a6a31b65ab233de3c1d02b42791613c60ea84daeb8b"
    )
    records = []
    for row in analysis["selected_sources"]:
        physical = row["physical_source_id"]
        mechanism_source = canonical_hash(
            physical, "observation_staleness_v1", "delay_1,delay_3,delay_5"
        )
        policy_source = canonical_hash(mechanism_source, policy_identity)
        records.append(
            SourceRecord(
                physical_source_id=physical,
                mechanism_source_id=mechanism_source,
                policy_source_id=policy_source,
                mechanism_id="observation_staleness_v1",
                task_id=f"libero_spatial:{int(row['task_id'])}",
                policy_id="pi0",
                source_state_sha256=row["source_state_sha256"],
                reset_seed=row["reset_seed"],
                scene_fingerprint=row["scene_fingerprint"],
                formal_attempt_id=row["attempt_id"],
            )
        )
    counts = split_config["primary_policy_per_mechanism_task"]
    manifest = freeze_split_manifest(
        records,
        counts_by_cell=counts,
        protocol_sha256=args.protocol_sha256,
        exposure_registry=ExposureRegistry.load(args.exposure_registry),
        prospective_formal_attempt_ids=allowed_attempts,
    )
    manifest["formal_analysis_sha256"] = analysis["analysis_sha256"]
    manifest["formal_plan_sha256"] = plan["plan_sha256"]
    # Re-pin after adding parent provenance.
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "physical_sources": manifest["physical_source_count"], "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
