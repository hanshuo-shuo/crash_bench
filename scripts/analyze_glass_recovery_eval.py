#!/usr/bin/env python3
"""Source-state clustered, paired analysis for P0-D evaluation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.metrics import analyze_recovery_evaluation
from scripts.eval_glass_recovery import (
    EVALUATION_MODES,
    NONPRIVILEGED_CONDITIONS,
    PRIVILEGED_CONDITIONS,
    SANITY_CONDITIONS,
    condition_role,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True, nargs="+")
    parser.add_argument("--mode", required=True, choices=EVALUATION_MODES)
    parser.add_argument("--reference-condition", default="base")
    parser.add_argument(
        "--panel",
        choices=("nonprivileged", "privileged", "sanity", "all"),
        default="nonprivileged",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = [Path(value).resolve() for value in args.evaluation]
    out = Path(args.out).resolve()
    missing = [source for source in sources if not source.is_file()]
    if missing:
        raise SystemExit(f"evaluation output does not exist: {missing}")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}; pass --overwrite")
    if args.bootstrap_replicates < 0:
        raise SystemExit("bootstrap-replicates cannot be negative")
    payloads = [json.loads(source.read_text()) for source in sources]
    if any(
        payload.get("kind") != "glass_recovery_accepted_cohort_evaluation"
        for payload in payloads
    ):
        raise SystemExit("every input must be a P0-D accepted-cohort evaluation")
    identity_keys = (
        "placement_manifest_sha256",
        "trajectory_manifest_sha256",
        "evaluation_cohort_sha256",
        "primary_protocol_sha256",
        "evaluation_protocol_sha256",
    )
    identities = {
        tuple(payload.get("inputs", {}).get(key) for key in identity_keys)
        for payload in payloads
    }
    if len(identities) != 1 or any(value in (None, "") for value in next(iter(identities))):
        raise SystemExit("evaluation inputs do not share one frozen cohort/protocol identity")
    seed_sets = []
    rows = []
    panel_conditions = {
        "nonprivileged": set(NONPRIVILEGED_CONDITIONS),
        "privileged": set(PRIVILEGED_CONDITIONS),
        "sanity": set(SANITY_CONDITIONS),
    }
    for payload in payloads:
        for row in payload.get("episodes", []):
            condition = str(row.get("condition", ""))
            if row.get("condition_role") not in (None, condition_role(condition)):
                raise SystemExit(f"evaluation row has inconsistent condition role: {condition}")
        payload_rows = [
            row for row in payload.get("episodes", [])
            if row.get("evaluation_mode") == args.mode
            and (
                args.panel == "all"
                or row.get("condition") in panel_conditions[args.panel]
                or row.get("condition") == args.reference_condition
            )
        ]
        seeds = {int(row["training_seed"]) for row in payload_rows}
        if len(seeds) != 1:
            raise SystemExit("each evaluation input must contain one explicit training seed")
        seed_sets.append(next(iter(seeds)))
        rows.extend(payload_rows)
    if len(set(seed_sets)) != len(seed_sets):
        raise SystemExit("duplicate training-seed evaluation inputs would double-count rows")
    analysis = analyze_recovery_evaluation(
        rows,
        reference_condition=args.reference_condition,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    result = {
        "schema_version": 1,
        "kind": "glass_recovery_source_cluster_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluations": [str(source) for source in sources],
        "evaluation_mode": args.mode,
        "panel": args.panel,
        "analysis": analysis,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis, indent=2, sort_keys=True), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
