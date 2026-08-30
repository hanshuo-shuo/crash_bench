#!/usr/bin/env python3
"""Generate the scoped benchmark/manuscript draft only after D10 audit passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-audit", type=Path, required=True)
    parser.add_argument("--corrected-gate-a", type=Path, required=True)
    parser.add_argument("--gate-b", type=Path, required=True)
    parser.add_argument("--d8-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite scoped manuscript: {args.output}")
    release = json.loads((args.release_root / "release_manifest.json").read_text())
    audit = json.loads(args.release_audit.read_text())
    d5 = json.loads(args.corrected_gate_a.read_text())
    gate_b = json.loads(args.gate_b.read_text())
    d8 = json.loads(args.d8_analysis.read_text())
    if audit["status"] != "GO" or release["test_reopen_permitted"] is not False:
        raise ValueError("manuscript generation requires a passed frozen D10 audit")
    if release["method_claim_authorized"] or release["multi_mechanism_claim_authorized"]:
        raise ValueError("scoped manuscript cannot carry method or multi-mechanism claims")
    primary = d8["primary"]
    gate_b_criteria = {
        row["criterion_id"]: row["actual"] for row in gate_b["gate"]["criteria"]
    }
    test_status = d8["gate"]["status"]
    if test_status == "GO":
        headline = (
            "The once-opened 32-source confirmatory split reproduced the predeclared scoped "
            "heterogeneity criteria for observation staleness."
        )
    else:
        headline = (
            "The once-opened 32-source confirmatory split did not satisfy every predeclared scoped "
            "heterogeneity criterion; we release the complete scope-failure result."
        )
    text = f"""# Exact-State Realized Option Utilities under Observation Staleness

## Abstract

Failure risk alone does not identify which finite recovery action improves task–safety utility.
We study this distinction for a frozen π0 policy under one controlled observation-staleness
mechanism in two LIBERO-Spatial tasks. Our exact-state branch engine evaluates Base continuation,
observation refresh, and safe stop from identical simulator, controller, policy-continuation, and
random-number state. Train and development contain 36 independent physical sources; calibration
contains 12 disjoint sources; confirmatory evaluation uses one immutable 32-source split.
{headline} A five-seed option-conditioned model did not beat the frozen DirectQ comparator and its
conformal selector degenerated to safe stop on every development decision. We therefore make no
learned-method, multi-mechanism, sequential-safeguard, or deployability claim. The release retains
all nulls, the post-open Gate-A role-filter correction, source-level denominators, and one-time-test
provenance.

## 1. Question and scope

Given a frozen VLA and a finite deployable option set, can exact-state realized outcomes expose
states with the same canonical Base risk but different beneficial actions? This paper answers only
for π0, observation staleness v1, LIBERO-Spatial tasks 0 and 2, and three options. The broader
multi-mechanism and learned-router objectives did not pass their upstream gates.

## 2. Exact-state benchmark

Every branch restores simulator, controller, policy queue/continuation, environment RNG, and
framework RNG state. D5 collected 48 train/calibration/development sources, 1,296 anchors, and
3,888 complete option outcomes with 100% branch-start hash agreement and zero mechanical-invalid
blocks. Gate A correctly analyzes train+development only (n=36), excluding all 12 calibration
sources. The original Gate-A artifact mistakenly included calibration; this was discovered after
the D8 lock opened, corrected without inspecting D8 outcomes, and did not change D8 identities,
thresholds, status, or next action.

## 3. Development heterogeneity

On train+development, B=0/B=1 support was {d5['benefit_zero_sources']}/{d5['benefit_one_sources']}
physical sources. Strict observation-refresh and safe-stop support were
{d5['strict_support_sources']['observation_refresh']} and
{d5['strict_support_sources']['safe_stop']} sources; {d5['same_risk_flip_sources']} sources showed
same-risk benefit or strict-option flips. The B=0 count missed the frozen target of 8, so this was
`SCOPED_CONTINUE`, not a broad benchmark pass.

## 4. Method null

DirectQ was the strongest frozen deployable comparator. Only
{gate_b_criteria['seeds_beating_comparator']}/5 train-only ODUR seeds exceeded it; median
development delta U0 was {gate_b_criteria['median_delta_u0']:.4f}. Source-conformal
calibration was finite but extremely conservative. The calibrated selector chose safe stop for
100% of development decisions, reducing catastrophe point rate by 0.0679 while decreasing
source-macro U0 by 0.698. Gate B therefore froze benchmark-only mode. No method superiority test
was opened.

## 5. Confirmatory benchmark

The one-time D8-B split contains 32 physical sources, 16 per task. Its machine status is
`{test_status}`. B=0/B=1 support was {primary['benefit_zero_sources']}/
{primary['benefit_one_sources']}; strict refresh/safe-stop support was
{primary['strict_support_sources']['observation_refresh']}/
{primary['strict_support_sources']['safe_stop']}; and {primary['same_risk_flip_sources']} sources
showed a same-risk flip. Exact branch-start rate was {d8['exact_branch_start_rate']:.3f}; mechanical
invalidity was {d8['mechanical_invalid_rate']:.3f}. The confirmatory gate and all catastrophe-cost
sensitivity rows are reported without threshold changes or source top-up.

## 6. Limitations

This is simulation-only, single-policy, single-mechanism, and limited to a three-option catalog.
Safe stop is useful for bounding catastrophe but can destroy task utility, as the method null makes
clear. Exact-state branching is not a real-world safety certificate. Calibration exchangeability
is source-level and does not transfer automatically to new mechanisms, tasks, policies, or robots.
No sequential execution, option-return, or online safeguard claim is made.

## 7. Reproducibility and artifact boundary

The release includes raw shard JSON, compact merged tables, content-addressed feature identities,
five selection and five refit model artifacts, calibration and gate records, Slurm provenance,
the one-time authorization and completion seal, utility sensitivity, machine tables, figures with
effective physical n, and an explicit claim boundary. Raw observation blobs remain content-addressed
and Quest-available. The test cannot be reopened or adaptively topped up.
"""
    if "TBD" in text or "TODO" in text:
        raise ValueError("generated manuscript contains unfinished placeholders")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(json.dumps({"output": str(args.output), "confirmatory_status": test_status}, sort_keys=True))


if __name__ == "__main__":
    main()
