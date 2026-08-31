#!/usr/bin/env python3
"""Build the D10 single-mechanism benchmark or scope-failure release package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d5-raw-audit", type=Path, required=True)
    parser.add_argument("--corrected-gate-a", type=Path, required=True)
    parser.add_argument("--gate-a-correction-audit", type=Path, required=True)
    parser.add_argument("--d6-run-manifest", type=Path, required=True)
    parser.add_argument("--gate-b", type=Path, required=True)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--d8-run-root", type=Path, required=True)
    parser.add_argument("--d8-postprocess-dir", type=Path, required=True)
    parser.add_argument("--figure-manifest", type=Path, required=True)
    parser.add_argument("--full-sensitivity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite D10 release root: {args.output_dir}")
    d5_audit = load(args.d5_raw_audit)
    gate_a = load(args.corrected_gate_a)
    correction = load(args.gate_a_correction_audit)
    d6 = load(args.d6_run_manifest)
    gate_b = load(args.gate_b)
    benchmark = load(args.benchmark_freeze)
    d8_post = args.d8_postprocess_dir
    if d8_post.parent.resolve() != args.d8_run_root.resolve():
        raise ValueError("D8 postprocess directory is not inside the authorized run root")
    d8_manifest = load(d8_post / "run_manifest.json")
    d8_analysis = load(d8_post / "benchmark_test_analysis.json")
    figure_manifest = load(args.figure_manifest)
    full_sensitivity = load(args.full_sensitivity)
    test_seal = args.d8_run_root / "authorization" / "test_complete.seal"
    if not test_seal.is_file() or d8_manifest.get("test_complete_sealed") is not True:
        raise ValueError("D10 release requires a sealed one-time D8 test")
    if d5_audit["status"] != "GO" or gate_a["gate"]["hard_failures"]:
        raise ValueError("D10 release inputs fail D5 hard validity")
    if gate_a.get("calibration_rows_read") != 0 or gate_a.get("calibration_sources_excluded") != 12:
        raise ValueError("D10 release requires corrected train+development Gate A")
    if correction.get("continue_same_authorized_d8_run") is not True:
        raise ValueError("D10 release lacks the post-open lineage correction audit")
    if d6["test_rows_read"] != 0 or gate_b["test_rows_read"] != 0:
        raise ValueError("D6/D7 reports pre-test leakage")
    if benchmark["method_claim_authorized"] or d8_analysis["method_superiority_evaluated"]:
        raise ValueError("benchmark-only release cannot contain a method superiority result")
    if figure_manifest.get("method_superiority_depicted") is not False:
        raise ValueError("release figure crosses the method claim boundary")
    if (
        full_sensitivity.get("weight_setting_count") != 108
        or full_sensitivity.get("confirmatory_gate_changed") is not False
        or full_sensitivity.get("analysis_role") != "SUPPLEMENTAL_NON_GATING_PREDECLARED_WEIGHT_GRID"
    ):
        raise ValueError("release lacks the complete non-gating 108-setting utility sensitivity")
    confirmatory_status = d8_analysis["gate"]["status"]
    release_mode = (
        "SCOPED_SINGLE_MECHANISM_BENCHMARK_DATA"
        if confirmatory_status == "GO"
        else "SCOPED_TEST_SCOPE_FAILURE_RELEASE"
    )
    args.output_dir.mkdir(parents=True)
    tables = args.output_dir / "tables"
    tables.mkdir()
    coverage_rows = [
        {
            "phase": "D5_train_calibration_development",
            "role": "mixed_non_test",
            "physical_sources": 48,
            "anchors": d5_audit["complete_blocks"],
            "branches": d5_audit["complete_blocks"] * 3,
            "invalid_blocks": 0,
            "test_sources_read": 0,
        },
        {
            "phase": "D5_Gate_A",
            "role": "train_development_only",
            "physical_sources": gate_a["source_count"],
            "anchors": gate_a["block_decision_count"],
            "branches": gate_a["block_decision_count"] * 3,
            "invalid_blocks": 0,
            "test_sources_read": 0,
        },
        {
            "phase": "D8_B_confirmatory",
            "role": "confirmatory_id_test",
            "physical_sources": d8_manifest["physical_source_shards"],
            "anchors": d8_analysis["primary"]["block_decision_count"],
            "branches": d8_analysis["primary"]["block_decision_count"] * 3,
            "invalid_blocks": int(round(d8_analysis["mechanical_invalid_rate"] * d8_analysis["primary"]["block_decision_count"])),
            "test_sources_read": d8_manifest["test_sources_read"],
        },
    ]
    write_csv(tables / "dataset_coverage.csv", coverage_rows)
    heterogeneity_rows = []
    for phase, payload in (
        ("D5_train_development", gate_a),
        ("D8_confirmatory", d8_analysis["primary"]),
    ):
        heterogeneity_rows.append(
            {
                "phase": phase,
                "effective_physical_n": payload["source_count"],
                "benefit_zero_sources": payload["benefit_zero_sources"],
                "benefit_one_sources": payload["benefit_one_sources"],
                "strict_refresh_sources": payload["strict_support_sources"]["observation_refresh"],
                "strict_safe_stop_sources": payload["strict_support_sources"]["safe_stop"],
                "same_risk_flip_sources": payload["same_risk_flip_sources"],
            }
        )
    write_csv(tables / "heterogeneity_support.csv", heterogeneity_rows)
    gate_rows = [
        {"stage": "D5_Gate_A_corrected", "status": gate_a["gate"]["status"], "method_claim": False, "benchmark_claim": False},
        {"stage": "D7_Gate_B", "status": gate_b["gate"]["status"], "method_claim": False, "benchmark_claim": False},
        {
            "stage": "D8_B_Gate_A_Test", "status": confirmatory_status,
            "method_claim": False, "benchmark_claim": confirmatory_status == "GO",
        },
    ]
    write_csv(tables / "gate_decisions.csv", gate_rows)
    claims = [
        "No multi-mechanism benchmark claim is authorized.",
        "No learned-method superiority claim is authorized.",
        "No sequential safeguard, online recovery, or deployability claim is authorized.",
        (
            "A single-policy, single-staleness-mechanism exact-state realized-option benchmark claim is authorized."
            if confirmatory_status == "GO"
            else "The confirmatory single-mechanism benchmark gate failed; only a complete scope-failure/null release is authorized."
        ),
    ]
    (args.output_dir / "CLAIMS.md").write_text("# Frozen claim boundary\n\n" + "\n".join(f"- {line}" for line in claims) + "\n")
    data_card = f"""# CrashBench scoped staleness data card

- Release mode: `{release_mode}`
- Primary policy: π0
- Tasks: LIBERO-Spatial task 0 and task 2
- Mechanism: observation staleness v1 only
- Train/calibration/development physical sources: 24/12/12
- One-time confirmatory physical sources: 32 (16 per task)
- D8 test status: `{confirmatory_status}`
- Test lock: sealed; no reopen or top-up permitted
- D5 Gate-A correction: calibration excluded; correction discovered after D8 opened and fully disclosed
- Method result: development null/negative; 100% safe-stop degeneration after calibration
- Utility sensitivity: all 108 predeclared settings reported as supplemental non-gating analysis
- Raw observations: content-addressed and Quest-available; compact shards, merged tables, models, and manifests tracked
"""
    (args.output_dir / "DATA_CARD.md").write_text(data_card)
    artifacts = {
        "d5_raw_audit": args.d5_raw_audit,
        "corrected_gate_a": args.corrected_gate_a,
        "gate_a_correction_audit": args.gate_a_correction_audit,
        "d6_run_manifest": args.d6_run_manifest,
        "gate_b": args.gate_b,
        "benchmark_freeze": args.benchmark_freeze,
        "d8_postprocess_manifest": d8_post / "run_manifest.json",
        "d8_analysis": d8_post / "benchmark_test_analysis.json",
        "d8_test_complete_seal": test_seal,
        "figure_manifest": args.figure_manifest,
        "full_utility_sensitivity": args.full_sensitivity,
        "dataset_coverage_table": tables / "dataset_coverage.csv",
        "heterogeneity_table": tables / "heterogeneity_support.csv",
        "gate_table": tables / "gate_decisions.csv",
        "claims": args.output_dir / "CLAIMS.md",
        "data_card": args.output_dir / "DATA_CARD.md",
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_scoped_benchmark_release_manifest",
        "release_mode": release_mode,
        "confirmatory_status": confirmatory_status,
        "method_claim_authorized": False,
        "sequential_claim_authorized": False,
        "multi_mechanism_claim_authorized": False,
        "single_mechanism_benchmark_claim_authorized": confirmatory_status == "GO",
        "test_reopen_permitted": False,
        "test_sources_read": 32,
        "gate_a_correction_disclosed": True,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in artifacts.items()
        },
        "unfinished_required_artifacts": [],
        "frozen": True,
    }
    manifest["release_sha256"] = canonical_sha(manifest)
    (args.output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), "mode": release_mode, "confirmatory_status": confirmatory_status}, sort_keys=True))


if __name__ == "__main__":
    main()
