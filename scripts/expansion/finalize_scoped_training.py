#!/usr/bin/env python3
"""Operationally finalize a D6/D7 run whose training artifacts already sealed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.expansion.hash_tree_manifest import resolve_git_head
from scripts.expansion.run_scoped_training import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--gate-a", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_root.is_dir():
        raise FileNotFoundError("finalizer requires an existing partial training root")
    gate_b = args.run_root / "scoped_gate_b_analysis.json"
    run_manifest = args.run_root / "run_manifest.json"
    if gate_b.exists() or run_manifest.exists():
        raise FileExistsError("refusing to overwrite an existing Gate B or run manifest")
    baseline = args.run_root / "baselines" / "baseline_selection.json"
    development = args.run_root / "development_model_selection.json"
    calibration = args.run_root / "calibration" / "statewise_calibration_freeze.json"
    refit_dirs = [args.run_root / "refit" / f"seed_{seed}" for seed in range(5)]
    required = [baseline, development, calibration, args.feature_cache, args.gate_a]
    required.extend(path / "model.pt" for path in refit_dirs)
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"partial training run lacks sealed prerequisite: {missing}")
    command = [
        sys.executable, "scripts/expansion/analyze_scoped_gate_b.py",
        "--merged-dir", str(args.merged_dir),
        "--artifact-store", str(args.artifact_store),
        "--feature-cache", str(args.feature_cache),
        "--utility-config", str(args.utility_config),
        "--calibration-freeze", str(calibration),
        "--development-selection", str(development),
        "--baseline-selection", str(baseline),
        "--output", str(gate_b),
    ]
    for directory in refit_dirs:
        command.extend(["--refit-seed-dir", str(directory)])
    subprocess.run(command, cwd=ROOT, check=True)
    gate_payload = json.loads(gate_b.read_text())
    calibration_payload = json.loads(calibration.read_text())
    seed_manifest = json.loads((refit_dirs[0] / "manifest.json").read_text())
    manifest = {
        "schema_version": 1,
        "kind": "crashbench_expansion_scoped_d6_d7_training_run",
        "scope": "single_primary_policy_single_staleness_mechanism_pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operational_finalization": "Gate evaluator operator compatibility retry; no model/calibration rerun",
        "model_training_git_commit": seed_manifest["git_commit"],
        "finalizer_git_commit": resolve_git_head(ROOT),
        "seeds": list(range(5)),
        "merged_manifest_sha256": sha256_file(args.merged_dir / "dataset_manifest.json"),
        "gate_a_sha256": sha256_file(args.gate_a),
        "development_selection_sha256": sha256_file(development),
        "calibration_freeze_sha256": sha256_file(calibration),
        "gate_b_sha256": sha256_file(gate_b),
        "gate_b_status": gate_payload["gate"]["status"],
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "calibration_rows_read": calibration_payload["calibration_row_count"],
        "test_rows_read": 0,
        "status": "SEALED",
        "frozen": True,
    }
    run_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.run_root), "status": "SEALED", "gate_b": manifest["gate_b_status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
