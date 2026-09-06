#!/usr/bin/env python3
"""Execute the immutable D6/D7 scoped training chain on a sealed D5 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--gate-a", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, default=ROOT / "configs/expansion/utility_v1.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite scoped training root: {args.output_dir}")
    gate_a = json.loads(args.gate_a.read_text())
    if gate_a["gate"]["status"] not in {"GO", "SCOPED_CONTINUE"}:
        raise ValueError("D6 training is unauthorized because scoped Gate A did not continue")
    if gate_a.get("test_rows_read") != 0:
        raise ValueError("Gate A reports test access")
    args.output_dir.mkdir(parents=True)
    python = sys.executable
    baseline_dir = args.output_dir / "baselines"
    common = [
        "--merged-dir", str(args.merged_dir),
        "--artifact-store", str(args.artifact_store),
        "--utility-config", str(args.utility_config),
    ]
    if args.feature_cache is not None:
        common.extend(["--feature-cache", str(args.feature_cache)])
    run([python, "scripts/expansion/train_baselines.py", *common, "--output-dir", str(baseline_dir)])
    selection_dirs = []
    for seed in SEEDS:
        output = args.output_dir / "selection" / f"seed_{seed}"
        run([
            python, "scripts/expansion/train_option_value.py", *common,
            "--seed", str(seed), "--epochs", str(args.epochs), "--fit-on", "train",
            "--output-dir", str(output),
        ])
        selection_dirs.append(output)
    development_selection = args.output_dir / "development_model_selection.json"
    command = [
        python, "scripts/expansion/analyze_development_models.py",
        "--merged-dir", str(args.merged_dir),
        "--baseline-dir", str(baseline_dir),
        "--output", str(development_selection),
    ]
    for directory in selection_dirs:
        command.extend(["--seed-dir", str(directory)])
    run(command)
    calibration_dir = args.output_dir / "calibration"
    command = [
        python, "scripts/expansion/calibrate_selector.py", *common,
        "--development-selection", str(development_selection),
        "--baseline-selection", str(baseline_dir / "baseline_selection.json"),
        "--output-dir", str(calibration_dir),
    ]
    for directory in selection_dirs:
        command.extend(["--seed-dir", str(directory)])
    run(command)
    gate_b = args.output_dir / "scoped_gate_b_analysis.json"
    command = [
        python, "scripts/expansion/analyze_scoped_gate_b.py", *common,
        "--calibration-freeze", str(calibration_dir / "statewise_calibration_freeze.json"),
        "--development-selection", str(development_selection),
        "--baseline-selection", str(baseline_dir / "baseline_selection.json"),
        "--output", str(gate_b),
    ]
    for directory in selection_dirs:
        command.extend(["--seed-dir", str(directory)])
    run(command)
    manifest = {
        "schema_version": 2,
        "evaluation_role": "development_held_out_from_fit",
        "fit_on": "train",
        "kind": "crashbench_expansion_scoped_d6_d7_training_run",
        "scope": "single_primary_policy_single_staleness_mechanism_pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": args.epochs,
        "seeds": list(SEEDS),
        "merged_manifest_sha256": sha256_file(args.merged_dir / "dataset_manifest.json"),
        "gate_a_sha256": sha256_file(args.gate_a),
        "development_selection_sha256": sha256_file(development_selection),
        "calibration_freeze_sha256": sha256_file(calibration_dir / "statewise_calibration_freeze.json"),
        "gate_b_sha256": sha256_file(gate_b),
        "gate_b_status": json.loads(gate_b.read_text())["gate"]["status"],
        "feature_cache_sha256": None if args.feature_cache is None else sha256_file(args.feature_cache),
        "calibration_rows_read": json.loads(
            (calibration_dir / "statewise_calibration_freeze.json").read_text()
        )["calibration_row_count"],
        "test_rows_read": 0,
        "status": "SEALED",
        "frozen": True,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), "status": "SEALED", "test_rows_read": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
