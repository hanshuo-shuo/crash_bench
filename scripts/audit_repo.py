#!/usr/bin/env python3
"""Zero-GPU repository integrity audit for current CrashBench artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.scenario import scenario_fingerprint


REQUIRED_MANIFEST_FIELDS = {
    "experiment_id", "result_files", "git_commit", "created_date", "model", "checkpoint",
    "scenario_root", "scenario_fingerprints", "task_suite", "task_id", "repeat_k",
    "policy_options", "threshold", "crash_predicate", "force_threshold_N",
    "seed_or_nondeterminism", "generating_script", "sbatch", "raw_log", "analysis_script",
    "figures", "status", "notes",
}
CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/CURRENT.md",
    ROOT / "docs/PAPER_PLAN.md",
    ROOT / "docs/CLAIMS.md",
    ROOT / "docs/EXPERIMENT_INDEX.md",
    ROOT / "docs/REPRODUCIBILITY.md",
    ROOT / "docs/NEGATIVE_RESULTS.md",
)
BANNED_CURRENT_TEXT = ("98.3%", "98.3 %", "It still might just be ood still.")


def read_json(path: Path):
    return json.loads(path.read_text())


def dotted_get(value, path: str):
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def audit() -> list[str]:
    errors: list[str] = []
    # Current docs must share the complete C0..C12 ledger vocabulary and no obsolete headline.
    claim_ids = set(re.findall(r"\bC(?:1[0-2]|[0-9])\b", (ROOT / "docs/CLAIMS.md").read_text()))
    expected_claim_ids = {f"C{i}" for i in range(13)}
    if claim_ids != expected_claim_ids:
        errors.append(f"docs/CLAIMS.md IDs {sorted(claim_ids)} != {sorted(expected_claim_ids)}")
    for name in ("CURRENT.md", "PAPER_PLAN.md"):
        if "C0–C12" not in (ROOT / "docs" / name).read_text():
            errors.append(f"docs/{name} does not declare the C0–C12 claim vocabulary")
    for path in CURRENT_DOCS:
        text = path.read_text()
        for banned in BANNED_CURRENT_TEXT:
            if banned in text:
                errors.append(f"deprecated text {banned!r} in current doc {path.relative_to(ROOT)}")
    for name in ("CURRENT.md", "PAPER_PLAN.md", "CLAIMS.md"):
        if not (ROOT / "docs" / name).exists():
            errors.append(f"missing current document docs/{name}")

    # Exact on-disk scenario bytes must match the tracked fingerprint registry.
    fp_path = ROOT / "results/scenario_fingerprints.json"
    if not fp_path.exists():
        errors.append("missing results/scenario_fingerprints.json")
    else:
        seen_ids = set()
        for row in read_json(fp_path).get("scenarios", []):
            scenario_dir = ROOT / row["path"]
            if row["id"] in seen_ids:
                errors.append(f"duplicate scenario id in fingerprint manifest: {row['id']}")
            seen_ids.add(row["id"])
            if not scenario_dir.exists():
                errors.append(f"fingerprint scenario missing: {row['path']}")
            elif scenario_fingerprint(scenario_dir) != row["fingerprint_sha256"]:
                errors.append(f"scenario fingerprint drift: {row['path']}")
        tall = next((r for r in read_json(fp_path).get("scenarios", [])
                     if r["id"] == "env_collision__T5__libero_spatial_t0_wall_d62"), None)
        low = next((r for r in read_json(fp_path).get("scenarios", [])
                    if r["id"].endswith("__lowwall_detour_v1")), None)
        if not tall or not low or tall["fingerprint_sha256"] == low["fingerprint_sha256"]:
            errors.append("tall-wall and low-wall d62 scenarios are not distinct in fingerprint manifest")

    # Experiment manifest must point to existing files/scripts or deliberately named historical gaps.
    manifest = read_json(ROOT / "results/manifest.json")
    for entry in manifest.get("entries", []):
        missing = REQUIRED_MANIFEST_FIELDS - set(entry)
        if missing:
            errors.append(f"manifest {entry.get('experiment_id')} lacks {sorted(missing)}")
        for rel in entry.get("result_files", []):
            if not (ROOT / rel).exists():
                errors.append(f"manifest result missing: {rel}")
        script = entry.get("generating_script")
        if script and script.startswith("scripts/") and not (ROOT / script).exists():
            errors.append(f"manifest script missing: {script}")

    # Every claim result path must exist and declared numeric checks must equal raw JSON values.
    ledger = read_json(ROOT / "results/claims_ledger.json")
    for claim in ledger.get("claims", []):
        files = claim.get("result_files", [])
        for rel in files:
            if not (ROOT / rel).exists():
                errors.append(f"claim {claim['id']} result missing: {rel}")
        if not files:
            continue
        default_data = read_json(ROOT / files[0])
        for check in claim.get("checks", []):
            data = read_json(ROOT / check["file"]) if "file" in check else default_data
            if "array_length" in check:
                if len(data) != check["array_length"]:
                    errors.append(f"claim {claim['id']} array length mismatch for {files[0]}")
            elif dotted_get(data, check["path"]) != check["equals"]:
                errors.append(f"claim {claim['id']} mismatch: {check['path']}")
    return errors


def main() -> None:
    errors = audit()
    if errors:
        print("Repository audit FAILED:")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("Repository audit passed: current docs, scenario fingerprints, manifest paths, and claim checks are consistent.")


if __name__ == "__main__":
    main()
