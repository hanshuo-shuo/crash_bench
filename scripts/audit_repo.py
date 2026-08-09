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
SUPERSEDED_ROOT_DOCS = (
    "GIT_WORKFLOW.md",
    "OVERVIEW.md",
    "PAPER_PLAN.md",
    "PLAN.md",
    "REPORT.md",
    "ROADMAP.md",
    "STATUS.md",
    "STRATEGY.md",
    "motivation.md",
)
REQUIRED_ARCHIVE_DOCS = (
    "OVERVIEW.md",
    "P0_HANDOFF_20260726.md",
    "PHASE1.md",
    "REPORT.md",
    "REPO_AUDIT.md",
    "STRATEGY.md",
    "motivation.md",
)


def read_json(path: Path):
    return json.loads(path.read_text())


def dotted_get(value, path: str):
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def audit() -> list[str]:
    errors: list[str] = []
    # Keep one obvious documentation entrypoint. Historical narratives belong
    # in docs/archive rather than competing with README.md at repository root.
    for name in SUPERSEDED_ROOT_DOCS:
        if (ROOT / name).exists():
            errors.append(f"superseded root document reintroduced: {name}")
    for name in REQUIRED_ARCHIVE_DOCS:
        if not (ROOT / "docs/archive" / name).exists():
            errors.append(f"historical archive document missing: docs/archive/{name}")

    # Current docs must share the complete C0..C13 ledger vocabulary and no obsolete headline.
    claim_ids = set(re.findall(r"\bC(?:1[0-3]|[0-9])\b", (ROOT / "docs/CLAIMS.md").read_text()))
    expected_claim_ids = {f"C{i}" for i in range(14)}
    if claim_ids != expected_claim_ids:
        errors.append(f"docs/CLAIMS.md IDs {sorted(claim_ids)} != {sorted(expected_claim_ids)}")
    for name in ("CURRENT.md", "PAPER_PLAN.md"):
        if "C0–C13" not in (ROOT / "docs" / name).read_text():
            errors.append(f"docs/{name} does not declare the C0–C13 claim vocabulary")
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

    # The provenance-complete P0 is a frozen negative result.  Pin the fields
    # that prevent it from drifting into a positive external-validity claim.
    p0_entry = next((entry for entry in manifest.get("entries", [])
                     if entry.get("experiment_id") == "E12"), None)
    if p0_entry is None:
        errors.append("manifest lacks E12 provenance-complete P0")
    else:
        p0_path = ROOT / "results/p0_core_20260726_retry1/summary.json"
        if p0_path.exists():
            p0 = read_json(p0_path)
            expected = {
                "status": "complete_negative_result",
                "run_git_commit": "68d0195cc4bd93832f23cc949f8986814f75719b",
                "checkpoint_revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
                "design.capture_episode_count": 86,
                "design.guard_episode_count": 550,
                "probe_analysis.dissociation_supported": False,
                "online_guard.headline_metrics.vanilla.n": 50,
                "online_guard.headline_metrics.vanilla.crash_rate": 0.0,
            }
            for dotted, value in expected.items():
                if dotted_get(p0, dotted) != value:
                    errors.append(f"E12 P0 mismatch: {dotted}")
            summary_fingerprints = {
                row["fingerprint_sha256"] for row in p0["design"]["scenarios"]
            }
            if len(summary_fingerprints) != 11:
                errors.append("E12 P0 does not contain 11 unique scenario fingerprints")
            if summary_fingerprints != set(p0_entry["scenario_fingerprints"]):
                errors.append("E12 P0 scenario fingerprints differ from manifest")
            if p0_entry["git_commit"] != p0["run_git_commit"]:
                errors.append("E12 P0 run commit differs from manifest")

    # E13 is a completed prompt-scope follow-up. Pin the paired denominators and
    # task-success boundary that prevent lower crash rates becoming a recovery claim.
    e13_entry = next((entry for entry in manifest.get("entries", [])
                      if entry.get("experiment_id") == "E13"), None)
    if e13_entry is None:
        errors.append("manifest lacks E13 careful-prompt result provenance")
    else:
        source_payloads = {}
        for hazard in ("wall", "glass"):
            source_path = ROOT / f"results/careful_prompt/{hazard}_prompt_matrix.json"
            if source_path.exists():
                payload = read_json(source_path)
                source_payloads[hazard] = payload
                if len(payload["episodes"]) != 90:
                    errors.append(f"E13 {hazard} does not contain 90 episodes")
                if len(payload["scenarios"]) != 10:
                    errors.append(f"E13 {hazard} does not contain 10 scenarios")
                if payload["config"]["git_commit"] != e13_entry["git_commit"]:
                    errors.append(f"E13 {hazard} run commit differs from manifest")
        combined_path = ROOT / "results/careful_prompt/combined_summary.json"
        if combined_path.exists() and len(source_payloads) == 2:
            combined = read_json(combined_path)
            for hazard, payload in source_payloads.items():
                if combined["hazards"][hazard] != payload["summary"]:
                    errors.append(f"E13 combined {hazard} summary differs from source")
            expected = {
                "hazards.wall.by_condition_and_regime.vanilla.treatment.n_crash": 15,
                "hazards.wall.by_condition_and_regime.hazard_specific.treatment.n_crash": 13,
                "hazards.wall.by_condition_and_regime.hazard_specific.treatment.n_recovery_success": 0,
                "hazards.glass.by_condition_and_regime.vanilla.treatment.n_crash": 9,
                "hazards.glass.by_condition_and_regime.hazard_specific.treatment.n_crash": 2,
                "hazards.glass.by_condition_and_regime.hazard_specific.treatment.n_recovery_success": 0,
            }
            for dotted, value in expected.items():
                if dotted_get(combined, dotted) != value:
                    errors.append(f"E13 careful-prompt mismatch: {dotted}")
            source_fingerprints = {
                row["fingerprint_sha256"]
                for payload in source_payloads.values()
                for row in payload["scenarios"]
            }
            if source_fingerprints != set(e13_entry["scenario_fingerprints"]):
                errors.append("E13 scenario fingerprints differ from manifest")

    # E14 is a verified environment-validity smoke, not a learned recovery
    # result. Pin the admission gate and denominators that bound that wording.
    e14_entry = next((entry for entry in manifest.get("entries", [])
                      if entry.get("experiment_id") == "E14"), None)
    if e14_entry is None:
        errors.append("manifest lacks E14 acceptance-smoke provenance")
    else:
        e14_path = ROOT / "results/glass_recovery_acceptance_smoke_20260809.json"
        if e14_path.exists():
            e14 = read_json(e14_path)
            expected = {
                "status": "acceptance_smoke_complete",
                "provenance.git_commit": "7bb6d7de805280d084b2dc68084796aec2e0619e",
                "aggregate_attempt_accounting.candidate_rollout_attempts": 109,
                "aggregate_attempt_accounting.accepted_admissions": 3,
                "aggregate_attempt_accounting.rejected_attempts": 106,
                "aggregate_attempt_accounting.accepted_by_split.train": 2,
                "aggregate_attempt_accounting.accepted_by_split.validation": 0,
                "aggregate_attempt_accounting.accepted_by_split.heldout": 1,
                "validation.accepted_pairs": 3,
                "validation.trajectory_records": 12,
            }
            for dotted, value in expected.items():
                if dotted_get(e14, dotted) != value:
                    errors.append(f"E14 acceptance smoke mismatch: {dotted}")
            if e14_entry["git_commit"] != e14["provenance"]["git_commit"]:
                errors.append("E14 acceptance-smoke run commit differs from manifest")
            scene_hashes = {row["on_path_scene_sha256"] for row in e14["accepted"]}
            if scene_hashes != set(e14_entry["scenario_fingerprints"]):
                errors.append("E14 accepted scene hashes differ from manifest")
            for row in e14["accepted"]:
                if not row["base"]["crashed"] or not row["careful"]["crashed"]:
                    errors.append(f"E14 primary crash gate failed for {row['placement_id']}")
                if row["oracle"]["crashed"] or not row["oracle"]["task_succeeded"]:
                    errors.append(f"E14 oracle gate failed for {row['placement_id']}")

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
