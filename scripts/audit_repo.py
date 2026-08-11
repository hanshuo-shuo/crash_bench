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
E15_DOCS = (
    "GLASS_RECOVERY_V1.md",
    "CURRENT.md",
    "PAPER_PLAN.md",
    "CLAIMS.md",
    "EXPERIMENT_INDEX.md",
    "SCRIPT_INDEX.md",
    "REPRODUCIBILITY.md",
)
E15_MAIN_BASELINE_CONDITIONS = {
    "base",
    "generic_careful",
    "hazard_specific_careful",
    "risk_gate_retreat_hold",
    "full_learned_gate_recovery",
    "risk_gate_oracle_recovery",
    "oracle_timed_learned_recovery",
    "oracle_timed_oracle_recovery",
}
LEARNED_PROMOTION_STATUSES = {
    "frozen_learned_result",
    "verified_learned_result",
    "claim_ready",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def dotted_get(value, path: str):
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def learned_recovery_semantic_errors(payload: dict) -> list[str]:
    """Validate the minimum machine-readable contract for an E15 claim.

    Code-complete, smoke, and fake-environment artifacts deliberately do not
    satisfy this schema.  Keeping the check here makes a future manifest status
    promotion fail closed instead of relying on prose review alone.
    """

    errors: list[str] = []
    if payload.get("kind") != "glass_recovery_learned_result":
        errors.append("kind must be glass_recovery_learned_result")
    cohort = payload.get("accepted_cohort")
    if not isinstance(cohort, dict):
        return [*errors, "accepted_cohort must be an object"]
    cohort_sha = str(cohort.get("sha256", ""))
    if len(cohort_sha) != 64 or any(c not in "0123456789abcdef" for c in cohort_sha):
        errors.append("accepted_cohort.sha256 must be a lowercase SHA-256")
    pair_ids = cohort.get("pair_ids")
    if not isinstance(pair_ids, list) or not pair_ids or len(pair_ids) != len(set(pair_ids)):
        errors.append("accepted_cohort.pair_ids must be nonempty and unique")

    replay = payload.get("exact_replay_summary")
    if not isinstance(replay, dict) or replay.get("all_passed") is not True:
        errors.append("exact_replay_summary.all_passed must be true")
    elif not isinstance(replay.get("n_pairs"), int) or replay["n_pairs"] != len(pair_ids or []):
        errors.append("exact_replay_summary.n_pairs must equal accepted pair count")

    counts = payload.get("source_state_counts")
    if not isinstance(counts, dict):
        errors.append("source_state_counts must be an object")
    else:
        for split in ("train", "validation", "final_heldout"):
            if not isinstance(counts.get(split), int) or counts[split] < 1:
                errors.append(f"source_state_counts.{split} must be positive")
    family_counts = payload.get("family_counts")
    if not isinstance(family_counts, dict) or not all(
        isinstance(family_counts.get(split), dict) and family_counts[split]
        for split in ("train", "validation", "final_heldout")
    ):
        errors.append("family_counts must cover train/validation/final_heldout")
    leakage = payload.get("leakage_checks")
    for key in (
        "source_state_overlap_count",
        "family_overlap_count",
        "physical_scene_overlap_count",
    ):
        if not isinstance(leakage, dict) or leakage.get(key) != 0:
            errors.append(f"leakage_checks.{key} must equal zero")

    identities = payload.get("identities")
    identity_fields = {
        "checkpoint": ("sha256", "training_seed"),
        "base": ("resolved_revision", "unnorm_key"),
        "dataset": (
            "train_manifest_sha256",
            "validation_manifest_sha256",
            "final_heldout_manifest_sha256",
        ),
        "protocol": ("primary_sha256", "evaluation_sha256"),
    }
    for key, fields in identity_fields.items():
        value = identities.get(key) if isinstance(identities, dict) else None
        missing = [field for field in fields if not isinstance(value, dict) or value.get(field) in (None, "")]
        if missing:
            errors.append(f"identities.{key} lacks {missing}")

    evaluation = payload.get("evaluation")
    conditions = set(evaluation.get("conditions", [])) if isinstance(evaluation, dict) else set()
    missing_conditions = sorted(E15_MAIN_BASELINE_CONDITIONS - conditions)
    if missing_conditions:
        errors.append(f"evaluation.conditions lacks {missing_conditions}")
    if not isinstance(evaluation, dict) or evaluation.get("primary_mode") != "source_to_task":
        errors.append("evaluation.primary_mode must be source_to_task")

    analysis = payload.get("source_cluster_analysis")
    if not isinstance(analysis, dict) or analysis.get("independent_cluster") != "source_state_sha256":
        errors.append("source_cluster_analysis must use source_state_sha256")
    metrics = payload.get("primary_metrics")
    for key in (
        "safe_task_success",
        "catastrophe",
        "clean_control_false_intervention",
        "clean_control_task_preservation",
    ):
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if not isinstance(value, dict) or not isinstance(value.get("denominator"), int) \
                or value["denominator"] < 1 or "estimate" not in value:
            errors.append(f"primary_metrics.{key} needs estimate and positive denominator")
    return errors


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

    # E15 is a new protocol, while E14 remains immutable history.  Every user-
    # facing experiment index must make that boundary explicit.
    for name in E15_DOCS:
        text = (ROOT / "docs" / name).read_text()
        if "E15" not in text or "E14" not in text:
            errors.append(f"docs/{name} does not distinguish E15 from E14")

    smoke = (ROOT / "setup/glass_recovery_smoke.sbatch").read_text()
    wrapper = (ROOT / "setup/submit_glass_recovery_smoke.sh").read_text()
    for token in (
        "CB_GLASS_RECOVERY_TRAIN_MANIFEST",
        "CB_GLASS_RECOVERY_VALIDATION_MANIFEST",
        "CB_GLASS_RECOVERY_PRIMARY_PROTOCOL_SHA256",
        "--placement-manifest",
        "--trajectory-manifest",
        "--evaluation-cohort",
        "--protocol",
        "CB_GLASS_RECOVERY_EVALUATION_PROTOCOL_SHA256",
    ):
        if token not in smoke:
            errors.append(f"E15 smoke lacks required accepted-input token {token}")
    for token in ("train|evaluate", "CB_GLASS_RECOVERY_EVALUATION_COHORT"):
        if token not in wrapper:
            errors.append(f"E15 submit wrapper lacks required token {token}")
    for stale in (
        "results/glass_recovery_v1",
        "prepare_glass_recovery_placements.py",
        "collect_glass_recovery_pairs.py",
        "CB_GLASS_RECOVERY_SOURCE_RUN_ROOT",
        "--max-placements",
    ):
        if stale in smoke:
            errors.append(f"E15 smoke still contains v1/authored-cohort path {stale}")

    # P0-E must remain an executable, fail-closed path rather than a prose-only
    # prerequisite.  These tokens pin the safeguards that previously caused the
    # code-complete no-go.
    p0e_contracts = {
        "scripts/capture_glass_nominal_source_traces.py": (
            "glass_recovery_nominal_source_traces",
            "robot_body_xyz",
            "checkpoint_revision",
        ),
        "scripts/prepare_glass_recovery_placements.py": (
            "--source-trace-manifest",
            "_fixed_action_hazard_screen",
            "physical_scene_sha256",
            "candidate_order_index",
            "--legacy-straight-path",
        ),
        "scripts/audit_glass_core_artifacts.py": (
            "--read-only",
            "core_salvage_audit",
            "direct_v2_promotion_allowed",
            "source_root_read_only",
        ),
        "scripts/run_glass_avoidability_frontier.py": (
            "FRONTIER_HORIZONS = (40, 30, 20, 15, 10, 5)",
            "--print-commands",
            "--execute",
            "recommended_horizon_actions",
        ),
    }
    for relative, tokens in p0e_contracts.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"P0-E executable missing: {relative}")
            continue
        text = path.read_text()
        for token in tokens:
            if token not in text:
                errors.append(f"P0-E {relative} lacks semantic guard {token}")
    collector_text = (ROOT / "scripts/collect_glass_recovery_pairs.py").read_text()
    if "_ordered_placements(placements)" not in collector_text:
        errors.append("collector does not consume the predeclared P0-E candidate order")
    if "-p.nominal_fraction" in collector_text:
        errors.append("collector reintroduced high-fraction-first candidate selection")

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

    # E15 Pilot A is a completed salvage/replay gate, not a learned result. Pin
    # its denominators, exact-H result, and explicit non-promotion boundary.
    e15_entry = next((entry for entry in manifest.get("entries", [])
                      if entry.get("experiment_id") == "E15"), None)
    if e15_entry is None:
        errors.append("manifest lacks E15 Pilot A provenance")
    else:
        pilot_a_path = ROOT / "results/glass_recovery_pilot_a_20260811.json"
        if pilot_a_path.exists():
            pilot_a = read_json(pilot_a_path)
            expected = {
                "kind": "glass_recovery_pilot_a_result",
                "status": "pilot_a_complete_go",
                "decision.value": "pilot_a_go",
                "decision.go": True,
                "decision.pilot_b_allowed": True,
                "decision.pilot_b_executed": False,
                "provenance.runner_git_commit": "42c30607b23afaeff264dbe343dcb82add3d6656",
                "provenance.slurm_job.job_id": 9044175,
                "provenance.slurm_job.state": "COMPLETED",
                "historical_attempt_accounting.attempts": 109,
                "historical_attempt_accounting.unique_attempt_ids": 109,
                "historical_attempt_accounting.attempts_have_unique_provenance": True,
                "historical_attempt_accounting.unrecoverable_attempt_identity_lower_bound": 0,
                "gate.target_h_actions": 20,
                "gate.candidate_count": 3,
                "gate.rates.exact_controller_state_available": 1.0,
                "gate.rates.repaired_first_action_predicate_checks": 1.0,
                "gate.rates.exact_h_nominal_suffix_replay": 1.0,
                "gate.rates.oracle_independent_replay": 1.0,
                "gate.direct_v2_promotions": 0,
                "gate.candidate_role": "development_salvage_only",
            }
            for dotted, value in expected.items():
                if dotted_get(pilot_a, dotted) != value:
                    errors.append(f"E15 Pilot A mismatch: {dotted}")
            expected_ids = {
                "glass_recovery_heldout_0023",
                "glass_recovery_train_0051",
                "glass_recovery_train_0074",
            }
            candidates = pilot_a.get("candidates", [])
            if {row.get("placement_id") for row in candidates} != expected_ids:
                errors.append("E15 Pilot A candidate IDs differ from the final summary")
            if any(
                row.get("catastrophe_action_index_zero_based") != 19
                or row.get("remaining_actions") != 20
                or row.get("oracle_independent_recapture_success") is not True
                or row.get("oracle_relevant_observation_exact_across_resets") is not True
                for row in candidates
            ):
                errors.append("E15 Pilot A candidate replay/oracle gate drifted")
            if e15_entry["git_commit"] != pilot_a["provenance"]["runner_git_commit"]:
                errors.append("E15 Pilot A run commit differs from manifest")
            if e15_entry["status"] != pilot_a["status"]:
                errors.append("E15 Pilot A status differs from manifest")
            if e14_entry is not None and set(e15_entry["scenario_fingerprints"]) != set(
                e14_entry["scenario_fingerprints"]
            ):
                errors.append("E15 Pilot A source scenes differ from E14 manifest")

    # A Pilot A entry must not be mistaken for a learned result. If E15 is ever
    # promoted, require a separate machine-readable learned-result artifact with
    # every semantic guard declared by P0-F.
    if e15_entry is not None and e15_entry.get("status") in LEARNED_PROMOTION_STATUSES:
        learned_payload = None
        for rel in e15_entry.get("result_files", []):
            path = ROOT / rel
            if path.suffix == ".json" and path.is_file():
                candidate = read_json(path)
                if candidate.get("kind") == "glass_recovery_learned_result":
                    learned_payload = candidate
                    break
        if learned_payload is None:
            errors.append("promoted E15 lacks a glass_recovery_learned_result artifact")
        else:
            errors.extend(
                f"E15 learned-result contract: {error}"
                for error in learned_recovery_semantic_errors(learned_payload)
            )

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
    print("Repository audit passed: current docs, E14/E15 semantics, smoke contracts, scenario fingerprints, manifest paths, and claim checks are consistent.")


if __name__ == "__main__":
    main()
