#!/usr/bin/env python3
"""Zero-GPU repository integrity audit for current CrashBench artifacts."""

from __future__ import annotations

import hashlib
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
    ROOT / "docs/COUNTERFACTUAL_ROUTER_MAIN_RESULT.md",
    ROOT / "docs/EXPERIMENT_INDEX.md",
    ROOT / "docs/REPRODUCIBILITY.md",
    ROOT / "docs/SCRIPT_INDEX.md",
)
APPENDIX_DOCS = (
    ROOT / "docs/appendix/README.md",
    ROOT / "docs/appendix/GLASS_SAFETY_UTILITY.md",
    ROOT / "docs/appendix/CAREFUL_PROMPT_EXPERIMENT.md",
    ROOT / "docs/appendix/P0_EXPERIMENT.md",
    ROOT / "docs/appendix/NEGATIVE_RESULTS.md",
)
ICLR27_DOCS = (
    ROOT / "docs/iclr27/MASTER_PLAN.md",
    ROOT / "docs/iclr27/BASELINE_AUDIT_RESULT.md",
    ROOT / "docs/iclr27/CLAIM_LEDGER.md",
    ROOT / "docs/iclr27/RELATED_WORK_MATRIX.md",
    ROOT / "docs/iclr27/REVIEWER_RISK_AUDIT.md",
)
ICLR27_CLAIM_FIELDS = {
    "claim_id",
    "paper_wording",
    "status",
    "independent_unit",
    "n_sources",
    "cohort",
    "method_hash",
    "protocol_hash",
    "result_files",
    "statistical_test",
    "limitations",
    "forbidden_stronger_wording",
}
ICLR27_CLAIM_STATUSES = {"supported", "conditional", "unsupported"}
BANNED_CURRENT_TEXT = (
    "98.3%",
    "98.3 %",
    "It still might just be ood still.",
    "The current paper question is E15",
    "the new main exploit target is task-completing learned glass recovery",
    "current v2 learned-recovery line",
    "frontier and later pilots pending",
    "For sequential Pilots C--F",
)
REQUIRED_CURRENT_TEXT = {
    ROOT / "README.md": (
        "Knowing When to Intervene",
        "option-conditioned outcome prediction",
        "Always Retreat",
    ),
    ROOT / "docs/CURRENT.md": (
        "C0–C14",
        "Four predeclared points meet all four frontier criteria",
        "Do not tune a deeper sequence model",
    ),
    ROOT / "docs/PAPER_PLAN.md": (
        "Risk detection is not intervention selection",
        "no single all-criteria point",
        "Do not tune this cohort further",
    ),
    ROOT / "docs/COUNTERFACTUAL_ROUTER_MAIN_RESULT.md": (
        "Seven methods in the matched comparison",
        "Hazard Prompt",
        "Oracle value recovered",
    ),
}
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
REQUIRED_ARCHIVE_PATHS = (
    "docs/archive/OVERVIEW.md",
    "docs/archive/P0_HANDOFF_20260726.md",
    "docs/archive/PHASE1.md",
    "docs/archive/REPORT.md",
    "docs/archive/REPO_AUDIT.md",
    "docs/archive/STRATEGY.md",
    "docs/archive/motivation.md",
    "docs/archive/SETUP_README_20260812.md",
    "docs/archive/glass_recovery_20260812/README.md",
    "docs/archive/glass_recovery_20260812/GLASS_PAPER_EXECUTION.md",
    "docs/archive/glass_recovery_20260812/GLASS_RECOVERY_V1.md",
    "docs/archive/glass_recovery_20260812/GLASS_RECOVERY_PROGRESS_REPORT_20260812.md",
    "docs/archive/glass_recovery_20260812/E15_EXPERIMENT_LOG_20260811.md",
    "docs/archive/glass_recovery_20260812/PILOT_B_REPAIR_20260811.md",
    "docs/archive/glass_recovery_20260812/PILOT_B_FRESH_H20_20260812.md",
    "docs/archive/glass_recovery_20260812/PILOT_B_CONTROLLER_COMPATIBLE_H20_20260812.md",
    "docs/archive/preliminary_reports/OpenVLA_Recovery_Finetune_Preliminary_Report_20260730.docx",
    "docs/archive/preliminary_reports/OpenVLA_Safety_Baseline_Preliminary_Report_20260730.docx",
    "docs/archive/preliminary_reports/OpenVLA_Safety_Baseline_Prompt_Audit_20260731.docx",
    "legacy/README.md",
    "legacy/setup/commit_p0_core.sh",
    "legacy/setup/submit_next_round.sh",
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


def markdown_local_link_errors(paths: tuple[Path, ...]) -> list[str]:
    """Return missing relative Markdown targets for current, maintained docs."""

    errors: list[str] = []
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for source in paths:
        if not source.is_file():
            errors.append(f"maintained document missing: {source.relative_to(ROOT)}")
            continue
        for raw_target in link_pattern.findall(source.read_text()):
            target = raw_target.strip()
            if target.startswith("<") and ">" in target:
                target = target[1:target.index(">")]
            else:
                # None of the maintained links use spaces in paths. Removing an
                # optional Markdown title keeps the checker intentionally small.
                target = target.split(maxsplit=1)[0]
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            destination = (source.parent / target).resolve()
            if not destination.exists():
                errors.append(
                    f"broken local link in {source.relative_to(ROOT)}: {raw_target}"
                )
    return errors


def iclr27_truth_source_errors() -> list[str]:
    """Validate the ICLR truth source and the frozen bytes behind its claims."""

    errors: list[str] = []
    for path in ICLR27_DOCS:
        if not path.is_file():
            errors.append(f"ICLR27 truth-source document missing: {path.relative_to(ROOT)}")
    manifest_path = ROOT / "results/iclr27/manifest.json"
    if not manifest_path.is_file():
        return [*errors, "ICLR27 truth-source manifest missing: results/iclr27/manifest.json"]
    if errors:
        return errors

    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        errors.append("ICLR27 manifest schema_version must be 1")
    if manifest.get("kind") != "iclr27_paper_truth_source_manifest":
        errors.append("ICLR27 manifest kind mismatch")
    decision = manifest.get("stage_1_decision", {})
    if decision.get("verdict") != "conditional_go" or decision.get("method_paper_ready") is not False:
        errors.append("ICLR27 stage-1 decision must remain conditional_go and not method-paper-ready")
    if decision.get("stage_1_complete") is not True:
        errors.append("ICLR27 stage-1 decision must record completed truth-source verification")
    attempt_fields = {
        "attempt", "git_commit", "slurm_job_id", "slurm_account", "slurm_partition",
        "status", "exit_code", "failure", "slurm_log", "result_dir",
    }
    seen_attempts: set[int] = set()
    for attempt in manifest.get("verification_attempts", []):
        missing = attempt_fields - set(attempt)
        if missing:
            errors.append(f"ICLR27 verification attempt lacks {sorted(missing)}")
        attempt_number = attempt.get("attempt")
        if attempt_number in seen_attempts:
            errors.append(f"ICLR27 verification attempt duplicated: {attempt_number}")
        seen_attempts.add(attempt_number)
        if attempt.get("slurm_account") != "p33100" or attempt.get("slurm_partition") != "short":
            errors.append(f"ICLR27 verification attempt {attempt_number} has invalid Slurm provenance")
    passed_attempts = [
        attempt for attempt in manifest.get("verification_attempts", [])
        if attempt.get("status") == "passed" and attempt.get("exit_code") == "0:0"
    ]
    if len(passed_attempts) != 1:
        errors.append("ICLR27 stage 1 must have exactly one recorded successful Quest verification")

    master_text = (ROOT / "docs/iclr27/MASTER_PLAN.md").read_text()
    for token in (
        "CONDITIONAL GO",
        "One-page viability judgment",
        "strongest-baseline gate",
        "source state",
    ):
        if token not in master_text:
            errors.append(f"ICLR27 master plan lacks required token {token!r}")

    risk_text = (ROOT / "docs/iclr27/REVIEWER_RISK_AUDIT.md").read_text()
    hardcoded_risk_ids = {f"R{i:02d}" for i in range(1, 16)}
    declared_risk_ids = set(manifest.get("required_reviewer_risks", []))
    if declared_risk_ids != hardcoded_risk_ids:
        errors.append("ICLR27 manifest must declare reviewer risks R01--R15")
    missing_risks = sorted(risk_id for risk_id in hardcoded_risk_ids if risk_id not in risk_text)
    if missing_risks:
        errors.append(f"ICLR27 reviewer audit lacks {missing_risks}")
    for token in (
        "Risk -> Detour",
        "Direct Choice",
        "45 frontier points",
        "privileged geometry",
        "source-block exact",
        "CoWAM",
        "CheckVLA",
        "SAFE",
    ):
        if token not in risk_text:
            errors.append(f"ICLR27 reviewer audit lacks required risk text {token!r}")

    related_text = (ROOT / "docs/iclr27/RELATED_WORK_MATRIX.md").read_text()
    for url in (
        "https://arxiv.org/abs/2506.09937",
        "https://arxiv.org/abs/2607.26789",
        "https://arxiv.org/abs/2608.02578",
    ):
        if url not in related_text:
            errors.append(f"ICLR27 related-work matrix lacks primary source {url}")

    claim_index = manifest.get("claim_index", {})
    indexed_claims: dict[str, str] = {}
    for status in ICLR27_CLAIM_STATUSES:
        for claim_id in claim_index.get(status, []):
            if claim_id in indexed_claims:
                errors.append(f"ICLR27 claim indexed more than once: {claim_id}")
            indexed_claims[claim_id] = status

    ledger_text = (ROOT / "docs/iclr27/CLAIM_LEDGER.md").read_text()
    ledger_claims: dict[str, str] = {}
    for block in re.split(r"(?m)^### ", ledger_text)[1:]:
        claim_match = re.search(r"(?m)^claim_id:\s*([^\s]+)\s*$", block)
        if not claim_match:
            continue
        claim_id = claim_match.group(1)
        if claim_id in ledger_claims:
            errors.append(f"ICLR27 ledger duplicates claim {claim_id}")
        present_fields = set(re.findall(r"(?m)^([a-z_]+):(?:\s|$)", block))
        missing_fields = ICLR27_CLAIM_FIELDS - present_fields
        if missing_fields:
            errors.append(f"ICLR27 claim {claim_id} lacks fields {sorted(missing_fields)}")
        status_match = re.search(r"(?m)^status:\s*([^\s]+)\s*$", block)
        status = status_match.group(1) if status_match else ""
        if status not in ICLR27_CLAIM_STATUSES:
            errors.append(f"ICLR27 claim {claim_id} has invalid status {status!r}")
        ledger_claims[claim_id] = status
    if ledger_claims != indexed_claims:
        errors.append(
            "ICLR27 ledger/index mismatch: "
            f"ledger={sorted(ledger_claims.items())}, index={sorted(indexed_claims.items())}"
        )

    artifact_paths: set[str] = set()
    for artifact in manifest.get("artifacts", []):
        relative = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not isinstance(relative, str) or not relative:
            errors.append("ICLR27 manifest artifact lacks path")
            continue
        if relative in artifact_paths:
            errors.append(f"ICLR27 manifest duplicates artifact {relative}")
        artifact_paths.add(relative)
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"ICLR27 pinned artifact missing: {relative}")
            continue
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            errors.append(f"ICLR27 frozen artifact hash drift: {relative}")
        for claim_id in artifact.get("claims", []):
            if claim_id not in indexed_claims:
                errors.append(f"ICLR27 artifact {relative} names unknown claim {claim_id}")

    for check in manifest.get("value_checks", []):
        relative = check.get("path")
        if relative not in artifact_paths:
            errors.append(f"ICLR27 value check uses unpinned artifact: {relative}")
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        data = read_json(path)
        if "array_length" in check:
            actual = len(data)
            expected = check["array_length"]
        else:
            try:
                actual = dotted_get(data, check["json_path"])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                errors.append(
                    f"ICLR27 value check path missing for {check.get('claim_id')}: "
                    f"{relative}:{check.get('json_path')} ({exc})"
                )
                continue
            expected = check.get("equals")
        if actual != expected:
            errors.append(
                f"ICLR27 value drift for {check.get('claim_id')}: "
                f"{relative}:{check.get('json_path', 'array_length')} "
                f"expected {expected!r}, got {actual!r}"
            )
    return errors


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


def expansion_governance_errors() -> list[str]:
    """Validate the D0 exposure firewall without reading any test outcome."""

    errors: list[str] = []
    required = (
        "configs/expansion/exposed_sources_v1.yaml",
        "configs/expansion/benchmark_v1.yaml",
        "configs/expansion/utility_v1.yaml",
        "configs/expansion/mechanism_screens_v1.yaml",
        "configs/expansion/compute_budget_v1.yaml",
        "schemas/source_registry.schema.json",
        "schemas/exact_state.schema.json",
        "schemas/split_manifest.schema.json",
        "schemas/option_catalog.schema.json",
        "schemas/utility.schema.json",
        "schemas/branch_row.schema.json",
        "schemas/gate_decision.schema.json",
        "configs/expansion/deployable_options_v1.yaml",
        "configs/expansion/diagnostic_oracle_options_v1.yaml",
        "configs/expansion/source_sampling_v1.yaml",
        "configs/expansion/splits_v1.yaml",
        "schemas/source_attempt.schema.json",
        "schemas/mechanism.schema.json",
        "schemas/d1_conformance_decision.schema.json",
        "schemas/shard_audit.schema.json",
        "results/expansion/governance/exposure_registry.json",
        "results/expansion/governance/power_planning.json",
        "results/expansion/governance/remote_missing_lineage.json",
        "results/expansion/governance/d1_backend_capability_report.json",
        "results/expansion/governance/protocol_v1_1_freeze.json",
        "results/expansion/governance/split_manifest_v1_1.json",
        "results/expansion/d3_formal_nominal/05ddc6a3a7c3_acb3d2d28dc2_20260830T103120Z/formal_nominal_analysis.json",
        "results/expansion/governance/exposure_attempts.jsonl",
        "results/expansion/d2_sources/screen_plan_256ce74521254ec9f692b77cb5197b5aa29be73f58d91ce29a6c945841cba7de.json",
        "results/expansion/d3_sources/formal_nominal_plan_a42e34a64bf2e0f8200e3171f80bd646e59ffc442edbcd09fa62fe376029e667.json",
        "results/expansion/d2_preflight/77ea231ebf97_a1464cf4ddf0_20260830T084407Z/nominal_preflight_analysis.json",
        "results/expansion/d2_mechanical/d79973fd46e3_f2a8d011f62d_20260830T090458Z/mechanical_preflight_analysis.json",
        "results/expansion/d2_fragile_screen/0f270c3d9c8c_64cc00fdccbb_20260830T091340Z/fragile_screen_analysis.json",
        "results/expansion/d2_staleness_screen/1f39d658a829_0100901d1cc6_20260830T093356Z/staleness_screen_analysis.json",
        "results/expansion/d2_action_drift_screen/117904fb3d5c_cb828fc5e0e0_20260830T095028Z/action_drift_screen_analysis.json",
        "results/expansion/d2_narrow_screen/a1d7e8ab026a_384b032a68fc_20260830T100829Z/narrow_screen_analysis.json",
        "results/expansion/governance/d2_selection_decision.json",
        "scripts/expansion/verify_exact_branching.py",
        "setup/expansion_verify.sbatch",
        "setup/submit_expansion_verify.sh",
        "scripts/expansion/run_nominal_preflight.py",
        "scripts/expansion/analyze_nominal_preflight.py",
        "scripts/expansion/derive_mechanical_preflight.py",
        "scripts/expansion/analyze_mechanical_preflight.py",
        "setup/expansion_author_sources.sbatch",
        "setup/submit_expansion_preflight.sh",
        "setup/expansion_mechanical_preflight.sbatch",
        "setup/submit_expansion_mechanical_preflight.sh",
        "scripts/expansion/run_fragile_screen.py",
        "scripts/expansion/analyze_fragile_screen.py",
        "scripts/expansion/run_staleness_screen.py",
        "scripts/expansion/analyze_staleness_screen.py",
        "setup/expansion_staleness_screen.sbatch",
        "setup/submit_expansion_staleness_screen.sh",
        "scripts/expansion/run_action_drift_screen.py",
        "scripts/expansion/analyze_action_drift_screen.py",
        "setup/expansion_action_drift_screen.sbatch",
        "setup/submit_expansion_action_drift_screen.sh",
        "scripts/expansion/run_narrow_clearance_screen.py",
        "scripts/expansion/analyze_narrow_clearance_screen.py",
        "setup/expansion_narrow_screen.sbatch",
        "setup/submit_expansion_narrow_screen.sh",
        "scripts/expansion/analyze_d2_selection.py",
        "scripts/expansion/run_formal_nominal.py",
        "scripts/expansion/analyze_formal_nominal.py",
        "setup/expansion_formal_nominal.sbatch",
        "setup/submit_expansion_formal_nominal.sh",
        "scripts/expansion/freeze_splits.py",
        "scripts/expansion/freeze_protocol.py",
        "schemas/protocol_freeze.schema.json",
        "scripts/expansion/collect_staleness_statewise.py",
        "setup/expansion_collect_statewise.sbatch",
        "setup/submit_expansion_statewise_collection.sh",
        "scripts/expansion/merge_statewise_dataset.py",
        "scripts/expansion/analyze_statewise_gate_a.py",
        "scripts/expansion/train_option_value.py",
        "scripts/expansion/train_baselines.py",
        "scripts/expansion/analyze_development_models.py",
        "scripts/expansion/calibrate_selector.py",
        "scripts/expansion/run_scoped_training.py",
        "scripts/expansion/analyze_scoped_gate_b.py",
        "setup/expansion_fragile_screen.sbatch",
        "setup/submit_expansion_fragile_screen.sh",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"D0 governance artifact missing: {relative}")
    if errors:
        return errors

    registry = read_json(ROOT / "results/expansion/governance/exposure_registry.json")
    expected_digest = registry.get("registry_sha256")
    digest_payload = dict(registry)
    digest_payload.pop("registry_sha256", None)
    actual_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_digest != actual_digest:
        errors.append("D0 exposure registry self hash mismatch")
    if registry.get("gate", {}).get("status") != "GO":
        errors.append("D0 exposure registry gate is not GO")
    if registry.get("gate", {}).get("unresolved_unblacklisted_count") != 0:
        errors.append("D0 exposure registry has unresolved unblacklisted lineage")
    if registry.get("gate", {}).get("missing_required_root_count") != 0:
        errors.append("D0 exposure registry has missing required roots")
    if registry.get("counts", {}).get("source_union_count", 0) < 74:
        errors.append("D0 exposure registry fell below the audited 74-source lower bound")
    if any(row.get("test_eligible") is not False for row in registry.get("sources", [])):
        errors.append("D0 exposure registry contains a test-eligible historical source")

    benchmark = read_json(ROOT / "configs/expansion/benchmark_v1.yaml")
    test_policy = benchmark.get("test_policy", {})
    if test_policy.get("test_authorized") is not False:
        errors.append("D0 benchmark config prematurely authorizes test")
    if test_policy.get("test_outcomes_may_be_read") is not False:
        errors.append("D0 benchmark config permits test-outcome access")
    if benchmark.get("mcv", {}).get("statewise_test_total") != 96:
        errors.append("D0 benchmark statewise test size drifted from 96")
    if benchmark.get("mcv", {}).get("fresh_sequential_total") != 60:
        errors.append("D0 benchmark sequential size drifted from 60")

    remote = read_json(ROOT / "results/expansion/governance/remote_missing_lineage.json")
    if remote.get("unblacklisted_missing_pool_count") != 0:
        errors.append("D0 remote lineage ledger has an unblacklisted pool")
    d1 = read_json(ROOT / "results/expansion/governance/d1_backend_capability_report.json")
    if d1.get("d1_gate", {}).get("status") not in {"NOT_EVALUATED", "PARTIAL_PASS", "GO", "NO_GO"}:
        errors.append("D1 live gate has an unregistered status")
    if d1.get("claim_boundary") != (
        "Synthetic tests establish software contracts only and are not benchmark evidence."
    ):
        errors.append("D1 capability report does not preserve the synthetic-evidence boundary")
    for backend in d1.get("backends", []):
        for cell in backend.get("live_cells", []):
            result_path = ROOT / cell["result"]
            if not result_path.is_file():
                errors.append(f"D1 live conformance result missing: {cell['result']}")
                continue
            result = read_json(result_path)
            claimed_pass = cell.get("classification") == "DETERMINISTIC_EXACT"
            result_pass = result.get("gate", {}).get("status") == "PASS"
            if claimed_pass != result_pass:
                errors.append(
                    f"D1 live cell classification disagrees with result gate: {cell['result']}"
                )
            if result.get("source", {}).get("role") != "EXPOSED_ENGINEERING_ONLY":
                errors.append(f"D1 live source is not engineering-only: {cell['result']}")
    pi0 = benchmark.get("primary_policy", {})
    if pi0.get("backend") != "pi0":
        errors.append("D1 resolved benchmark primary policy must be pi0")
    if benchmark.get("d1_primary_resolution", {}).get("multipolicy_claim_active") is not False:
        errors.append("D1 resolved benchmark must keep multipolicy claim inactive")
    identity_path = pi0.get("identity_manifest")
    if not identity_path or not (ROOT / identity_path).is_file():
        errors.append("D0 pi0 identity manifest is missing")
    else:
        identity = read_json(ROOT / identity_path)
        canonical_files = json.dumps(
            identity.get("files", []), sort_keys=True, separators=(",", ":")
        ).encode()
        tree_hash = hashlib.sha256(canonical_files).hexdigest()
        if tree_hash != identity.get("tree_manifest_sha256") or tree_hash != pi0.get(
            "checkpoint_tree_manifest_sha256"
        ):
            errors.append("D0 pi0 checkpoint tree manifest hash mismatch")
        if identity.get("file_count") != 19 or identity.get("total_size_bytes") != 12014131888:
            errors.append("D0 pi0 checkpoint tree inventory drift")
        if identity.get("slurm", {}).get("job_id") != "5203988":
            errors.append("D0 pi0 checkpoint identity lacks Slurm provenance")
    screen_plan_path = ROOT / (
        "results/expansion/d2_sources/"
        "screen_plan_256ce74521254ec9f692b77cb5197b5aa29be73f58d91ce29a6c945841cba7de.json"
    )
    screen_plan = read_json(screen_plan_path)
    plan_without_hash = dict(screen_plan)
    declared_plan_hash = plan_without_hash.pop("plan_sha256", None)
    actual_plan_hash = hashlib.sha256(
        json.dumps(plan_without_hash, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    attempts = screen_plan.get("attempts", [])
    ledger_rows = [
        json.loads(line)
        for line in (ROOT / "results/expansion/governance/exposure_attempts.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    if declared_plan_hash != actual_plan_hash:
        errors.append("D2 screen source plan self hash mismatch")
    screen_ledger_rows = [
        row for row in ledger_rows if row.get("artifact_role") == "ENGINEERING_SCREEN"
    ]
    if len(attempts) != 80 or len(screen_ledger_rows) != 80:
        errors.append("D2 screen source plan/ledger must each contain exactly 80 attempts")
    if len({row.get("attempt_id") for row in screen_ledger_rows}) != 80:
        errors.append("D2 screen exposure ledger attempt IDs are not unique")
    if screen_plan.get("outcomes_opened") != 0 or screen_plan.get("router_scores_read") != 0:
        errors.append("D2 source authoring opened outcomes or router scores")
    frozen_reset_seeds = {
        str(row["value"])
        for row in registry.get("identifiers", [])
        if row.get("identifier_type") == "reset_seed"
    }
    if not {str(row["reset_seed"]) for row in attempts} <= frozen_reset_seeds:
        errors.append("D2 planned reset seeds are not all frozen in exposure registry")
    formal_plan = read_json(
        ROOT
        / "results/expansion/d3_sources/formal_nominal_plan_a42e34a64bf2e0f8200e3171f80bd646e59ffc442edbcd09fa62fe376029e667.json"
    )
    formal_attempts = formal_plan.get("attempts", [])
    formal_ledger_rows = [
        row for row in ledger_rows if row.get("artifact_role") == "EXPOSED_NOMINAL_AUTHORING"
    ]
    if len(formal_attempts) != 126 or len(formal_ledger_rows) != 126:
        errors.append("D3 formal plan/ledger must each contain exactly 126 attempts")
    if len({row.get("attempt_id") for row in formal_ledger_rows}) != 126:
        errors.append("D3 formal exposure ledger attempt IDs are not unique")
    if formal_plan.get("outcomes_opened") != 0 or formal_plan.get("router_scores_read") != 0:
        errors.append("D3 source authoring opened outcomes or router scores")
    if not {str(row["reset_seed"]) for row in formal_attempts} <= frozen_reset_seeds:
        errors.append("D3 formal reset seeds are not all frozen in exposure registry")
    mechanism_config = read_json(ROOT / "configs/expansion/mechanism_screens_v1.yaml")
    if mechanism_config.get("unstable_final_placement_v2", {}).get(
        "requires_separate_pre_run_user_authorization"
    ) is not True:
        errors.append("D2 source freeze removed unstable-placement explicit authorization")
    nominal_root = ROOT / (
        "results/expansion/d2_preflight/77ea231ebf97_a1464cf4ddf0_20260830T084407Z"
    )
    nominal_cells = sorted(nominal_root.glob("cell_*.json"))
    if len(nominal_cells) != 8:
        errors.append("D2 nominal preflight must contain exactly eight cell artifacts")
    else:
        cell_payloads = [read_json(path) for path in nominal_cells]
        if sum(row.get("attempted_sources", 0) for row in cell_payloads) != 64:
            errors.append("D2 nominal preflight does not account for 64 planned attempts")
        if any(row.get("complete_sources") != 8 for row in cell_payloads):
            errors.append("D2 nominal preflight has an incomplete cell")
        if any(row.get("option_outcomes_opened") != 0 for row in cell_payloads):
            errors.append("D2 nominal preflight opened option outcomes")
    nominal_analysis = read_json(nominal_root / "nominal_preflight_analysis.json")
    if nominal_analysis.get("gate", {}).get("status") != "GO":
        errors.append("D2 nominal preflight Gate A0 is not GO")
    if nominal_analysis.get("task_summary", {}).get("0", {}).get(
        "nominal_success_rate"
    ) != 0.84375:
        errors.append("D2 nominal task0 success rate drift")
    if nominal_analysis.get("task_summary", {}).get("2", {}).get(
        "nominal_success_rate"
    ) != 1.0:
        errors.append("D2 nominal task2 success rate drift")
    mechanical_root = ROOT / (
        "results/expansion/d2_mechanical/d79973fd46e3_f2a8d011f62d_20260830T090458Z"
    )
    mechanical_cells = sorted(mechanical_root.glob("cell_*.json"))
    if len(mechanical_cells) != 8:
        errors.append("D2 mechanical preflight must contain exactly eight cell artifacts")
    else:
        mechanical_payloads = [read_json(path) for path in mechanical_cells]
        if sum(row.get("attempted_sources", 0) for row in mechanical_payloads) != 64:
            errors.append("D2 mechanical preflight does not account for 64 attempts")
        if any(row.get("mechanically_valid_sources") != 8 for row in mechanical_payloads):
            errors.append("D2 mechanical preflight has a non-valid source")
        if any(row.get("option_outcomes_opened") != 0 for row in mechanical_payloads):
            errors.append("D2 mechanical preflight opened option outcomes")
    mechanical_analysis = read_json(mechanical_root / "mechanical_preflight_analysis.json")
    if mechanical_analysis.get("gate", {}).get("status") != "GO":
        errors.append("D2 mechanical preflight gate is not GO")
    if mechanical_analysis.get("unstable_v2_authorized") is not False:
        errors.append("D2 mechanical preflight unexpectedly authorizes unstable v2")
    fragile_root = ROOT / (
        "results/expansion/d2_fragile_screen/"
        "0f270c3d9c8c_64cc00fdccbb_20260830T091340Z"
    )
    fragile_shards = sorted(fragile_root.glob("task*_source*.json"))
    if len(fragile_shards) != 16:
        errors.append("D2 fragile screen must contain exactly sixteen source shards")
    fragile_analysis = read_json(fragile_root / "fragile_screen_analysis.json")
    fragile_summary = fragile_analysis.get("summary", {})
    if fragile_analysis.get("gate", {}).get("status") != "SCOPED_CONTINUE":
        errors.append("D2 fragile screen decision must remain SCOPED_CONTINUE")
    expected_fragile = {
        "task0_eligible_sources": 8,
        "task2_eligible_sources": 8,
        "exact_restore_rate": 1.0,
        "admissible_execution_rate": 1.0,
        "benefit_zero_sources": 2,
        "benefit_one_sources": 14,
        "two_distinct_strict_winner_sources": 3,
    }
    for key, value in expected_fragile.items():
        if fragile_summary.get(key) != value:
            errors.append(f"D2 fragile screen summary drift: {key}")
    if fragile_analysis.get("gate", {}).get("claim_scope_failures") != [
        "benefit_zero_sources"
    ]:
        errors.append("D2 fragile screen failure scope drift")
    staleness_root = ROOT / (
        "results/expansion/d2_staleness_screen/"
        "1f39d658a829_0100901d1cc6_20260830T093356Z"
    )
    staleness_shards = sorted(staleness_root.glob("task*_source*.json"))
    if len(staleness_shards) != 16:
        errors.append("D2 staleness screen must contain exactly sixteen source shards")
    staleness = read_json(staleness_root / "staleness_screen_analysis.json")
    if staleness.get("gate", {}).get("status") != "GO":
        errors.append("D2 staleness screen gate is not GO")
    expected_staleness = {
        "task0_eligible_sources": 8,
        "task2_eligible_sources": 8,
        "exact_restore_rate": 1.0,
        "admissible_execution_rate": 1.0,
        "benefit_zero_sources": 6,
        "benefit_one_sources": 10,
        "two_distinct_strict_winner_sources": 2,
        "stale_base_catastrophe": 0.25,
        "stale_base_success": 0.25,
        "control_base_success": 0.8958333333333334,
    }
    for key, value in expected_staleness.items():
        if staleness.get("summary", {}).get(key) != value:
            errors.append(f"D2 staleness screen summary drift: {key}")
    drift_root = ROOT / (
        "results/expansion/d2_action_drift_screen/"
        "117904fb3d5c_cb828fc5e0e0_20260830T095028Z"
    )
    drift_shards = sorted(drift_root.glob("task*_source*.json"))
    if len(drift_shards) != 16:
        errors.append("D2 action-drift screen must contain exactly sixteen shards")
    drift = read_json(drift_root / "action_drift_screen_analysis.json")
    if drift.get("gate", {}).get("status") != "SCOPED_CONTINUE":
        errors.append("D2 action-drift decision must remain SCOPED_CONTINUE")
    expected_drift = {
        "task0_eligible_sources": 8,
        "task2_eligible_sources": 8,
        "exact_restore_rate": 1.0,
        "admissible_execution_rate": 1.0,
        "benefit_zero_sources": 0,
        "benefit_one_sources": 16,
        "two_distinct_strict_winner_sources": 1,
        "drift_base_catastrophe": 0.10416666666666667,
        "drift_base_success": 0.6041666666666666,
        "control_base_success": 0.8020833333333334,
    }
    for key, value in expected_drift.items():
        if drift.get("summary", {}).get(key) != value:
            errors.append(f"D2 action-drift summary drift: {key}")
    if drift.get("gate", {}).get("claim_scope_failures") != [
        "benefit_zero_sources",
        "two_distinct_strict_winner_sources",
    ]:
        errors.append("D2 action-drift failure scope drift")
    narrow_root = ROOT / (
        "results/expansion/d2_narrow_screen/"
        "a1d7e8ab026a_384b032a68fc_20260830T100829Z"
    )
    narrow_shards = sorted(narrow_root.glob("task*_source*.json"))
    if len(narrow_shards) != 16:
        errors.append("D2 narrow screen must contain exactly sixteen shards")
    narrow = read_json(narrow_root / "narrow_screen_analysis.json")
    if narrow.get("gate", {}).get("status") != "NO_GO":
        errors.append("D2 narrow screen must remain NO_GO")
    if narrow.get("gate", {}).get("hard_failures") != ["matched_control_catastrophe"]:
        errors.append("D2 narrow hard-failure scope drift")
    if narrow.get("summary", {}).get("matched_control_catastrophe") != 0.11363636363636363:
        errors.append("D2 narrow control catastrophe drift")
    selection = read_json(ROOT / "results/expansion/governance/d2_selection_decision.json")
    if selection.get("status") != "SINGLE_FORMAL_PLUS_EXPLORATORY_METHOD_PILOT":
        errors.append("D2 overall selection status drift")
    if selection.get("strict_broad_gate_pass") is not False:
        errors.append("D2 strict broad gate was incorrectly promoted")
    if selection.get("formal_go_count") != 1 or selection.get("scoped_continue_count") != 2:
        errors.append("D2 overall mechanism counts drift")
    formal_root = ROOT / (
        "results/expansion/d3_formal_nominal/05ddc6a3a7c3_acb3d2d28dc2_20260830T103120Z"
    )
    formal_shards = sorted(formal_root.glob("shard_*.json"))
    if len(formal_shards) != 6:
        errors.append("D3 formal nominal collection must contain six shards")
    formal_analysis = read_json(formal_root / "formal_nominal_analysis.json")
    if formal_analysis.get("status") != "GO":
        errors.append("D3 formal nominal selection is not GO")
    if formal_analysis.get("selected_physical_sources") != 100:
        errors.append("D3 formal nominal selection does not contain 100 sources")
    expected_formal_rates = {"0": 57 / 63, "2": 61 / 63}
    for task, rate in expected_formal_rates.items():
        if formal_analysis.get("task_summary", {}).get(task, {}).get(
            "nominal_success_rate"
        ) != rate:
            errors.append(f"D3 formal nominal task{task} success-rate drift")
    protocol = read_json(ROOT / "results/expansion/governance/protocol_v1_1_freeze.json")
    protocol_inputs = []
    for row in protocol.get("inputs", []):
        path = ROOT / row["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"]:
            errors.append(f"D4 protocol input drift: {row['path']}")
        protocol_inputs.append({"path": row["path"], "sha256": actual})
    actual_protocol_sha = hashlib.sha256(
        json.dumps(protocol_inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_protocol_sha != protocol.get("protocol_sha256"):
        errors.append("D4 composite protocol SHA mismatch")
    if protocol.get("test_authorized") is not False or protocol.get("test_outcomes_read") != 0:
        errors.append("D4 protocol prematurely opens test")
    split = read_json(ROOT / "results/expansion/governance/split_manifest_v1_1.json")
    split_payload = dict(split)
    split_hash = split_payload.pop("manifest_sha256", None)
    actual_split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if split_hash != actual_split_hash:
        errors.append("D4 split manifest self hash mismatch")
    if split.get("protocol_sha256") != protocol.get("protocol_sha256"):
        errors.append("D4 split manifest protocol mismatch")
    if split.get("physical_source_count") != 100 or split.get("policy_source_count") != 100:
        errors.append("D4 split manifest source count drift")
    if split.get("test_outcomes_read") != 0:
        errors.append("D4 split manifest records test outcome access")
    role_counts = {}
    for row in split.get("assignments", []):
        key = (row["task_id"], row["role"])
        role_counts[key] = role_counts.get(key, 0) + 1
    expected_roles = {
        (f"libero_spatial:{task}", role): count
        for task in (0, 2)
        for role, count in {
            "train": 12,
            "calibration": 6,
            "development": 6,
            "confirmatory_id_test": 16,
            "fresh_sequential_test": 10,
        }.items()
    }
    if role_counts != expected_roles:
        errors.append("D4 per-task split counts drift")
    return errors


def audit() -> list[str]:
    errors: list[str] = []
    # Keep one obvious documentation entrypoint. Historical narratives belong
    # in docs/archive rather than competing with README.md at repository root.
    for name in SUPERSEDED_ROOT_DOCS:
        if (ROOT / name).exists():
            errors.append(f"superseded root document reintroduced: {name}")
    for relative in REQUIRED_ARCHIVE_PATHS:
        if not (ROOT / relative).exists():
            errors.append(f"historical archive artifact missing: {relative}")
    for path in APPENDIX_DOCS:
        if not path.is_file():
            errors.append(f"appendix document missing: {path.relative_to(ROOT)}")

    # Current docs must share the complete C0..C14 ledger vocabulary, foreground
    # intervention-value routing, and reject the superseded E15-first framing.
    claim_ids = set(re.findall(r"\bC(?:1[0-4]|[0-9])\b", (ROOT / "docs/CLAIMS.md").read_text()))
    expected_claim_ids = {f"C{i}" for i in range(15)}
    if claim_ids != expected_claim_ids:
        errors.append(f"docs/CLAIMS.md IDs {sorted(claim_ids)} != {sorted(expected_claim_ids)}")
    for name in ("CURRENT.md", "PAPER_PLAN.md"):
        if "C0–C14" not in (ROOT / "docs" / name).read_text():
            errors.append(f"docs/{name} does not declare the C0–C14 claim vocabulary")
    for path in CURRENT_DOCS:
        text = path.read_text()
        for banned in BANNED_CURRENT_TEXT:
            if banned in text:
                errors.append(f"deprecated text {banned!r} in current doc {path.relative_to(ROOT)}")
    for path, required_tokens in REQUIRED_CURRENT_TEXT.items():
        text = path.read_text()
        normalized_text = " ".join(text.split())
        for token in required_tokens:
            if " ".join(token.split()) not in normalized_text:
                errors.append(
                    f"current framing token {token!r} missing from {path.relative_to(ROOT)}"
                )
    for name in ("CURRENT.md", "PAPER_PLAN.md", "CLAIMS.md"):
        if not (ROOT / "docs" / name).exists():
            errors.append(f"missing current document docs/{name}")

    maintained_link_docs = CURRENT_DOCS + APPENDIX_DOCS + ICLR27_DOCS + (
        ROOT / "setup/README.md",
        ROOT / "crashbench/README.md",
        ROOT / "docs/archive/README.md",
        ROOT / "docs/archive/glass_recovery_20260812/README.md",
        ROOT / "docs/archive/preliminary_reports/README.md",
        ROOT / "legacy/README.md",
    )
    errors.extend(markdown_local_link_errors(maintained_link_docs))
    errors.extend(iclr27_truth_source_errors())
    errors.extend(expansion_governance_errors())

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
            errors.append(f"legacy E15 smoke lacks required accepted-input token {token}")
    for token in (
        "train|evaluate",
        "CB_GLASS_RECOVERY_EVALUATION_COHORT",
        "CB_ENABLE_LEGACY_GLASS_RECOVERY",
    ):
        if token not in wrapper:
            errors.append(f"legacy E15 submit wrapper lacks required token {token}")
    for relative in (
        "setup/glass_recovery_smoke.sbatch",
        "setup/glass_recovery_pilot_b.sbatch",
        "setup/glass_core_realign.sbatch",
        "setup/submit_glass_recovery_smoke.sh",
        "setup/submit_glass_recovery_pilot_b.sh",
        "setup/submit_glass_core_realign.sh",
    ):
        if "CB_ENABLE_LEGACY_GLASS_RECOVERY" not in (ROOT / relative).read_text():
            errors.append(f"legacy glass entry point lacks opt-in guard: {relative}")
    for stale in (
        "results/glass_recovery_v1",
        "prepare_glass_recovery_placements.py",
        "collect_glass_recovery_pairs.py",
        "CB_GLASS_RECOVERY_SOURCE_RUN_ROOT",
        "--max-placements",
    ):
        if stale in smoke:
            errors.append(f"legacy E15 smoke still contains v1/authored-cohort path {stale}")

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

    # E15 Pilot A is a completed salvage/replay gate and Pilot B is a completed
    # frontier no-go. Pin both results without promoting either to a learned
    # recovery claim.
    e15_entry = next((entry for entry in manifest.get("entries", [])
                      if entry.get("experiment_id") == "E15"), None)
    if e15_entry is None:
        errors.append("manifest lacks E15 execution provenance")
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
            if "results/glass_recovery_pilot_a_20260811.json" not in e15_entry.get(
                "result_files", []
            ):
                errors.append("E15 manifest omits the Pilot A result")

        pilot_b_path = ROOT / "results/glass_recovery_pilot_b_frontier_20260811.json"
        if pilot_b_path.exists():
            pilot_b = read_json(pilot_b_path)
            expected = {
                "kind": "glass_recovery_pilot_b_frontier_result",
                "status": "pilot_b_frontier_complete_no_go",
                "decision.value": "pilot_b_no_go",
                "decision.go": False,
                "decision.recommended_horizon_actions": None,
                "decision.qualified_horizons": [],
                "decision.pilot_b_fixed_h_collection_executed": False,
                "decision.pilot_c_allowed": False,
                "provenance.runner_git_commit": "dd10252fdab4648423010c4666833b539ca402bf",
                "provenance.slurm_job.job_id": 9055676,
                "provenance.slurm_job.state": "FAILED",
                "provenance.slurm_job.exit_code": "1:0",
                "candidate_design.candidates": 20,
                "candidate_design.physical_scenes": 20,
                "candidate_design.unique_source_states": 18,
                "attempt_accounting.terminal_attempts": 120,
                "attempt_accounting.unique_attempt_keys": 120,
                "attempt_accounting.attempts_have_unique_provenance": True,
                "attempt_accounting.one_terminal_attempt_per_candidate_horizon": True,
                "attempt_accounting.technical_failures": 0,
                "attempt_accounting.accepted_pairs_across_all_horizons": 5,
            }
            for dotted, value in expected.items():
                if dotted_get(pilot_b, dotted) != value:
                    errors.append(f"E15 Pilot B mismatch: {dotted}")
            horizons = pilot_b.get("frontier", {}).get("horizons", {})
            if set(horizons) != {"40", "30", "20", "15", "10", "5"}:
                errors.append("E15 Pilot B horizon grid differs from the frozen frontier")
            elif any(row.get("qualified") is not False for row in horizons.values()):
                errors.append("E15 Pilot B no-go contains a qualified horizon")
            if "results/glass_recovery_pilot_b_frontier_20260811.json" not in (
                e15_entry.get("result_files", [])
            ):
                errors.append("E15 manifest omits the Pilot B frontier result")
            if e15_entry.get("git_commit") != pilot_b["provenance"]["runner_git_commit"]:
                errors.append("E15 Pilot B run commit differs from manifest")
            if e15_entry.get("status") != pilot_b["status"]:
                errors.append("E15 Pilot B status differs from manifest")
            if set(e15_entry.get("scenario_fingerprints", [])) != set(
                pilot_b.get("candidate_design", {}).get("physical_scene_sha256", [])
            ):
                errors.append("E15 Pilot B physical-scene fingerprints differ from manifest")

    # The scoped re-entry is retained only as a development Oracle upper bound.
    # Pin the small independent unit and validation reuse so 6/6 cannot drift
    # into learned or final-held-out wording during future documentation edits.
    scoped_entry = next((entry for entry in manifest.get("entries", [])
                         if entry.get("experiment_id") == "E15-SCOPED"), None)
    if scoped_entry is None:
        errors.append("manifest lacks E15-SCOPED Oracle-upper-bound provenance")
    else:
        scoped_b_path = ROOT / "results/glass_recovery_pilot_b_scoped_20260812.json"
        scoped_c_path = ROOT / "results/glass_recovery_pilot_c_scoped_20260812.json"
        readiness_path = ROOT / "results/glass_recovery_checkpoint_readiness_audit_20260812.json"
        if scoped_b_path.exists() and scoped_c_path.exists() and readiness_path.exists():
            scoped_b = read_json(scoped_b_path)
            scoped_c = read_json(scoped_c_path)
            readiness = read_json(readiness_path)
            expected_b = {
                "status": "scoped_pilot_b_certification_complete",
                "decision.broad_population_claim_allowed": False,
                "frontier.horizon_actions": 20,
                "frontier.candidate_scenes": 15,
                "frontier.base_catastrophes": 12,
                "frontier.exact_h_replays": 9,
                "frontier.accepted_pairs": 3,
                "frontier.yield_given_base_catastrophe": 0.25,
            }
            for dotted, value in expected_b.items():
                if dotted_get(scoped_b, dotted) != value:
                    errors.append(f"E15-SCOPED Pilot B mismatch: {dotted}")
            expected_c = {
                "status": "scoped_development_oracle_upper_bound_complete_go",
                "scope.evaluation_mode": "exact_anchor",
                "scope.condition": "oracle_timed_oracle_recovery",
                "scope.cohort_role": "development_only",
                "metrics.source_states": 2,
                "metrics.episodes": 6,
                "metrics.safe_task_successes": 6,
                "metrics.catastrophes": 0,
                "metrics.exact_simulator_controller_restores": 6,
                "training_and_protocol.training_pairs": 1,
                "training_and_protocol.checkpoint_validation_pairs": 2,
                "training_and_protocol.checkpoint_validation_reuses_development_evaluation_pairs": True,
            }
            for dotted, value in expected_c.items():
                if dotted_get(scoped_c, dotted) != value:
                    errors.append(f"E15-SCOPED Pilot C mismatch: {dotted}")
            accepted = scoped_b.get("accepted", [])
            development_ids = {
                row.get("placement_id") for row in accepted
                if row.get("pilot_c_role") == "development_evaluation"
            }
            evaluated_ids = {
                row.get("placement_id") for row in scoped_c.get("by_pair", [])
            }
            if len({row.get("source_state_sha256") for row in accepted}) != 3:
                errors.append("E15-SCOPED accepted pairs do not have three source states")
            if development_ids != evaluated_ids or len(evaluated_ids) != 2:
                errors.append("E15-SCOPED Pilot C cohort differs from Pilot B development pairs")
            if (
                scoped_b["provenance"]["primary_protocol_sha256"]
                != scoped_c["training_and_protocol"]["primary_protocol_sha256"]
            ):
                errors.append("E15-SCOPED Pilot B/C primary protocol hashes differ")
            expected_readiness = {
                "kind": "glass_recovery_checkpoint_readiness_audit",
                "status": "not_ready_for_learned_timing_or_joint_recovery",
                "source.episodes": 6,
                "source.source_states": 2,
                "source.evaluation_git_commit": "b16bce3a93aa7819a237ba376bb50a8281d4f066",
                "checkpoint.training_git_commit": "2d53dcfef2a93a28be042ff4057520f7ba66a331",
                "checkpoint.validation_manifest_sha256": "139a13ccaa9220f7c001fec2bfa78dc64a2115855c38070ade104f574501cba3",
                "checkpoint.evaluation_trajectory_manifest_sha256": "139a13ccaa9220f7c001fec2bfa78dc64a2115855c38070ade104f574501cba3",
                "timing_gate.threshold": 1.0,
                "timing_gate.checkpoint_timely_trigger_rate": 0.0,
                "timing_gate.threshold_crossing_episodes": 0,
                "timing_gate.pilot_d_decision": "no_go_at_frozen_checkpoint",
                "action_head.validation_gripper_sign_accuracy": 0.8458646616541353,
                "action_head.required_validation_gripper_sign_accuracy": 0.95,
                "action_head.passes_gate": False,
                "action_head.pilot_e_status": "not_run",
                "pilot_decisions.pilot_f": "do_not_run_before_independent_d_and_e_signals",
            }
            for dotted, value in expected_readiness.items():
                if dotted_get(readiness, dotted) != value:
                    errors.append(f"E15-SCOPED checkpoint readiness mismatch: {dotted}")
            maxima = readiness.get("timing_gate", {}).get(
                "selected_risk_probability_maxima", []
            )
            if len(maxima) != 6 or max(row.get("maximum", 1.0) for row in maxima) >= 1.0:
                errors.append("E15-SCOPED readiness audit does not pin six subthreshold episodes")
            if (
                readiness.get("source", {}).get("evaluation_sha256")
                != scoped_c.get("provenance", {}).get("evaluation_json_sha256")
            ):
                errors.append("E15-SCOPED readiness source hash differs from Pilot C summary")
            raw_eval = ROOT / readiness.get("source", {}).get("evaluation_path_ignored", "")
            if raw_eval.is_file():
                raw_bytes = raw_eval.read_bytes()
                if hashlib.sha256(raw_bytes).hexdigest() != readiness["source"]["evaluation_sha256"]:
                    errors.append("local Pilot C evaluation differs from readiness source hash")
                else:
                    raw_payload = json.loads(raw_bytes)
                    raw_maxima = [
                        {
                            "pair_id": episode["pair_id"],
                            "rollout_seed": episode["rollout_seed"],
                            "maximum": max(
                                frame["selected_risk_probability"] for frame in episode["trace"]
                            ),
                        }
                        for episode in raw_payload["episodes"]
                    ]
                    if raw_maxima != maxima:
                        errors.append("readiness risk maxima differ from local Pilot C evaluation")
                    crossing_count = sum(
                        any(frame["threshold_crossing"] for frame in episode["trace"])
                        for episode in raw_payload["episodes"]
                    )
                    if crossing_count != readiness["timing_gate"]["threshold_crossing_episodes"]:
                        errors.append("readiness crossing count differs from local Pilot C evaluation")
            expected_files = {
                "results/glass_recovery_pilot_b_scoped_20260812.json",
                "results/glass_recovery_pilot_c_scoped_20260812.json",
            }
            if set(scoped_entry.get("result_files", [])) != expected_files:
                errors.append("E15-SCOPED manifest must contain only scoped Pilot B/C results")
            if "results/glass_recovery_checkpoint_readiness_audit_20260812.json" in set(
                scoped_entry.get("result_files", [])
            ):
                errors.append(
                    "derived readiness audit is incorrectly attributed to E15-SCOPED run provenance"
                )

            readiness_entry = next((entry for entry in manifest.get("entries", [])
                                    if entry.get("experiment_id") == "E15-SCOPED-READINESS"), None)
            if readiness_entry is None:
                errors.append("manifest lacks independent E15 checkpoint-readiness provenance")
            else:
                if readiness_entry.get("result_files") != [
                    "results/glass_recovery_checkpoint_readiness_audit_20260812.json"
                ]:
                    errors.append("E15 checkpoint-readiness manifest result differs from decision record")
                if readiness_entry.get("generating_script") is not None:
                    errors.append("E15 checkpoint-readiness manifest fabricates a generating script")
                if readiness_entry.get("analysis_script") is not None:
                    errors.append("E15 checkpoint-readiness manifest fabricates an analysis script")
                if readiness_entry.get("sbatch") != "none (zero-GPU direct extraction)":
                    errors.append("E15 checkpoint-readiness manifest does not identify zero-GPU extraction")
                if readiness_entry.get("status") != readiness.get("status"):
                    errors.append("E15 checkpoint-readiness status differs from manifest")

    # Pilot A/B entries must not be mistaken for a learned result. If E15 is ever
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
    print(
        "Repository audit passed: intervention-value routing framing, maintained links, "
        "legacy glass guards, frozen E14/E15 semantics, scenario fingerprints, "
        "manifest paths, claim checks, and the ICLR27 truth source are consistent."
    )


if __name__ == "__main__":
    main()
