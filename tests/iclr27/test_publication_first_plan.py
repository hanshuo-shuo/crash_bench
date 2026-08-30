from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "docs/iclr27/PUBLICATION_FIRST_RESOLUTION.md"
V1_AUDIT = ROOT / "docs/iclr27/NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md"


def test_publication_first_plan_has_a_terminal_writing_objective():
    text = CANONICAL.read_text()
    assert "PUBLICATION_FIRST_GLASS_SCOPED_DIAGNOSTIC" in text
    assert "New outcome-bearing experiments required before writing:** none" in text
    assert "PAPER_PACKAGE_READY_GLASS_SCOPED" in text
    assert "Completion does not depend on non-glass v2" in text
    assert "No Phase 2.5B-R2" in text


def test_publication_claim_and_frozen_method_null_are_both_preserved():
    text = CANONICAL.read_text()
    assert "19/273" in text
    assert "three tight" in text
    assert "0.2544" in text
    assert "0.5642" in text
    assert "+0.0178" in text
    assert "24/24" in text
    assert "0/8" in text
    assert "successful, superior, or deployable learned router" in text


def test_non_glass_v1_is_not_misclassified_as_scientific_evidence():
    text = V1_AUDIT.read_text()
    assert "INVALID_PREFLIGHT_ZERO_OPTION_OUTCOMES" in text
    assert "5165648" in text
    assert "53f62a64ae5b184b70676491e4d5353aa0d072ea" in text
    assert "c0bb9015a0a3dfb4aa9d70672e2c7a046c56f688842d880e46b8d46a68bc9413" in text
    assert "Ordinary option-outcome rows | 0" in text
    assert "does not establish" in text


def test_optional_v2_does_not_recreate_the_moving_finish_line():
    text = CANONICAL.read_text()
    assert "This section is inactive unless the user explicitly chooses" in text
    assert "Strict Base support remains a reported diagnostic" in text
    assert "V2_VALID_REPLICATED_INTERVENTION_FLIP" in text
    assert "V2_VALID_PARTIAL_SUPPORT" in text
    assert "V2_VALID_NULL" in text
    assert "V2_BLOCKED_ENGINEERING" in text
    assert "does not authorize a v3 rescue" in text
    assert "eight v1 source hashes/seeds remain permanently excluded" in text


def test_active_navigation_files_point_to_the_canonical_resolution():
    paths = (
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "CrashBench_ICLR2027_Revised_Plan_After_Baseline_NoGo.md",
        ROOT / "docs/CURRENT.md",
        ROOT / "docs/PAPER_PLAN.md",
        ROOT / "docs/PAPER_PLAN_RISK_IS_NOT_REGRET.md",
        ROOT / "docs/EXPERIMENT_INDEX.md",
        ROOT / "docs/iclr27/MASTER_PLAN.md",
        ROOT / "docs/iclr27/CLAIM_LEDGER.md",
        ROOT / "docs/iclr27/STORY_PIVOT_RISK_IS_NOT_REGRET.md",
        ROOT / "docs/iclr27/BALANCED_INTERVENTION_BENCHMARK_SPEC.md",
    )
    for path in paths:
        text = path.read_text()
        assert "PUBLICATION_FIRST_RESOLUTION.md" in text, path
