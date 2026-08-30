from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/iclr27/audit_paper_package.py"
MODULE_SPEC = importlib.util.spec_from_file_location("paper_package_audit", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
paper_audit = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(paper_audit)
PIVOT_DIR = ROOT / paper_audit.PIVOT_RELATIVE_DIR


DISPLAYED_VALUES = {
    "corpus": "273 eligible decisions / 20 exposed-development sources",
    "risk_benefit_disagreement": "19/273 decisions across 12 sources",
    "tight_witnesses": "3 tight pairs / 2 sources",
    "oracle_opportunity": "0.2544 -> 0.5642; opportunity 0.3097",
    "outcome_router_null": "+0.0178; method-gate null",
    "tiny_adr_null": "0.2311; below the risk reference",
    "glass_concentration": "23 strict Retreat states; 51/60 strict D/R states",
    "sequential_boundary": "24/24 Base; 0/8 recovered; 2/2 opportunities missed",
}

EFFECTIVE_SOURCES = {
    "corpus": "20 denominator sources",
    "risk_benefit_disagreement": "12 disagreement sources; 20 denominator sources",
    "tight_witnesses": "2 sources overall; 1 source for Detour-Retreat",
    "oracle_opportunity": "20 source-macro blocks",
    "outcome_router_null": "20 outer source-held-out blocks",
    "tiny_adr_null": "20 outer source-held-out blocks",
    "glass_concentration": "9 Retreat sources; 16 glass D/R sources; 17 overall sources",
    "sequential_boundary": "8 fresh sources; 24 source-condition episodes",
}

EVIDENCE_SCOPES = {
    "corpus": "frozen glass_recovery synthesis; all sources are exposed development",
    "risk_benefit_disagreement": "frozen exposed-development glass_recovery corpus",
    "tight_witnesses": "frozen exposed-development glass_recovery corpus",
    "oracle_opportunity": "exposed-development attribution over Base/Detour/Retreat",
    "outcome_router_null": "Phase 2.5B exposed-development evaluation",
    "tiny_adr_null": "Phase 2.5B-R exposed-development capacity diagnosis",
    "glass_concentration": "one glass_recovery family with offpath and noglass controls",
    "sequential_boundary": "fresh sequential closeout in the glass_recovery design",
}


def _write_artifact_map(package_dir: Path) -> None:
    path = package_dir / "artifact_map.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paper_audit.ARTIFACT_MAP_COLUMNS)
        writer.writeheader()
        for key, (artifact, digest, _) in paper_audit.EXPECTED_ARTIFACT_MAP.items():
            generation_commit, record_commit = paper_audit.EXPECTED_ARTIFACT_COMMITS[key]
            writer.writerow({
                "claim_key": key,
                "displayed_value": DISPLAYED_VALUES[key],
                "canonical_artifact": artifact,
                "selector": "frozen reviewed row or manifest field",
                "value_columns": "documented in ARTIFACT_MAP.md",
                "independent_unit": "source_state_sha256",
                "effective_sources": EFFECTIVE_SOURCES[key],
                "evidence_scope": EVIDENCE_SCOPES[key],
                "artifact_sha256": digest,
                "generation_commit": generation_commit,
                "record_commit": record_commit,
                "repository_availability": (
                    paper_audit.EXPECTED_REPOSITORY_AVAILABILITY[key]
                ),
                "caveat": "diagnostic evidence only; no broad or deployable claim",
            })


@pytest.fixture()
def valid_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "manuscript"
    package_dir.mkdir()

    manuscript = r"""
\documentclass{article}
\begin{document}
\begin{abstract}
Across 273 eligible decisions, 19/273 disagree in risk and benefit. Three tight
matched pairs span two sources. Risk-to-best-fixed utility is 0.2544 and Oracle
utility is 0.5642, leaving opportunity 0.3097. OutcomeRouter adds +0.0178, but
the learned-method result is null because no method gate passed. All 23 strict
Retreat states and 51/60 strict Detour-or-Retreat states are glass. Fresh
selection chose Base in 24/24 episodes, recovered 0/8, and missed 2/2 known
opportunities.
\end{abstract}
\begin{figure}
\caption{Effective source count: 20 sources. One \texttt{glass\_recovery}
mechanical family is separated into the \texttt{glass}, \texttt{offpath}, and
\texttt{noglass} conditions.}
\end{figure}
This exact-state diagnostic does not provide a successful or deployable learned
router. It is not a broad VLA-safety benchmark, does not claim generality across
tasks, hazards, backbones, or mechanisms, and does not establish reliable
sequential intervention.
\end{document}
"""
    (package_dir / "manuscript.tex").write_text(manuscript, encoding="utf-8")
    (package_dir / "appendix.tex").write_text(
        "Evidence contracts and source-aware tables.\n", encoding="utf-8"
    )
    (package_dir / "references.bib").write_text(
        "@misc{case-study, title={Scoped exact-state diagnostics}}\n",
        encoding="utf-8",
    )
    safe_markdown = (
        "# Scoped package\n\nThis is not a broad benchmark and does not claim a "
        "successful router or reliable sequential intervention.\n"
    )
    for name in (
        "README.md",
        "ARTIFACT_MAP.md",
        "FIGURE_TABLE_INVENTORY.md",
        "REPRODUCIBILITY_STATEMENT.md",
        "LIMITATIONS_AND_ETHICS.md",
        "CONSISTENCY_AUDIT.md",
        "COVER_SUMMARY.md",
    ):
        (package_dir / name).write_text(safe_markdown, encoding="utf-8")
    _write_artifact_map(package_dir)

    output = package_dir / paper_audit.FINAL_PDF_RELATIVE
    output.parent.mkdir(parents=True)
    shutil.copyfile(PIVOT_DIR / "figure_risk_benefit_flow.pdf", output)
    return package_dir


def test_current_frozen_pivot_manifest_and_hashes_pass() -> None:
    assert paper_audit.audit_pivot_manifest(ROOT) == []


def test_minimal_complete_package_passes(valid_package: Path) -> None:
    assert paper_audit.audit_paper_package(
        valid_package,
        repo_root=ROOT,
        require_clean=False,
    ) == []


def test_canonical_package_uses_repository_output_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    package_dir = repo_root / "docs/iclr27/manuscript"
    assert paper_audit.final_pdf_path(package_dir, repo_root) == (
        repo_root / paper_audit.FINAL_PDF_RELATIVE
    )
    custom = tmp_path / "custom-package"
    assert paper_audit.final_pdf_path(custom, repo_root) == (
        custom / paper_audit.FINAL_PDF_RELATIVE
    )


def test_pivot_hash_drift_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / paper_audit.PIVOT_RELATIVE_DIR
    copied.parent.mkdir(parents=True)
    shutil.copytree(PIVOT_DIR, copied)
    with (copied / "family_support.csv").open("a", encoding="utf-8") as handle:
        handle.write("drift\n")

    errors = paper_audit.audit_pivot_manifest(tmp_path)
    assert any("family_support.csv" in error and "hash drift" in error for error in errors)


def test_abstract_requires_each_frozen_headline_fact(valid_package: Path) -> None:
    path = valid_package / "manuscript.tex"
    path.write_text(
        path.read_text(encoding="utf-8").replace("19/273", "nineteen decisions"),
        encoding="utf-8",
    )

    errors = paper_audit.abstract_errors(valid_package)
    assert "abstract lacks required fact: 19/273 disagreement" in errors


def test_claim_scan_accepts_limits_and_rejects_affirmative_overclaims() -> None:
    safe = """
    This is not a broad VLA-safety benchmark. We do not present a successful
    learned router, do not claim generality across hazards, and do not establish
    reliable sequential intervention. Non-glass v1 provides no scientific evidence.
    """
    assert paper_audit.affirmative_claim_errors(safe, "safe.tex") == []

    unsafe = """
    We present a successful learned router and a broad VLA-safety benchmark.
    The result establishes generality across tasks and reliable sequential intervention.
    """
    errors = paper_audit.affirmative_claim_errors(unsafe, "unsafe.tex")
    assert any("successful/superior/deployable router" in error for error in errors)
    assert any("broad VLA-safety benchmark" in error for error in errors)
    assert any("cross-domain generality" in error for error in errors)
    assert any("reliable sequential intervention" in error for error in errors)

    adversative = "We do not beat a baseline, but we present a successful router."
    assert any(
        "successful/superior/deployable router" in error
        for error in paper_audit.affirmative_claim_errors(adversative, "but.tex")
    )

    tricky = (
        "This is not a successful router, but it is a successful router. "
        "Without new experiments, this is a successful learned router. "
        "This is not a theorem: it is a broad VLA-safety benchmark. "
        "The selector works across tasks and hazards."
    )
    tricky_errors = paper_audit.affirmative_claim_errors(tricky, "tricky.tex")
    assert sum("successful/superior/deployable router" in error for error in tricky_errors) == 2
    assert any("broad VLA-safety benchmark" in error for error in tricky_errors)
    assert any("cross-domain generality" in error for error in tricky_errors)


def test_claim_scan_accepts_an_explicit_not_established_table_column() -> None:
    table = r"""
    \begin{tabular}{ll}
    Supported within scope & Not established \\
    Benefit gating dominates the reported loss. & A successful, superior, or
    deployable learned router. \\
    \end{tabular}
    """
    assert paper_audit.affirmative_claim_errors(table, "boundary.tex") == []


def test_caption_requires_source_count_and_single_family_conditions(
    valid_package: Path,
) -> None:
    path = valid_package / "manuscript.tex"
    text = path.read_text(encoding="utf-8")
    start = text.index("\\caption{")
    end = text.index("\n\\end{figure}", start)
    path.write_text(
        text[:start] + "\\caption{Pooled branches.}" + text[end:],
        encoding="utf-8",
    )

    errors = paper_audit.caption_errors(valid_package)
    assert any("effective source count" in error for error in errors)
    assert any("one glass_recovery family" in error for error in errors)


@pytest.mark.parametrize(
    "caption",
    (
        "Results use the same 20-source contract.",
        "The closeout uses one fresh eight-source cohort.",
        "Utilities use 20 exposed-development sources.",
        "Effective independent sources: 20.",
        "Supporting-source counts are shown and do not sum to 20.",
    ),
)
def test_caption_accepts_unambiguous_source_count_forms(
    valid_package: Path,
    caption: str,
) -> None:
    path = valid_package / "manuscript.tex"
    text = path.read_text(encoding="utf-8")
    insert_at = text.index("\\end{document}")
    text = text[:insert_at] + f"\\caption{{{caption}}}\n" + text[insert_at:]
    path.write_text(text, encoding="utf-8")

    assert not any(
        "caption lacks an effective source count" in error
        for error in paper_audit.caption_errors(valid_package)
    )


@pytest.mark.parametrize("field", ["canonical_artifact", "artifact_sha256"])
def test_artifact_map_rejects_noncanonical_path_or_hash(
    valid_package: Path,
    field: str,
) -> None:
    path = valid_package / "artifact_map.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = "0" * 64 if field == "artifact_sha256" else "results/not-canonical.json"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paper_audit.ARTIFACT_MAP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    errors = paper_audit.artifact_map_errors(valid_package, ROOT)
    assert any("corpus" in error and ("canonical path" in error or "SHA-256" in error) for error in errors)


def test_placeholder_and_invalid_pdf_are_rejected(valid_package: Path) -> None:
    (valid_package / "README.md").write_text("TBD\n", encoding="utf-8")
    (valid_package / paper_audit.FINAL_PDF_RELATIVE).write_bytes(b"not a PDF")

    assert any("placeholder" in error for error in paper_audit.placeholder_errors(valid_package))
    assert paper_audit.pdf_errors(valid_package) == [
        "final paper PDF does not start with %PDF-"
    ]


def test_require_clean_gate_uses_porcelain_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="?? draft.tex\n", stderr="")

    monkeypatch.setattr(paper_audit.subprocess, "run", fake_run)
    errors = paper_audit.clean_worktree_errors(ROOT)

    assert calls == [["git", "status", "--porcelain"]]
    assert errors == ["git worktree is not clean: ?? draft.tex"]
