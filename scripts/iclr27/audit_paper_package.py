#!/usr/bin/env python3
"""Fail-closed audit for the glass-scoped ICLR27 paper package.

This audit is deliberately analysis-only.  It reads the tracked PIVOT-0
evidence package and the submission-facing files under
``docs/iclr27/manuscript``; it never launches a simulator, fits a model, or
rewrites an artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_DIR = ROOT / "docs/iclr27/manuscript"
PIVOT_RELATIVE_DIR = Path(
    "results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z"
)
FINAL_PDF_RELATIVE = Path(
    "output/pdf/risk_does_not_specify_intervention_glass_scoped.pdf"
)

REQUIRED_PACKAGE_FILES = (
    "README.md",
    "manuscript.tex",
    "appendix.tex",
    "references.bib",
    "ARTIFACT_MAP.md",
    "artifact_map.csv",
    "FIGURE_TABLE_INVENTORY.md",
    "REPRODUCIBILITY_STATEMENT.md",
    "LIMITATIONS_AND_ETHICS.md",
    "CONSISTENCY_AUDIT.md",
    "COVER_SUMMARY.md",
)

# This operational record necessarily spells out the forbidden phrases and the
# placeholder rule that the scanner enforces.  It is audited structurally, but
# is not submission prose and must not be fed back into its own lexical scan.
AUDIT_RECORD_NAME = "CONSISTENCY_AUDIT.md"

EXPECTED_PIVOT_REQUIRED_ARTIFACTS = (
    "manifest.json",
    "risk_benefit_crosstab.csv",
    "source_support.csv",
    "family_support.csv",
    "risk_only_policy_regret.csv",
    "oracle_hybrid_waterfall.csv",
    "same_risk_different_decision_witnesses.csv",
    "sequential_realization_gap.csv",
    "story_decision.json",
    "figure_risk_benefit_flow.pdf",
    "figure_oracle_gap_waterfall.pdf",
    "figure_source_support.pdf",
)

ARTIFACT_MAP_COLUMNS = (
    "claim_key",
    "displayed_value",
    "canonical_artifact",
    "selector",
    "value_columns",
    "independent_unit",
    "effective_sources",
    "evidence_scope",
    "artifact_sha256",
    "generation_commit",
    "record_commit",
    "repository_availability",
    "caveat",
)

EXPECTED_ARTIFACT_MAP = {
    "corpus": (
        "results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/manifest.json",
        "dbb3df3b3c94b26cd0b923a864b47ea25da36a279f3f4d36b0272ccf93110bd7",
        ("273", "20"),
    ),
    "risk_benefit_disagreement": (
        "results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/risk_benefit_crosstab.csv",
        "ecc7bc9f76eb59c04aa9f398aa8fb63ff60698265e9fd9659ba948fb92e52bef",
        ("19/273", "12"),
    ),
    "tight_witnesses": (
        "results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/same_risk_different_decision_witnesses.csv",
        "4cbb9c8086af86e51f02bb1fc719f3001e01f1682ae1895e666a8e77edf4c305",
        ("3", "2"),
    ),
    "oracle_opportunity": (
        "results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/oracle_hybrid_waterfall.csv",
        "919479455cb8c2c2243a01845c29e69ed29714150f031282ab45aebf2b9822b3",
        ("0.2544", "0.5642", "0.3097"),
    ),
    "outcome_router_null": (
        "results/iclr27/support_crossfit_df3168c75843_20260829T115455Z_job5137872/gate_decision.json",
        "b152a143dcabc17f5dd12593af3d96f68b0db363f0925f3a3305c3ed40e09e96",
        ("+0.0178",),
    ),
    "tiny_adr_null": (
        "results/iclr27/advantage_router_resolver_d4751330395e_20260829T135723Z_job5148751/overall_metrics.csv",
        "a461daef7af23ccd64fc8874ca8c58b7c93fd5e26ee1c944b872817011edb29b",
        ("0.2311",),
    ),
    "glass_concentration": (
        "results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/family_support.csv",
        "85e715d8ce266e542c684badbc04124b299fbaeb83b9838a01814bf6a1170ba0",
        ("23", "51/60"),
    ),
    "sequential_boundary": (
        "results/p3_2_frozen_dynamic_closeout_analysis_20260819.json",
        "01d1a76707d34aecbe24999d5abd613b8f0456d7e60467f888531bd417ceaad1",
        ("24/24", "0/8", "2/2"),
    ),
}

EXPECTED_ARTIFACT_COMMITS = {
    "corpus": (
        "98b3ffe0aec2838448d0536b6c62036e37487f6f",
        "cdc4802494a7c0e540fe80bfebc2dfc304af35d0",
    ),
    "risk_benefit_disagreement": (
        "98b3ffe0aec2838448d0536b6c62036e37487f6f",
        "cdc4802494a7c0e540fe80bfebc2dfc304af35d0",
    ),
    "tight_witnesses": (
        "98b3ffe0aec2838448d0536b6c62036e37487f6f",
        "cdc4802494a7c0e540fe80bfebc2dfc304af35d0",
    ),
    "oracle_opportunity": (
        "98b3ffe0aec2838448d0536b6c62036e37487f6f",
        "cdc4802494a7c0e540fe80bfebc2dfc304af35d0",
    ),
    "outcome_router_null": (
        "df3168c75843f33484209b9908ac939814de31e9",
        "d2fc01d7fa4745d97e512f52eff33bc89014ae86",
    ),
    "tiny_adr_null": (
        "bf032cb1d6d482423b229e304997ea29134bb4bb",
        "e36ab5203997bd7e01115663e28c1935b1f1de14",
    ),
    "glass_concentration": (
        "98b3ffe0aec2838448d0536b6c62036e37487f6f",
        "cdc4802494a7c0e540fe80bfebc2dfc304af35d0",
    ),
    "sequential_boundary": (
        "c3874a146632e76596c1b21039b9f50186df1dbf",
        "aa5e81769f74ee58b7d30250144781e8a5b9cdd6",
    ),
}

EXPECTED_REPOSITORY_AVAILABILITY = {
    "corpus": "tracked_summary_self_unpinned; raw_capture_local_untracked_hash_pinned",
    "risk_benefit_disagreement": "tracked_sha_pinned_by_pivot_manifest",
    "tight_witnesses": "tracked_sha_pinned_by_pivot_manifest",
    "oracle_opportunity": "tracked_sha_pinned_by_pivot_manifest",
    "outcome_router_null": "tracked_sha_pinned_by_root_and_phase_manifests",
    "tiny_adr_null": "tracked_sha_pinned_by_root_and_phase_manifests",
    "glass_concentration": "tracked_sha_pinned_by_pivot_manifest",
    "sequential_boundary": "tracked_sha_pinned_by_root_and_pivot_manifests",
}

EXPECTED_EFFECTIVE_SOURCE_TOKENS = {
    "corpus": ("20",),
    "risk_benefit_disagreement": ("12", "20"),
    "tight_witnesses": ("2", "1"),
    "oracle_opportunity": ("20",),
    "outcome_router_null": ("20",),
    "tiny_adr_null": ("20",),
    "glass_concentration": ("9", "16", "17"),
    "sequential_boundary": ("8", "24"),
}

EXPECTED_SCOPE_TOKENS = {
    "corpus": ("glass_recovery", "exposed"),
    "risk_benefit_disagreement": ("glass_recovery", "exposed"),
    "tight_witnesses": ("glass_recovery", "exposed"),
    "oracle_opportunity": ("exposed-development", "Base/Detour/Retreat"),
    "outcome_router_null": ("Phase 2.5B", "exposed-development"),
    "tiny_adr_null": ("Phase 2.5B-R", "exposed-development"),
    "glass_concentration": ("glass_recovery", "offpath", "noglass"),
    "sequential_boundary": ("glass_recovery", "fresh sequential"),
}

PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|PLACEHOLDER)\b", re.IGNORECASE)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

# These expressions target affirmative overclaims, not mere keyword use.  A
# same-clause explicit negation/limitation is accepted below.
DANGEROUS_CLAIM_PATTERNS = (
    (
        "successful/superior/deployable router",
        re.compile(
            r"\b(?:(?:successful|superior|deployable|production[- ]ready)\s+"
            r"(?:learned\s+)?router|(?:learned\s+)?router\s+(?:is|was|becomes?|"
            r"proved|constitutes?)\s+(?:an?\s+)?(?:successful|superior|deployable|"
            r"production[- ]ready))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "broad VLA-safety benchmark",
        re.compile(
            r"\b(?:broad|comprehensive|general[- ]purpose)\s+"
            r"(?:vla[- ]safety\s+)?benchmark\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cross-domain generality",
        re.compile(
            r"\b(?:general(?:ity|ization|izes?|ized)?|robust(?:ness)?|"
            r"architecture[- ]independent|works?|performs?|transfers?|applies?)\b"
            r".{0,60}\b(?:across|to)\b.{0,60}"
            r"\b(?:tasks?|hazards?|backbones?|mechanisms?|architectures?|vlas?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reliable sequential intervention",
        re.compile(
            r"\b(?:reliable|reliably)\b.{0,50}\bsequential\b.{0,50}"
            r"\b(?:intervention|recovery|rescue|realization)\b|"
            r"\bsequential\b.{0,50}\b(?:intervention|recovery|rescue|realization)\b"
            r".{0,50}\b(?:is|was|proved|becomes?)\s+(?:reliable|reliably)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "end-to-end learned recovery",
        re.compile(r"\bend[- ]to[- ]end\s+learned\s+recovery\b", re.IGNORECASE),
    ),
    (
        "multiple independent mechanical families",
        re.compile(
            r"\b(?:three|3|multiple)\s+(?:independent\s+)?(?:mechanical\s+)?families\b",
            re.IGNORECASE,
        ),
    ),
    (
        "non-glass-v1 scientific evidence",
        re.compile(
            r"\bnon[- ]glass(?:\s+unstable[- ]placement)?(?:\s+v1)?\b.{0,100}"
            r"\b(?:provides?|constitutes?|offers?|establishes?|supports?)\b.{0,30}"
            r"\b(?:scientific\s+)?evidence\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unsupported first-method novelty",
        re.compile(
            r"\bfirst\s+(?:selective\s+vla\s+intervention\s+method|"
            r"sequential\s+vla\s+verifier|hidden[- ]state\s+vla\s+failure\s+detector)\b",
            re.IGNORECASE,
        ),
    ),
)

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|neither|nor|cannot|can't|does\s+not|do\s+not|did\s+not|"
    r"is\s+not|are\s+not|was\s+not|were\s+not|without|fails?\s+to|"
    r"rather\s+than|doesn't|isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path, errors: list[str], label: str | None = None) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {label or path}: {exc}")
        return None


def _read_json(path: Path, errors: list[str], label: str) -> dict | None:
    text = _read_text(path, errors, label)
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {label}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} must contain a JSON object")
        return None
    return payload


def audit_pivot_manifest(repo_root: Path = ROOT) -> list[str]:
    """Validate the complete, immutable PIVOT-0 artifact set and its hashes."""

    errors: list[str] = []
    pivot_dir = repo_root / PIVOT_RELATIVE_DIR
    manifest_path = pivot_dir / "manifest.json"
    if not manifest_path.is_file():
        return [f"PIVOT manifest missing: {manifest_path.relative_to(repo_root)}"]
    manifest = _read_json(manifest_path, errors, "PIVOT manifest")
    if manifest is None:
        return errors

    expected_scalars = {
        "kind": "iclr27_risk_value_decoupling",
        "schema_version": 1,
        "evidence_package_fingerprint": "dc48ff317dad",
        "source_count": 20,
        "decision_count": 273,
        "condition_family_count": 3,
        "independent_mechanical_family_count": 1,
        "new_fresh_test_outcomes_collected": False,
        "new_rollouts": False,
        "new_training": False,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            errors.append(
                f"PIVOT manifest {key} expected {expected!r}, got {manifest.get(key)!r}"
            )

    required = manifest.get("required_artifacts")
    if not isinstance(required, list):
        errors.append("PIVOT manifest required_artifacts must be a list")
        required = []
    if tuple(required) != EXPECTED_PIVOT_REQUIRED_ARTIFACTS:
        errors.append(
            "PIVOT manifest required_artifacts changed: "
            f"expected {list(EXPECTED_PIVOT_REQUIRED_ARTIFACTS)!r}, got {required!r}"
        )

    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        errors.append("PIVOT manifest artifact_sha256 must be an object")
        hashes = {}

    for name in required:
        path = pivot_dir / name
        if not path.is_file():
            errors.append(f"PIVOT required artifact missing: {name}")
            continue
        if name == "manifest.json":
            # A manifest cannot contain its own stable byte hash.
            continue
        expected_hash = hashes.get(name)
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            errors.append(f"PIVOT artifact lacks valid SHA-256: {name}")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            errors.append(
                f"PIVOT artifact hash drift: {name} expected {expected_hash}, got {actual_hash}"
            )

    # Validate optional artifacts too when the manifest promises their bytes.
    for name, expected_hash in hashes.items():
        path = pivot_dir / name
        if not path.is_file():
            errors.append(f"PIVOT hashed artifact missing: {name}")
        elif not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            errors.append(f"PIVOT artifact has invalid SHA-256: {name}")
        elif file_sha256(path) != expected_hash:
            errors.append(f"PIVOT artifact hash drift: {name}")
    return errors


def package_text_paths(package_dir: Path) -> list[Path]:
    suffixes = {".tex", ".md", ".bib", ".csv"}
    if not package_dir.is_dir():
        return []
    return sorted(
        path for path in package_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def final_pdf_path(package_dir: Path, repo_root: Path | None = None) -> Path:
    """Resolve the repository output for the canonical package.

    Custom package directories keep their PDF fixture beneath the package so
    the audit helpers remain independently testable without writing to the
    repository output tree.
    """

    if repo_root is not None:
        canonical = (repo_root / "docs/iclr27/manuscript").resolve()
        if package_dir.resolve() == canonical:
            return repo_root.resolve() / FINAL_PDF_RELATIVE
    return package_dir / FINAL_PDF_RELATIVE


def required_file_errors(
    package_dir: Path,
    repo_root: Path | None = None,
) -> list[str]:
    errors = [
        f"required paper-package file missing: {relative}"
        for relative in REQUIRED_PACKAGE_FILES
        if not (package_dir / relative).is_file()
    ]
    pdf_path = final_pdf_path(package_dir, repo_root)
    if not pdf_path.is_file():
        errors.append(f"final paper PDF missing: {FINAL_PDF_RELATIVE.as_posix()}")
    return errors


def placeholder_errors(package_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in package_text_paths(package_dir):
        if path.name == AUDIT_RECORD_NAME:
            continue
        text = _read_text(path, errors, str(path.relative_to(package_dir)))
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = PLACEHOLDER_RE.search(line)
            if match:
                errors.append(
                    f"placeholder {match.group(0)!r} in "
                    f"{path.relative_to(package_dir)}:{line_number}"
                )
    return errors


def _strip_tex_comments(text: str) -> str:
    stripped: list[str] = []
    for line in text.splitlines():
        cut = len(line)
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        stripped.append(line[:cut])
    return "\n".join(stripped)


def _plain_tex(text: str) -> str:
    text = _strip_tex_comments(text)
    text = text.replace("\\_", "_").replace("~", " ")
    # Retain command arguments while removing common formatting commands.
    for _ in range(3):
        text = re.sub(
            r"\\(?:texttt|textbf|textit|emph|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}",
            r"\1",
            text,
        )
    text = re.sub(r"\\(?:cite|citep|citet|ref|label)\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("$", " ")
    return " ".join(text.split())


def extract_abstract(package_dir: Path) -> str | None:
    pattern = re.compile(
        r"\\begin\s*\{abstract\}(.*?)\\end\s*\{abstract\}",
        re.IGNORECASE | re.DOTALL,
    )
    for path in sorted(package_dir.rglob("*.tex")) if package_dir.is_dir() else []:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        match = pattern.search(text)
        if match:
            return _plain_tex(match.group(1))
    return None


def abstract_errors(package_dir: Path) -> list[str]:
    abstract = extract_abstract(package_dir)
    if abstract is None:
        return ["manuscript abstract environment is missing"]

    checks: tuple[tuple[str, str], ...] = (
        ("19/273 disagreement", r"\b19\s*/\s*273\b"),
        ("three-pair fact", r"\b(?:three|3)\b.{0,80}\bpairs?\b"),
        ("two-source witness limit", r"\b(?:two|2)\b.{0,50}\bsources?\b"),
        ("Risk -> Best Fixed utility 0.2544", r"\b0\.2544\b"),
        ("Oracle utility 0.5642", r"\b0\.5642\b"),
        ("oracle opportunity 0.3097", r"\b0\.3097\b"),
        ("OutcomeRouter +0.0178 gain", r"\+\s*0\.0178\b"),
        (
            "learned-method null",
            r"\b(?:null|no\s+(?:method\s+)?gate\s+passes?|did\s+not\s+pass.{0,30}gate|"
            r"failed.{0,30}gate)\b",
        ),
        ("23-state glass concentration", r"\b23\b.{0,100}\bglass\b"),
        ("51/60 glass concentration", r"\b51\s*/\s*60\b.{0,100}\bglass\b"),
        ("24/24 Base selections", r"\b24\s*/\s*24\b"),
        ("0/8 recovered glass episodes", r"\b0\s*/\s*8\b"),
        ("2/2 missed known opportunities", r"\b2\s*/\s*2\b"),
    )
    errors = [
        f"abstract lacks required fact: {label}"
        for label, pattern in checks
        if not re.search(pattern, abstract, flags=re.IGNORECASE | re.DOTALL)
    ]
    return errors


def _extract_captions(text: str) -> list[str]:
    text = _strip_tex_comments(text)
    starts = re.finditer(r"\\caption(?:\s*\[[^\]]*\])?\s*\{", text)
    captions: list[str] = []
    for match in starts:
        cursor = match.end()
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth:
            char = text[cursor]
            escaped = cursor > 0 and text[cursor - 1] == "\\"
            if not escaped and char == "{":
                depth += 1
            elif not escaped and char == "}":
                depth -= 1
                if depth == 0:
                    captions.append(text[content_start:cursor])
                    break
            cursor += 1
    return captions


def caption_errors(package_dir: Path) -> list[str]:
    captions: list[tuple[str, int, str]] = []
    errors: list[str] = []
    for path in sorted(package_dir.rglob("*.tex")) if package_dir.is_dir() else []:
        text = _read_text(path, errors, str(path.relative_to(package_dir)))
        if text is None:
            continue
        captions.extend(
            (str(path.relative_to(package_dir)), index, _plain_tex(caption))
            for index, caption in enumerate(_extract_captions(text), start=1)
        )
    if not captions:
        return [*errors, "paper package contains no figure/table captions"]

    for relative, index, caption in captions:
        lowered = caption.lower()
        count_token = (
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
            r"nineteen|twenty)"
        )
        explicit_count = bool(
            re.search(
                rf"\b{count_token}(?:\s+|[-‐-—])"
                r"(?:(?:effective|independent|eligible|supporting|"
                r"exposed[- ]development)\s+){0,3}"
                r"(?:source|source-state)s?\b",
                lowered,
            )
            or re.search(r"\bn\s*[_ ]?sources?\s*[=:]\s*\d+\b", lowered)
            or re.search(
                r"\b(?:effective\s+)?(?:independent\s+)?sources?\s*[=:]\s*\d+\b",
                lowered,
            )
            or (
                re.search(r"\b(?:supporting[- ]?)?source[- ]counts?\b", lowered)
                and re.search(r"\b\d+\b", lowered)
            )
        )
        if "effective source" not in lowered and not explicit_count:
            errors.append(
                f"caption lacks an effective source count: {relative} caption {index}"
            )

    family_caption_found = False
    for _, _, caption in captions:
        lowered = caption.lower()
        if (
            "glass_recovery" in lowered
            and re.search(r"\bglass\b", lowered)
            and re.search(r"\boffpath\b", lowered)
            and re.search(r"\bnoglass\b", lowered)
            and re.search(r"\b(?:single|one|1)\b.{0,40}\bfamily\b", lowered)
        ):
            family_caption_found = True
            break
    if not family_caption_found:
        errors.append(
            "no caption identifies one glass_recovery family and the glass/offpath/noglass conditions"
        )
    return errors


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(
        text.rfind(mark, 0, start) for mark in (".", "!", "?", ";", ":", "\n")
    )
    right_positions = [
        position for mark in (".", "!", "?", ";", ":", "\n")
        if (position := text.find(mark, end)) >= 0
    ]
    right = min(right_positions) if right_positions else len(text)
    return left + 1, right


def _in_explicit_limitation_column(text: str, start: int) -> bool:
    """Recognize a LaTeX claim-boundary table's explicitly negative column.

    A cell such as ``& A successful router`` is not affirmative prose when the
    table's column header is ``Not established``.  Requiring both a nearby
    negative header and an ampersand in the current row keeps this exception
    narrower than a global look-behind for the word ``not``.
    """

    row_start = text.rfind("\\\\", 0, start)
    row_prefix = text[row_start + 2:start] if row_start >= 0 else text[:start]
    if "&" not in row_prefix:
        return False
    header_window = text[max(0, row_start - 2500):row_start]
    return bool(
        re.search(
            r"\b(?:not\s+(?:established|supported|claimed)|unsupported(?:\s+wording)?)\b",
            header_window,
            flags=re.IGNORECASE,
        )
    )


def _explicitly_negated(clause: str, match_start: int, match_end: int) -> bool:
    """Require negation to govern the dangerous phrase, not an earlier claim."""

    matched_text = clause[match_start:match_end]
    if NEGATION_RE.search(matched_text):
        return True
    prefix = clause[:match_start]
    # An adversative conjunction starts a new assertion scope.  Thus
    # ``not X, but a successful router`` remains an error, while
    # ``not a successful router or a broad benchmark`` remains safely scoped.
    adversatives = list(
        re.finditer(
            r"\b(?:but|however|yet|nevertheless|nonetheless|whereas)\b",
            prefix,
            flags=re.IGNORECASE,
        )
    )
    if adversatives:
        prefix = prefix[adversatives[-1].end():]
    # A leading concessive or context-setting limitation does not negate the
    # assertion after its comma: ``Without new experiments, this is ...``.
    last_comma = prefix.rfind(",")
    if last_comma >= 0 and re.search(
        r"\b(?:although|though|while|despite|without|even\s+though)\b",
        prefix[:last_comma],
        flags=re.IGNORECASE,
    ):
        prefix = prefix[last_comma + 1:]
    return bool(NEGATION_RE.search(prefix[-180:]))


def affirmative_claim_errors(text: str, source: str = "<text>") -> list[str]:
    """Return dangerous affirmative claims while allowing explicit limitations."""

    plain = _plain_tex(text).replace("not only", "affirmatively")
    errors: list[str] = []
    for label, pattern in DANGEROUS_CLAIM_PATTERNS:
        for match in pattern.finditer(plain):
            clause_start, clause_end = _clause_bounds(
                plain, match.start(), match.end()
            )
            clause = plain[clause_start:clause_end]
            explicitly_limited = (
                _explicitly_negated(
                    clause,
                    match.start() - clause_start,
                    match.end() - clause_start,
                )
                or _in_explicit_limitation_column(plain, match.start())
            )
            if explicitly_limited:
                continue
            excerpt = " ".join(clause.split())[:180]
            errors.append(f"dangerous affirmative claim ({label}) in {source}: {excerpt}")
    return errors


def prohibited_claim_errors(package_dir: Path) -> list[str]:
    errors: list[str] = []
    if not package_dir.is_dir():
        return errors
    for path in sorted(
        candidate for candidate in package_dir.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in {".tex", ".md"}
        and candidate.name != AUDIT_RECORD_NAME
    ):
        relative = str(path.relative_to(package_dir))
        text = _read_text(path, errors, relative)
        if text is not None:
            errors.extend(affirmative_claim_errors(text, relative))
    return errors


def artifact_map_errors(package_dir: Path, repo_root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    path = package_dir / "artifact_map.csv"
    if not path.is_file():
        return ["artifact_map.csv is missing"]
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return [f"cannot parse artifact_map.csv: {exc}"]

    if fieldnames != ARTIFACT_MAP_COLUMNS:
        errors.append(
            "artifact_map.csv columns changed: "
            f"expected {list(ARTIFACT_MAP_COLUMNS)!r}, got {list(fieldnames)!r}"
        )
        return errors

    by_key: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = (row.get("claim_key") or "").strip()
        if not key:
            errors.append(f"artifact_map.csv row {row_number} lacks claim_key")
            continue
        if key in by_key:
            errors.append(f"artifact_map.csv duplicates claim_key {key!r}")
        by_key[key] = row
        for column in ARTIFACT_MAP_COLUMNS:
            if not (row.get(column) or "").strip():
                errors.append(f"artifact_map.csv {key!r} has empty {column}")

    expected_keys = set(EXPECTED_ARTIFACT_MAP)
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        errors.append(
            "artifact_map.csv claim keys changed: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    for key, (expected_path, expected_hash, display_tokens) in EXPECTED_ARTIFACT_MAP.items():
        row = by_key.get(key)
        if row is None:
            continue
        canonical = row["canonical_artifact"].strip()
        declared_hash = row["artifact_sha256"].strip()
        if canonical != expected_path:
            errors.append(
                f"artifact_map.csv {key!r} canonical path expected {expected_path!r}, "
                f"got {canonical!r}"
            )
        if declared_hash != expected_hash:
            errors.append(
                f"artifact_map.csv {key!r} SHA-256 expected {expected_hash}, "
                f"got {declared_hash}"
            )
        displayed = row["displayed_value"]
        for token in display_tokens:
            if token not in displayed:
                errors.append(
                    f"artifact_map.csv {key!r} displayed_value lacks {token!r}"
                )
        expected_commits = EXPECTED_ARTIFACT_COMMITS[key]
        for commit_column, expected_commit in zip(
            ("generation_commit", "record_commit"), expected_commits
        ):
            commit = row[commit_column].strip()
            if not COMMIT_RE.fullmatch(commit):
                errors.append(
                    f"artifact_map.csv {key!r} {commit_column} is not a full commit hash"
                )
            elif commit != expected_commit:
                errors.append(
                    f"artifact_map.csv {key!r} {commit_column} expected "
                    f"{expected_commit}, got {commit}"
                )

        if row["independent_unit"].strip() != "source_state_sha256":
            errors.append(
                f"artifact_map.csv {key!r} independent_unit must be source_state_sha256"
            )
        effective_sources = row["effective_sources"].lower()
        if "source" not in effective_sources:
            errors.append(
                f"artifact_map.csv {key!r} effective_sources lacks a source unit"
            )
        for token in EXPECTED_EFFECTIVE_SOURCE_TOKENS[key]:
            if token not in effective_sources:
                errors.append(
                    f"artifact_map.csv {key!r} effective_sources lacks {token!r}"
                )
        evidence_scope = row["evidence_scope"]
        for token in EXPECTED_SCOPE_TOKENS[key]:
            if token.lower() not in evidence_scope.lower():
                errors.append(
                    f"artifact_map.csv {key!r} evidence_scope lacks {token!r}"
                )
        expected_availability = EXPECTED_REPOSITORY_AVAILABILITY[key]
        if row["repository_availability"].strip() != expected_availability:
            errors.append(
                f"artifact_map.csv {key!r} repository_availability expected "
                f"{expected_availability!r}, got {row['repository_availability'].strip()!r}"
            )

        artifact_path = repo_root / expected_path
        if not artifact_path.is_file():
            errors.append(f"artifact_map.csv {key!r} canonical artifact is missing: {expected_path}")
        else:
            actual_hash = file_sha256(artifact_path)
            if actual_hash != expected_hash:
                errors.append(
                    f"canonical artifact hash drift for {key!r}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
    return errors


def _pdf_page_count(path: Path) -> tuple[int | None, str | None]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        PdfReader = None

    if PdfReader is not None:
        try:
            return len(PdfReader(str(path)).pages), None
        except Exception as exc:  # pragma: no cover - depends on optional parser
            return None, f"pypdf could not parse final PDF: {exc}"

    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        try:
            completed = subprocess.run(
                [pdfinfo, str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"pdfinfo could not inspect final PDF: {exc}"
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            return None, f"pdfinfo rejected final PDF: {detail}"
        match = re.search(r"(?m)^Pages:\s*(\d+)\s*$", completed.stdout)
        if not match:
            return None, "pdfinfo output lacks a Pages field"
        return int(match.group(1)), None

    # Last-resort structural fallback for minimal environments.  The word
    # boundary excludes the /Pages tree object.
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"cannot read final PDF: {exc}"
    return len(re.findall(rb"/Type\s*/Page\b", data)), None


def pdf_errors(package_dir: Path, repo_root: Path | None = None) -> list[str]:
    path = final_pdf_path(package_dir, repo_root)
    if not path.is_file():
        return [f"final paper PDF missing: {FINAL_PDF_RELATIVE.as_posix()}"]
    try:
        header = path.read_bytes()[:5]
    except OSError as exc:
        return [f"cannot read final paper PDF: {exc}"]
    if header != b"%PDF-":
        return ["final paper PDF does not start with %PDF-"]
    page_count, error = _pdf_page_count(path)
    if error:
        return [error]
    if page_count is None or page_count <= 0:
        return [f"final paper PDF has invalid page count: {page_count!r}"]
    return []


def clean_worktree_errors(repo_root: Path = ROOT) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"cannot run git status --porcelain: {exc}"]
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return [f"git status --porcelain failed: {detail}"]
    if completed.stdout.strip():
        paths = [line for line in completed.stdout.splitlines() if line.strip()]
        return ["git worktree is not clean: " + "; ".join(paths)]
    return []


def audit_paper_package(
    package_dir: Path = DEFAULT_PACKAGE_DIR,
    *,
    repo_root: Path = ROOT,
    require_clean: bool = False,
) -> list[str]:
    """Return every paper-package audit error without mutating the repository."""

    package_dir = package_dir.resolve()
    repo_root = repo_root.resolve()
    if not package_dir.is_dir():
        return [f"paper-package directory missing: {package_dir}"]

    errors: list[str] = []
    errors.extend(required_file_errors(package_dir, repo_root))
    errors.extend(audit_pivot_manifest(repo_root))
    errors.extend(placeholder_errors(package_dir))
    errors.extend(abstract_errors(package_dir))
    errors.extend(caption_errors(package_dir))
    errors.extend(prohibited_claim_errors(package_dir))
    errors.extend(artifact_map_errors(package_dir, repo_root))
    if final_pdf_path(package_dir, repo_root).is_file():
        errors.extend(pdf_errors(package_dir, repo_root))
    if require_clean:
        errors.extend(clean_worktree_errors(repo_root))
    return errors


def _format_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=DEFAULT_PACKAGE_DIR,
        help="Paper-package directory (default: docs/iclr27/manuscript).",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Also require `git status --porcelain` to be empty.",
    )
    args = parser.parse_args(argv)

    package_dir = args.package_dir
    if not package_dir.is_absolute():
        package_dir = ROOT / package_dir
    errors = audit_paper_package(
        package_dir,
        repo_root=ROOT,
        require_clean=args.require_clean,
    )
    if errors:
        print("Paper-package audit FAILED:")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print(
        "Paper-package audit passed: required files, frozen PIVOT hashes, abstract "
        "facts, source-aware captions, scoped claims, artifact map, and final PDF "
        f"are consistent ({_format_path(package_dir, ROOT)})."
    )


if __name__ == "__main__":
    main()
