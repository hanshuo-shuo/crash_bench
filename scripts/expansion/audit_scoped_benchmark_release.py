#!/usr/bin/env python3
"""Fail-closed D10 hash, artifact, test-lock, and claim-boundary audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D10 release audit: {args.output}")
    manifest_path = args.release_root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.get("release_sha256")
    unhashed = dict(manifest)
    unhashed.pop("release_sha256", None)
    errors = []
    if canonical_sha(unhashed) != declared:
        errors.append("release_manifest_self_hash_mismatch")
    for name, artifact in manifest.get("artifacts", {}).items():
        path = Path(artifact["path"])
        if not path.is_file():
            errors.append(f"missing_artifact:{name}")
            continue
        if sha256_file(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            errors.append(f"artifact_identity_mismatch:{name}")
    if manifest.get("unfinished_required_artifacts") != []:
        errors.append("unfinished_required_artifacts")
    for key in (
        "method_claim_authorized", "sequential_claim_authorized",
        "multi_mechanism_claim_authorized", "test_reopen_permitted",
    ):
        if manifest.get(key) is not False:
            errors.append(f"forbidden_claim_or_reopen:{key}")
    if manifest.get("test_sources_read") != 32:
        errors.append("confirmatory_source_count_mismatch")
    if manifest.get("gate_a_correction_disclosed") is not True:
        errors.append("gate_a_correction_not_disclosed")
    claims = (args.release_root / "CLAIMS.md").read_text()
    required_claim_boundaries = (
        "No multi-mechanism benchmark claim is authorized.",
        "No learned-method superiority claim is authorized.",
        "No sequential safeguard, online recovery, or deployability claim is authorized.",
    )
    for sentence in required_claim_boundaries:
        if sentence not in claims:
            errors.append(f"missing_claim_boundary:{sentence}")
    figure_manifest_path = Path(manifest["artifacts"]["figure_manifest"]["path"])
    figure_manifest = json.loads(figure_manifest_path.read_text())
    if figure_manifest.get("effective_n") != {"D5_train_development": 36, "D8_confirmatory_test": 32}:
        errors.append("figure_effective_n_mismatch")
    if figure_manifest.get("method_superiority_depicted") is not False:
        errors.append("figure_depicts_forbidden_method_claim")
    sensitivity_path = Path(manifest["artifacts"]["full_utility_sensitivity"]["path"])
    sensitivity = json.loads(sensitivity_path.read_text())
    if sensitivity.get("weight_setting_count") != 108:
        errors.append("utility_sensitivity_grid_incomplete")
    if sensitivity.get("confirmatory_gate_changed") is not False:
        errors.append("supplemental_sensitivity_changed_confirmatory_gate")
    caption_path = figure_manifest_path.parent / "figure_scoped_support_caption.md"
    caption = caption_path.read_text() if caption_path.is_file() else ""
    if "effective n=36" not in caption or "effective n=32" not in caption:
        errors.append("figure_caption_lacks_effective_n")
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_scoped_release_audit",
        "release_manifest_sha256": sha256_file(manifest_path),
        "artifact_count": len(manifest.get("artifacts", {})),
        "errors": errors,
        "source_or_test_leakage_findings": [],
        "claim_boundary_violations": [error for error in errors if "claim" in error],
        "status": "GO" if not errors else "NO_GO",
        "next_action": "FREEZE_RELEASE_PACKAGE" if not errors else "FIX_GENERATION_OR_TEXT_ONLY__DO_NOT_CHANGE_EXPERIMENT",
    }
    payload["audit_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "errors": len(errors)}, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
