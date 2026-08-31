from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scoped_release_builder_and_audit_end_to_end(tmp_path):
    d5 = write(tmp_path / "d5.json", {"status": "GO", "complete_blocks": 1296})
    gate_a = write(
        tmp_path / "gate_a.json",
        {
            "gate": {"status": "SCOPED_CONTINUE", "hard_failures": []},
            "calibration_rows_read": 0,
            "calibration_sources_excluded": 12,
            "source_count": 36,
            "block_decision_count": 972,
            "benefit_zero_sources": 4,
            "benefit_one_sources": 32,
            "strict_support_sources": {"observation_refresh": 20, "safe_stop": 21},
            "same_risk_flip_sources": 29,
        },
    )
    correction = write(
        tmp_path / "correction.json", {"continue_same_authorized_d8_run": True}
    )
    d6 = write(tmp_path / "d6.json", {"test_rows_read": 0})
    gate_b = write(tmp_path / "gate_b.json", {"test_rows_read": 0, "gate": {"status": "SCOPED_CONTINUE"}})
    benchmark = write(tmp_path / "benchmark.json", {"method_claim_authorized": False})
    d8_root = tmp_path / "d8"
    seal = write(d8_root / "authorization" / "test_complete.seal", {"kind": "seal"})
    del seal
    write(
        d8_root / "postprocess" / "run_manifest.json",
        {"test_complete_sealed": True, "physical_source_shards": 32, "test_sources_read": 32},
    )
    d8_analysis = write(
        d8_root / "postprocess" / "benchmark_test_analysis.json",
        {
            "gate": {"status": "GO"},
            "method_superiority_evaluated": False,
            "mechanical_invalid_rate": 0.0,
            "primary": {
                "source_count": 32,
                "block_decision_count": 864,
                "benefit_zero_sources": 3,
                "benefit_one_sources": 29,
                "strict_support_sources": {"observation_refresh": 14, "safe_stop": 17},
                "same_risk_flip_sources": 20,
            },
        },
    )
    figure_root = tmp_path / "figures"
    figure_root.mkdir()
    png = figure_root / "figure_scoped_support.png"
    pdf = figure_root / "figure_scoped_support.pdf"
    caption = figure_root / "figure_scoped_support_caption.md"
    png.write_bytes(b"png")
    pdf.write_bytes(b"pdf")
    caption.write_text("effective n=36 and effective n=32\n")
    figure_manifest = write(
        figure_root / "figure_manifest.json",
        {
            "effective_n": {"D5_train_development": 36, "D8_confirmatory_test": 32},
            "method_superiority_depicted": False,
            "artifacts": {
                path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                for path in (png, pdf, caption)
            },
        },
    )
    sensitivity = write(
        tmp_path / "sensitivity.json",
        {
            "weight_setting_count": 108,
            "confirmatory_gate_changed": False,
            "analysis_role": "SUPPLEMENTAL_NON_GATING_PREDECLARED_WEIGHT_GRID",
        },
    )
    release = tmp_path / "release"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/expansion/build_scoped_benchmark_release.py"),
            "--d5-raw-audit", str(d5),
            "--corrected-gate-a", str(gate_a),
            "--gate-a-correction-audit", str(correction),
            "--d6-run-manifest", str(d6),
            "--gate-b", str(gate_b),
            "--benchmark-freeze", str(benchmark),
            "--d8-run-root", str(d8_root),
            "--d8-postprocess-dir", str(d8_root / "postprocess"),
            "--figure-manifest", str(figure_manifest),
            "--full-sensitivity", str(sensitivity),
            "--output-dir", str(release),
        ],
        cwd=ROOT,
        check=True,
    )
    audit = tmp_path / "release_audit.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/expansion/audit_scoped_benchmark_release.py"),
            "--release-root", str(release),
            "--output", str(audit),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(audit.read_text())
    assert payload["status"] == "GO"
    assert payload["errors"] == []
    assert json.loads((release / "release_manifest.json").read_text())[
        "single_mechanism_benchmark_claim_authorized"
    ] is True
