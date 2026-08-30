from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/expansion/write_scoped_benchmark_manuscript.py"


def test_manuscript_generator_requires_release_audit_and_preserves_claim_scope():
    source = SCRIPT.read_text()
    assert 'audit["status"] != "GO"' in source
    assert "No method superiority test" in source
    assert "single-policy, single-mechanism" in source
    assert "discovered after" in source
    assert 'if "TBD" in text or "TODO" in text' in source
