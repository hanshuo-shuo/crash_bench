from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/expansion/audit_scoped_benchmark_release.py"


def test_release_audit_enforces_hashes_claims_effective_n_and_no_reopen():
    source = SCRIPT.read_text()
    assert "release_manifest_self_hash_mismatch" in source
    assert "artifact_identity_mismatch" in source
    assert "test_reopen_permitted" in source
    assert "No learned-method superiority claim is authorized." in source
    assert '"D5_train_development": 36' in source
    assert "FIX_GENERATION_OR_TEXT_ONLY__DO_NOT_CHANGE_EXPERIMENT" in source
    assert "utility_sensitivity_grid_incomplete" in source
