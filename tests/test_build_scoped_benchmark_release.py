from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/expansion/build_scoped_benchmark_release.py"


def test_release_builder_enforces_sealed_test_and_claim_boundaries():
    source = SCRIPT.read_text()
    assert "test_complete.seal" in source
    assert '"method_claim_authorized": False' in source
    assert '"sequential_claim_authorized": False' in source
    assert '"multi_mechanism_claim_authorized": False' in source
    assert "SCOPED_TEST_SCOPE_FAILURE_RELEASE" in source
    assert "gate_a_correction_disclosed" in source
