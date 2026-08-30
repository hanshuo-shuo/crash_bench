from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_benchmark_test.py"


def test_confirmatory_benchmark_analysis_is_method_closed_and_nonadaptive():
    source = SCRIPT.read_text()
    assert '"method_superiority_evaluated": False' in source
    assert "confirmatory=True" in source
    assert "allow_scoped_continuation=False" in source
    assert "TEST_SCOPE_FAILURE" in source
    assert "B0>=3, B1>=8" in source
