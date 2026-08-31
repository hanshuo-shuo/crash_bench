from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/expansion/render_scoped_benchmark_release.py"


def test_scoped_figure_has_effective_n_roles_and_claim_boundary():
    source = SCRIPT.read_text()
    assert "D5 train+development (n=36)" in source
    assert "D8 confirmatory test (n=32)" in source
    assert '"method_superiority_depicted": False' in source
    assert "all 12 calibration sources excluded" in source
    assert "single-policy, single-staleness" in source
    assert "matplotlib" not in source
    assert "figure_scoped_support.svg" in source
    assert "left, right, top, bottom = 80, 100, 70, 105" in source
