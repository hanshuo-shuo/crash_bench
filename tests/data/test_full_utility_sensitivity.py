from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/analyze_full_utility_sensitivity.py"


def test_full_grid_is_explicitly_supplemental_and_non_gating():
    source = SCRIPT.read_text()
    assert "len(rows) != 108" in source
    assert '"analysis_role": "SUPPLEMENTAL_NON_GATING_PREDECLARED_WEIGHT_GRID"' in source
    assert '"confirmatory_gate_changed": False' in source
    assert '"method_ranking_claim": False' in source
