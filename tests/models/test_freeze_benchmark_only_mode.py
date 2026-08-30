from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/freeze_benchmark_only_mode.py"


def test_benchmark_freeze_keeps_test_closed_and_method_unauthorized():
    source = SCRIPT.read_text()
    assert '"method_claim_authorized": False' in source
    assert '"statewise_method_test_authorized": False' in source
    assert '"benchmark_outcome_collection_authorized": False' in source
    assert '"test_opened": False' in source
    assert "exactly 32 frozen physical test sources" in source
