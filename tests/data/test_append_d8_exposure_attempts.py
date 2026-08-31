from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/append_d8_exposure_attempts.py"


def test_d8_exposure_append_is_sealed_noneligible_and_append_only():
    source = SCRIPT.read_text()
    assert "exactly 32 physical-source shards" in source
    assert "crashbench_test_complete_seal" in source
    assert '"test_eligible": False' in source
    assert '"option_outcomes_opened": True' in source
    assert "os.O_APPEND" in source
    assert "already exists" in source
