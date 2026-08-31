from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/expansion/finalize_pinned_d8_postprocess.py"


def test_pinned_postprocess_finalizer_requires_seal_and_preserves_null_scope():
    source = SCRIPT.read_text()
    assert "completion seal does not pin the raw audit" in source
    assert '"method_superiority_evaluated": False' in source
    assert '"test_reopen_permitted": False' in source
    assert "refusing to overwrite" in source
