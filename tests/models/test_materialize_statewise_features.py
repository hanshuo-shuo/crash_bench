from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/materialize_statewise_features.py"


def test_materialized_cache_is_content_pinned_and_test_closed():
    source = SCRIPT.read_text()
    assert "store.validate(ref)" in source
    assert '"features_npz_sha256"' in source
    assert '"test_rows_read": 0' in source
    assert '"test" in str(row["split_role"])' in source
