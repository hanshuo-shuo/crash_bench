from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/expansion/postprocess_d8_benchmark.py"


def test_d8_postprocess_is_complete_case_then_seals_once():
    source = SCRIPT.read_text()
    assert "requires 32 shards" in source
    assert source.index("audit_benchmark_test_shards.py") < source.index("merge_benchmark_test_dataset.py")
    assert source.index("merge_benchmark_test_dataset.py") < source.index("analyze_benchmark_test.py")
    assert source.index("analyze_benchmark_test.py") < source.rindex("seal_test_complete")
    assert '"method_superiority_evaluated": False' in source
