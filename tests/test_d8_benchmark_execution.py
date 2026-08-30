from pathlib import Path


FREEZER = Path(__file__).resolve().parents[1] / "scripts/expansion/freeze_d8_benchmark_execution.py"
SBATCH = Path(__file__).resolve().parents[1] / "setup/expansion_benchmark_test.sbatch"
WRAPPER = Path(__file__).resolve().parents[1] / "setup/submit_expansion_benchmark_test.sh"


def test_d8_execution_freezes_all_claim_bearing_code_and_method_stays_closed():
    source = FREEZER.read_text()
    for filename in (
        "collect_staleness_statewise.py", "audit_benchmark_test_shards.py",
        "merge_benchmark_test_dataset.py", "analyze_benchmark_test.py",
        "expansion_benchmark_test.sbatch",
    ):
        assert filename in source
    assert '"method_superiority_test_authorized": False' in source
    assert '"test_outcomes_read": 0' in source


def test_d8_slurm_contract_is_one_array_on_authorized_paths():
    sbatch = SBATCH.read_text()
    wrapper = WRAPPER.read_text()
    assert "#SBATCH --account=p33100" in sbatch
    assert "#SBATCH --partition=gengpu" in sbatch
    assert "--mode benchmark_test" in sbatch
    assert "--array=0-31" in wrapper
    assert "test_open.lock" in wrapper and "test_complete.seal" in wrapper


def test_d8_postprocessors_reseal_hashes_after_authorization_metadata():
    root = Path(__file__).resolve().parents[1] / "scripts/expansion"
    assert 'manifest.pop("manifest_sha256", None)' in (root / "merge_benchmark_test_dataset.py").read_text()
    assert 'payload.pop("audit_sha256", None)' in (root / "audit_benchmark_test_shards.py").read_text()
    assert 'gate.pop("decision_sha256", None)' in (root / "analyze_benchmark_test.py").read_text()
