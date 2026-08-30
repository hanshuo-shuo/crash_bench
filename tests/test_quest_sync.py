from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/quest_sync.sh"


def test_packed_result_transfer_is_guarded_and_disables_recompression():
    source = SCRIPT.read_text()
    block = source[source.index("  pull-packed-result)"):source.index("  push-d8-freeze)")]
    assert "^results/" in block
    assert "tar\\.gz|tar\\.xz|tar\\.zst" in block
    assert "test -f" in block
    assert "--partial --append" in block
    assert "independently pulled SHA-256 sidecar" in block
    assert "rsync -a " in block
    assert "rsync -az" not in block


def test_d8_freeze_push_is_new_destination_only_and_checksum_verified():
    source = SCRIPT.read_text()
    block = source[source.index("  push-d8-freeze)"):source.index("  -h|--help|help|'')")]
    assert "^results/expansion/d8_benchmark/" in block
    assert "test ! -e" in block
    assert "test_open.lock" in block
    assert "rsync -azcn" in block
    assert "--delete" not in block
