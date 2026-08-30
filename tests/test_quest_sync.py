from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/quest_sync.sh"


def test_packed_result_transfer_is_guarded_and_disables_recompression():
    source = SCRIPT.read_text()
    block = source[source.index("  pull-packed-result)"):source.index("  -h|--help|help|'')")]
    assert "^results/" in block
    assert "tar\\.gz|tar\\.zst" in block
    assert "test -f" in block
    assert "--append-verify" in block
    assert "rsync -a " in block
    assert "rsync -az" not in block
