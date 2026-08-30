from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/run_narrow_clearance_screen.py"
SPEC = importlib.util.spec_from_file_location("run_narrow_clearance_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_narrow_runner_uses_three_deployable_options():
    assert MODULE.OPTIONS == ("base_continue", "backtrack_requery", "safe_stop")
