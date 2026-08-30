from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from crashbench.mechanisms.action_drift import ActionDriftInjector


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/run_action_drift_screen.py"
SPEC = importlib.util.spec_from_file_location("run_action_drift_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_injector_state_hash_is_stable_and_progress_sensitive():
    injector = ActionDriftInjector(
        np.array([0.1, 0, 0, 0, 0, 0, 0]),
        action_low=np.full(7, -1), action_high=np.full(7, 1)
    )
    first = MODULE.injector_state_sha256(injector)
    assert first == MODULE.injector_state_sha256(injector)
    injector.apply(np.zeros(7))
    assert first != MODULE.injector_state_sha256(injector)
