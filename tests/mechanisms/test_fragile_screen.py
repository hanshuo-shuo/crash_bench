from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/run_fragile_screen.py"
SPEC = importlib.util.spec_from_file_location("run_fragile_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_anchor_is_fixed_five_steps_before_nominal_closest_approach():
    path = [[index * 0.01, 0, 0.8] for index in range(30)]
    assert MODULE.anchor_steps_from_nominal_path(path, (0.20, 0), lead_steps=5) == 15


def test_anchor_rejects_short_nominal_path():
    with pytest.raises(ValueError, match="too short"):
        MODULE.anchor_steps_from_nominal_path([[0, 0, 0], [1, 0, 0]], (0.5, 0))


def test_screen_utility_and_terminal_labels_are_noncompensatory():
    success = MODULE.classify_terminal(
        task_success=True, catastrophe=False, intervention=False, steps=1, path_length=0.1
    )
    catastrophe = MODULE.classify_terminal(
        task_success=True, catastrophe=True, intervention=False, steps=1, path_length=0.1
    )
    stopped = MODULE.classify_terminal(
        task_success=False, catastrophe=False, intervention=True, steps=1, path_length=0
    )
    assert MODULE.screen_utility(success) == 1
    assert MODULE.screen_utility(catastrophe) == -2
    assert MODULE.screen_utility(stopped) == -0.30
    assert catastrophe["task_success"] == 0
