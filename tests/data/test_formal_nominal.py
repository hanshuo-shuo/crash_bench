from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/expansion/run_formal_nominal.py"
SPEC = importlib.util.spec_from_file_location("run_formal_nominal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_six_shards_cover_two_tasks_by_63_candidates():
    specs = [MODULE.shard_spec(index) for index in range(6)]
    assert specs == [(0, 0, 21), (0, 21, 42), (0, 42, 63), (2, 0, 21), (2, 21, 42), (2, 42, 63)]
    with pytest.raises(ValueError):
        MODULE.shard_spec(6)


def test_scene_fingerprint_uses_object_positions_not_robot_state():
    first = MODULE.scene_fingerprint(
        {"bowl_pos": np.array([0.1, 0.2, 0.3]), "robot0_eef_pos": np.array([1, 2, 3])}
    )
    second = MODULE.scene_fingerprint(
        {"bowl_pos": np.array([0.1, 0.2, 0.3]), "robot0_eef_pos": np.array([9, 9, 9])}
    )
    changed = MODULE.scene_fingerprint(
        {"bowl_pos": np.array([0.11, 0.2, 0.3]), "robot0_eef_pos": np.array([1, 2, 3])}
    )
    assert first == second
    assert first != changed
