from __future__ import annotations

import numpy as np
import pytest

from crashbench.mechanisms.narrow_clearance import (
    derive_corridor_from_path,
    validate_corridor_pre_outcome,
)


PATH = np.array([[0, 0, 0.2], [0.2, 0, 0.2], [0.4, 0, 0.2]])


def corridor(gap=0.12):
    return derive_corridor_from_path(
        PATH, path_fraction=0.5, gap_m=gap, length_m=0.2,
        wall_thickness_m=0.02, wall_height_m=0.15, table_z=0
    )


def test_corridor_walls_are_symmetric_around_nominal_path():
    geometry = corridor()
    walls = geometry.obstacles()
    assert len(walls) == 2
    assert walls[0]["pos"][0] == pytest.approx(walls[1]["pos"][0])
    assert walls[0]["pos"][1] == pytest.approx(-walls[1]["pos"][1])
    assert walls[0]["size"] == walls[1]["size"]


def test_corridor_validity_separates_impossible_narrow_and_wide():
    valid = validate_corridor_pre_outcome(
        corridor(0.12), robot_envelope_width_m=0.10,
        maximum_extra_clearance_m=0.03, wide_control_gap_m=0.20
    )
    assert valid.valid
    impossible = validate_corridor_pre_outcome(
        corridor(0.09), robot_envelope_width_m=0.10,
        maximum_extra_clearance_m=0.03, wide_control_gap_m=0.20
    )
    assert not impossible.valid
    too_wide = validate_corridor_pre_outcome(
        corridor(0.20), robot_envelope_width_m=0.10,
        maximum_extra_clearance_m=0.03, wide_control_gap_m=0.30
    )
    assert not too_wide.valid
