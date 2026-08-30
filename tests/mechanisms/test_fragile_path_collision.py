from __future__ import annotations

import numpy as np
import pytest

from crashbench.mechanisms.fragile_path_collision import (
    derive_fragile_injections,
    fragile_catastrophe,
    validate_fragile_pre_outcome,
)


PATH = np.array([[0, 0, 0.2], [0.2, 0, 0.2], [0.4, 0, 0.2]])


def test_fragile_hazard_is_on_path_and_control_is_offset():
    rows = derive_fragile_injections(
        PATH, path_fraction=0.5, table_z=0, radius_m=0.02,
        half_height_m=0.05, offpath_offset_m=0.10
    )
    on = rows["on_path"]
    off = rows["off_path_control"]
    assert on.position[:2] == pytest.approx((0.2, 0.0))
    assert off.position[:2] == pytest.approx((0.2, 0.1))
    assert rows["no_object_control"] is None
    assert on.movable_object()["type"] == "cylinder"
    assert validate_fragile_pre_outcome(PATH, on).valid


def test_offpath_control_must_clear_diameter():
    with pytest.raises(ValueError, match="diameter"):
        derive_fragile_injections(
            PATH, path_fraction=0.5, table_z=0, radius_m=0.03,
            half_height_m=0.05, offpath_offset_m=0.05
        )


def test_fragile_catastrophe_uses_displacement_or_tilt():
    assert fragile_catastrophe(
        initial_xy=(0, 0), current_xy=(0.03, 0), current_tilt_deg=0,
        displacement_threshold_m=0.02, tilt_threshold_deg=30
    )
    assert fragile_catastrophe(
        initial_xy=(0, 0), current_xy=(0, 0), current_tilt_deg=45,
        displacement_threshold_m=0.02, tilt_threshold_deg=30
    )
    assert not fragile_catastrophe(
        initial_xy=(0, 0), current_xy=(0.001, 0), current_tilt_deg=2,
        displacement_threshold_m=0.02, tilt_threshold_deg=30
    )
