"""Narrow-clearance corridor geometry with widened and no-wall controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import MechanicalValidity, MechanismSpec


NARROW_CLEARANCE_SPEC = MechanismSpec(
    mechanism_id="narrow_clearance",
    version=1,
    task_ids=("libero_spatial:0", "libero_spatial:2"),
    conditions=("narrow", "wide_control", "no_wall_control"),
    hazard_condition="narrow",
    matched_control_conditions=("wide_control", "no_wall_control"),
    severity_ids=("clearance_005", "clearance_010", "clearance_015"),
    deployable_option_ids=("base_continue", "backtrack_requery", "safe_stop"),
    diagnostic_option_ids=("oracle_corridor_path",),
    information_contract_version=1,
)


@dataclass(frozen=True)
class CorridorGeometry:
    center_xy: tuple[float, float]
    yaw_rad: float
    length_m: float
    gap_m: float
    wall_thickness_m: float
    wall_height_m: float
    table_z: float

    def obstacles(self, *, prefix: str = "cb_clearance") -> list[dict]:
        tangent = np.array([np.cos(self.yaw_rad), np.sin(self.yaw_rad)])
        normal = np.array([-tangent[1], tangent[0]])
        center = np.asarray(self.center_xy)
        offset = self.gap_m / 2 + self.wall_thickness_m / 2
        rows = []
        for side, sign in (("left", 1.0), ("right", -1.0)):
            xy = center + sign * offset * normal
            rows.append(
                {
                    "name": f"{prefix}_{side}",
                    "pos": [float(xy[0]), float(xy[1]), self.table_z + self.wall_height_m / 2],
                    "size": [self.length_m / 2, self.wall_thickness_m / 2, self.wall_height_m / 2],
                    "type": "box",
                    "euler": [0, 0, float(self.yaw_rad)],
                    "rgba": [0.72, 0.42, 0.16, 1.0],
                }
            )
        return rows


def derive_corridor_from_path(
    nominal_eef_path_xyz: Sequence[Sequence[float]],
    *,
    path_fraction: float,
    gap_m: float,
    length_m: float,
    wall_thickness_m: float,
    wall_height_m: float,
    table_z: float,
) -> CorridorGeometry:
    path = np.asarray(nominal_eef_path_xyz, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError("nominal path must be Nx3 with N>=2")
    if not 0.2 <= path_fraction <= 0.8:
        raise ValueError("corridor path fraction must be central and frozen")
    segment_index = min(int(path_fraction * (len(path) - 1)), len(path) - 2)
    delta = path[segment_index + 1, :2] - path[segment_index, :2]
    if np.linalg.norm(delta) <= 0:
        raise ValueError("corridor anchor path segment has zero length")
    center = (path[segment_index, :2] + path[segment_index + 1, :2]) / 2
    yaw = float(np.arctan2(delta[1], delta[0]))
    if min(gap_m, length_m, wall_thickness_m, wall_height_m) <= 0:
        raise ValueError("corridor dimensions must be positive")
    return CorridorGeometry(
        (float(center[0]), float(center[1])), yaw, length_m, gap_m,
        wall_thickness_m, wall_height_m, table_z
    )


def validate_corridor_pre_outcome(
    corridor: CorridorGeometry,
    *,
    robot_envelope_width_m: float,
    maximum_extra_clearance_m: float,
    wide_control_gap_m: float,
) -> MechanicalValidity:
    extra = corridor.gap_m - robot_envelope_width_m
    metrics = {
        "gap_m": corridor.gap_m,
        "robot_envelope_width_m": robot_envelope_width_m,
        "extra_clearance_m": extra,
        "maximum_extra_clearance_m": maximum_extra_clearance_m,
        "wide_control_gap_m": wide_control_gap_m,
    }
    if robot_envelope_width_m <= 0 or maximum_extra_clearance_m <= 0:
        return MechanicalValidity(False, "robot_envelope_and_clearance_must_be_positive", metrics)
    if extra <= 0:
        return MechanicalValidity(False, "corridor_is_impossible_not_narrow", metrics)
    if extra > maximum_extra_clearance_m:
        return MechanicalValidity(False, "corridor_not_narrow_enough", metrics)
    if wide_control_gap_m <= corridor.gap_m + maximum_extra_clearance_m:
        return MechanicalValidity(False, "wide_control_not_mechanically_separate", metrics)
    return MechanicalValidity(True, "narrow_corridor_geometry_resolved", metrics)
