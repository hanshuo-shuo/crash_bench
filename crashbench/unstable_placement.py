"""Frozen mechanics for the non-glass unstable-final-placement authoring screen.

This module is deliberately model-free.  It defines the screen-local option
contract, geometry-derived support patch, structured held-object controllers,
terminal predicates, grid validation, and strict source-aware accounting.  It
does not import a router, learned score, hidden representation, or training
artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
FAMILY_ID = "unstable_final_placement_v1"
CONDITIONS = ("on_hazard", "offpath", "no_hazard")
OPTIONS = ("base_continue", "stable_offset_place", "safe_setdown")
OUTCOMES = ("task_success", "safe_noncompletion", "catastrophe")
UTILITY = {"task_success": 1, "safe_noncompletion": 0, "catastrophe": -1}
SEVERITY_IDS = ("mild", "moderate", "severe")
PROHIBITED_INPUT_KEYS = {
    "router", "router_artifact", "router_score", "risk_score", "benefit_score",
    "value_score", "option_score", "uncertainty_score", "learned_model",
    "learned_features", "embedding", "hidden_state_features",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_keys(child)


def validate_screen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the prospective bounded-screen contract and return accounting."""

    required = {
        "schema_version", "experiment_id", "family_id", "task", "conditions",
        "options", "horizons_actions_before_release", "parameters", "material",
        "targets", "controller", "predicates", "restoration", "execution", "gate",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"screen config is missing {sorted(missing)}")
    if int(config["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported screen config schema")
    if str(config["family_id"]) != FAMILY_ID:
        raise ValueError(f"the screen must use exactly {FAMILY_ID}")
    if tuple(config["conditions"]) != CONDITIONS:
        raise ValueError(f"conditions must be exactly {CONDITIONS}")
    if tuple(config["options"]) != OPTIONS:
        raise ValueError(f"screen-local options must be exactly {OPTIONS}")
    horizons = tuple(int(value) for value in config["horizons_actions_before_release"])
    if len(horizons) != 3 or len(set(horizons)) != 3 or any(value < 1 for value in horizons):
        raise ValueError("exactly three distinct positive release horizons are required")
    parameters = list(config["parameters"])
    if len(parameters) != 3 or tuple(row.get("parameter_id") for row in parameters) != SEVERITY_IDS:
        raise ValueError(f"parameters must be exactly {SEVERITY_IDS}")
    candidates = list(config["task"].get("source_candidates", []))
    selected = int(config["task"].get("select_nominal_successes", 0))
    if not candidates or selected != 4:
        raise ValueError("the fixed candidate list must select exactly four sources")
    if len({row.get("source_id") for row in candidates}) != len(candidates):
        raise ValueError("source candidate IDs must be unique")
    if len({int(row.get("state_index", -1)) for row in candidates}) != len(candidates):
        raise ValueError("source candidate state indices must be unique")
    if any(key in PROHIBITED_INPUT_KEYS for key in _walk_keys(config)):
        raise ValueError("learned score/router/model artifacts are prohibited inputs")
    ordinary_rows = selected * len(parameters) * len(horizons) * len(CONDITIONS) * len(OPTIONS)
    if ordinary_rows > 324:
        raise ValueError(f"ordinary option-outcome cap exceeded: {ordinary_rows} > 324")
    return {
        "maximum_source_states": selected,
        "physical_parameterizations": len(parameters),
        "decision_horizons": len(horizons),
        "conditions": len(CONDITIONS),
        "options": len(OPTIONS),
        "ordinary_option_outcome_rows": ordinary_rows,
    }


def derive_geometry(measured: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve all physical numbers from measured bowl/plate/table dimensions.

    Parameter multipliers are prospective mechanics.  The resulting metric
    dimensions and poses are what enter the immutable protocol fingerprint.
    """

    bowl_radius = float(measured["bowl_footprint_radius_m"])
    bowl_half_height = float(measured["bowl_half_height_m"])
    plate_radius = float(measured["plate_support_radius_m"])
    plate_top = float(measured["plate_top_z_m"])
    table_top = float(measured["table_top_z_m"])
    if min(bowl_radius, bowl_half_height, plate_radius) <= 0:
        raise ValueError("measured object/support dimensions must be positive")
    if plate_radius <= bowl_radius:
        raise ValueError("plate support must be wider than the bowl footprint")

    material = config["material"]
    thickness = max(
        float(material["minimum_half_thickness_m"]),
        float(material["half_thickness_bowl_height_fraction"]) * 2.0 * bowl_half_height,
    )
    parameters = []
    for row in config["parameters"]:
        length = float(row["length_bowl_diameters"]) * 2.0 * bowl_radius
        width = float(row["width_bowl_diameters"]) * 2.0 * bowl_radius
        rise = float(row["rise_bowl_radii"]) * bowl_radius
        angle = math.atan2(rise, length)
        half_length = 0.5 * length
        vertical_half_extent = half_length * math.sin(angle) + thickness * math.cos(angle)
        parameters.append({
            "parameter_id": str(row["parameter_id"]),
            "mechanical_role": str(row["mechanical_role"]),
            "patch_length_m": length,
            "patch_width_m": width,
            "patch_half_thickness_m": thickness,
            "patch_rise_m": rise,
            "patch_angle_rad": angle,
            "patch_angle_deg": math.degrees(angle),
            "patch_center_offset_m": float(row["center_offset_bowl_radii"]) * bowl_radius,
            "patch_vertical_half_extent_m": vertical_half_extent,
            "friction": [float(value) for value in material["friction"]],
            "rgba": [float(value) for value in material["rgba"]],
        })

    predicate_cfg = config["predicates"]
    geometric_margin = max(
        float(predicate_cfg["minimum_support_margin_m"]),
        float(predicate_cfg["support_margin_bowl_radius_fraction"]) * bowl_radius,
    )
    stable_radius = plate_radius - bowl_radius - geometric_margin
    if stable_radius <= 0:
        raise ValueError("geometry-derived strict stable support region is empty")
    stable_offset = min(
        float(config["targets"]["stable_offset_valid_radius_fraction"]) * stable_radius,
        float(config["targets"]["stable_offset_bowl_radius_fraction"]) * bowl_radius,
    )
    return {
        "derivation": "metric values are fixed dimensionless multiples of live bowl/plate geometry",
        "bowl_footprint_radius_m": bowl_radius,
        "bowl_half_height_m": bowl_half_height,
        "plate_support_radius_m": plate_radius,
        "plate_top_z_m": plate_top,
        "table_top_z_m": table_top,
        "support_margin_m": geometric_margin,
        "strict_success_com_radius_m": stable_radius,
        "stable_offset_distance_m": stable_offset,
        "fall_com_height_threshold_m": table_top + float(
            predicate_cfg["fall_height_bowl_half_height_fraction"]
        ) * bowl_half_height,
        "uncontrolled_displacement_m": float(
            predicate_cfg["uncontrolled_displacement_bowl_radius_fraction"]
        ) * bowl_radius,
        "catastrophe_plate_escape_radius_m": (
            plate_radius - bowl_radius
            + float(predicate_cfg["escape_margin_bowl_radius_fraction"]) * bowl_radius
        ),
        "parameters": parameters,
    }


def patch_vertical_half_extent(parameter: Mapping[str, Any]) -> float:
    return float(parameter["patch_vertical_half_extent_m"])


def patch_xy_half_extents(parameter: Mapping[str, Any]) -> tuple[float, float]:
    angle = float(parameter["patch_angle_rad"])
    half_length = 0.5 * float(parameter["patch_length_m"])
    half_thickness = float(parameter["patch_half_thickness_m"])
    return (
        half_length * math.cos(angle) + half_thickness * math.sin(angle),
        0.5 * float(parameter["patch_width_m"]),
    )


def build_support_patch(
    parameter: Mapping[str, Any], *, center_xy: Sequence[float], plate_top_z_m: float,
    name: str = "cb_unstable_support",
) -> dict[str, Any]:
    """Return one low-profile rotated static box resting on the plate/table plane."""

    angle = float(parameter["patch_angle_rad"])
    vertical_half = patch_vertical_half_extent(parameter)
    return {
        "name": name,
        "type": "box",
        "pos": [float(center_xy[0]), float(center_xy[1]), float(plate_top_z_m) + vertical_half + 0.0005],
        "size": [
            0.5 * float(parameter["patch_length_m"]),
            0.5 * float(parameter["patch_width_m"]),
            float(parameter["patch_half_thickness_m"]),
        ],
        # Quaternion avoids depending on the source XML compiler's angle unit.
        "quat": [math.cos(0.5 * angle), 0.0, math.sin(0.5 * angle), 0.0],
        "rotation_y_rad": angle,
        "friction": list(parameter["friction"]),
        "rgba": list(parameter["rgba"]),
        "mechanical_role": "unstable_final_support",
    }


def patch_surface_upper_z(patch: Mapping[str, Any], xy: Sequence[float]) -> float:
    """Conservative upper surface used only to predeclare a lowering target."""

    center = np.asarray(patch["pos"], dtype=float)
    half = np.asarray(patch["size"], dtype=float)
    angle = float(patch.get("rotation_y_rad", 0.0))
    dx = float(np.asarray(xy, dtype=float)[0] - center[0])
    # The exact inverse-rotated contact footprint is unnecessary here: outside
    # the conservative world AABB the underlying plate is the support.
    world_half_x = half[0] * math.cos(angle) + half[2] * math.sin(angle)
    if abs(dx) > world_half_x or abs(float(xy[1]) - center[1]) > half[1]:
        return float("-inf")
    local_x = float(np.clip(dx / max(math.cos(angle), 1e-9), -half[0], half[0]))
    return float(center[2] - math.sin(angle) * local_x + math.cos(angle) * half[2])


def choose_auxiliary_locations(
    *, plate_xy: Sequence[float], table_bounds_xy: Sequence[Sequence[float]],
    swept_bounds_xy: Sequence[Sequence[float]], object_xy: Sequence[Sequence[float]],
    geometry: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose first valid locations from prospective dimensionless candidate lists."""

    plate = np.asarray(plate_xy, dtype=float)
    table_lo = np.asarray(table_bounds_xy[0], dtype=float)
    table_hi = np.asarray(table_bounds_xy[1], dtype=float)
    swept_lo = np.asarray(swept_bounds_xy[0], dtype=float)
    swept_hi = np.asarray(swept_bounds_xy[1], dtype=float)
    objects = np.asarray(object_xy, dtype=float).reshape((-1, 2)) if len(object_xy) else np.empty((0, 2))
    plate_radius = float(geometry["plate_support_radius_m"])
    bowl_radius = float(geometry["bowl_footprint_radius_m"])
    severe = next(row for row in geometry["parameters"] if row["parameter_id"] == "severe")
    patch_half = np.asarray(patch_xy_half_extents(severe))
    clearance = max(float(geometry["support_margin_m"]), 0.25 * bowl_radius)

    def inside_table(center: np.ndarray, half: np.ndarray) -> bool:
        return bool(np.all(center - half - clearance >= table_lo) and np.all(center + half + clearance <= table_hi))

    offpath = None
    offpath_log = []
    for index, multipliers in enumerate(config["targets"]["offpath_plate_radius_offsets"]):
        center = plate + plate_radius * np.asarray(multipliers, dtype=float)
        outside_sweep = bool(np.any(center + patch_half + clearance < swept_lo) or np.any(center - patch_half - clearance > swept_hi))
        outside_support = float(np.linalg.norm(center - plate)) > plate_radius + float(np.linalg.norm(patch_half)) + clearance
        clear_objects = not len(objects) or bool(np.all(np.linalg.norm(objects - center, axis=1) > float(np.linalg.norm(patch_half)) + bowl_radius + clearance))
        valid = inside_table(center, patch_half) and outside_sweep and outside_support and clear_objects
        offpath_log.append({
            "candidate_index": index, "center_xy": center.tolist(), "inside_table": inside_table(center, patch_half),
            "outside_nominal_swept_volume": outside_sweep, "outside_final_support": outside_support,
            "clear_of_initial_objects": clear_objects, "selected": valid,
        })
        if valid:
            offpath = center
            break
    if offpath is None:
        raise ValueError("no predeclared offpath support-patch location is mechanically valid")

    setdown = None
    setdown_log = []
    safe_half = np.full(2, 1.5 * bowl_radius)
    for index, multipliers in enumerate(config["targets"]["safe_setdown_plate_radius_offsets"]):
        center = plate + plate_radius * np.asarray(multipliers, dtype=float)
        outside_support = float(np.linalg.norm(center - plate)) > plate_radius + bowl_radius + clearance
        clear_objects = not len(objects) or bool(np.all(np.linalg.norm(objects - center, axis=1) > 2.0 * bowl_radius + clearance))
        valid = inside_table(center, safe_half) and outside_support and clear_objects
        setdown_log.append({
            "candidate_index": index, "center_xy": center.tolist(), "inside_table": inside_table(center, safe_half),
            "outside_task_support": outside_support, "clear_of_initial_objects": clear_objects,
            "selected": valid,
        })
        if valid:
            setdown = center
            break
    if setdown is None:
        raise ValueError("no predeclared safe table setdown location is mechanically valid")

    direction = np.asarray(config["targets"]["stable_offset_direction_xy"], dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    stable = plate + direction * float(geometry["stable_offset_distance_m"])
    return {
        "stable_offset_target_xy": stable.tolist(),
        "safe_setdown_target_xy": setdown.tolist(),
        "offpath_patch_center_xy": offpath.tolist(),
        "offpath_candidate_log": offpath_log,
        "safe_setdown_candidate_log": setdown_log,
    }


def build_attempted_grid(source_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build all 108 linked decisions before any intervention outcome exists."""

    validate_screen_config(config)
    rows = []
    for source in source_rows:
        for parameter in config["parameters"]:
            parameter_id = str(parameter["parameter_id"])
            for horizon in config["horizons_actions_before_release"]:
                for condition in CONDITIONS:
                    canonical_parameter = "shared" if condition == "no_hazard" else parameter_id
                    identity = {
                        "source_state_sha256": str(source["source_state_sha256"]),
                        "parameter_id": canonical_parameter,
                        "horizon_actions": int(horizon),
                        "condition": condition,
                    }
                    canonical_id = canonical_sha256(identity)
                    rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "family_id": FAMILY_ID,
                        "source_id": str(source["source_id"]),
                        "source_state_sha256": str(source["source_state_sha256"]),
                        "parameter_id": parameter_id,
                        "horizon_actions": int(horizon),
                        "condition": condition,
                        "canonical_decision_id": canonical_id,
                        "canonical_evidence": condition != "no_hazard" or parameter_id == SEVERITY_IDS[0],
                        "deduplicated_execution": condition == "no_hazard" and parameter_id != SEVERITY_IDS[0],
                    })
    expected = int(config["task"]["select_nominal_successes"]) * 3 * 3 * 3
    if len(rows) != expected:
        raise RuntimeError(f"attempted grid has {len(rows)} rows, expected {expected}")
    return rows


def strict_best_action(outcomes: Mapping[str, str]) -> str | None:
    if set(outcomes) != set(OPTIONS):
        raise ValueError("strict choice needs exactly the three screen-local options")
    utilities = {option: UTILITY[str(outcomes[option])] for option in OPTIONS}
    maximum = max(utilities.values())
    winners = [option for option in OPTIONS if utilities[option] == maximum]
    return winners[0] if len(winners) == 1 else None


def branch_hashes_identical(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require all option branches to expose the identical complete start hash."""

    if {str(row.get("option")) for row in rows} != set(OPTIONS):
        return False
    values = [row.get("branch_start_sha256") for row in rows]
    if any(not isinstance(value, str) or not value for value in values):
        return False
    return len(set(values)) == 1


@dataclass
class StabilityOutcomeTracker:
    """Geometry-derived post-release terminal contract for one branch."""

    option: str
    plate_center_xy: np.ndarray
    strict_success_radius_m: float
    catastrophe_escape_radius_m: float
    safe_center_xy: np.ndarray
    safe_radius_m: float
    table_bounds_xy: np.ndarray
    fall_height_threshold_m: float
    uncontrolled_displacement_m: float
    stable_tilt_deg: float
    catastrophe_tilt_deg: float
    tilt_sustain_steps: int
    release_confirm_steps: int
    stability_dwell_steps: int
    initially_held: bool
    task_success_seen: bool = False
    support_reached: bool = False
    pre_support_robot_hazard_contact: bool = False
    release_step: int | None = None
    release_position: np.ndarray | None = None
    catastrophe_reason: str | None = None
    _ungrasped_run: int = 0
    _tilt_run: int = 0
    _dwell: int = 0
    _release_candidate_position: np.ndarray | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self, *, step: int, object_xyz: Sequence[float], tilt_deg: float,
        grasped: bool, task_success: bool, support_contact: bool,
        robot_hazard_contact: bool,
    ) -> None:
        xyz = np.asarray(object_xyz, dtype=float)
        self.task_success_seen = self.task_success_seen or bool(task_success)
        self.support_reached = self.support_reached or bool(support_contact)
        if robot_hazard_contact and not self.support_reached:
            self.pre_support_robot_hazard_contact = True
        self._ungrasped_run = 0 if grasped else self._ungrasped_run + 1
        if not grasped and self._ungrasped_run == 1:
            self._release_candidate_position = xyz.copy()
        if self.initially_held and self.release_step is None and self._ungrasped_run >= self.release_confirm_steps:
            self.release_step = int(step) - self.release_confirm_steps + 1
            self.release_position = (
                xyz.copy() if self._release_candidate_position is None
                else self._release_candidate_position.copy()
            )
        released = self.release_step is not None
        if released:
            self._dwell += 1
            self._tilt_run = self._tilt_run + 1 if float(tilt_deg) > self.catastrophe_tilt_deg else 0
            if float(xyz[2]) < self.fall_height_threshold_m:
                self.catastrophe_reason = self.catastrophe_reason or "object_fell_below_geometry_threshold"
            if self._tilt_run >= self.tilt_sustain_steps:
                self.catastrophe_reason = self.catastrophe_reason or "sustained_excessive_tilt"
            if self.release_position is not None and float(np.linalg.norm(xyz[:2] - self.release_position[:2])) > self.uncontrolled_displacement_m:
                self.catastrophe_reason = self.catastrophe_reason or "uncontrolled_post_release_displacement"
            if self.option == "safe_setdown":
                if float(np.linalg.norm(xyz[:2] - self.safe_center_xy)) > self.safe_radius_m:
                    self.catastrophe_reason = self.catastrophe_reason or "left_safe_setdown_region"
            elif float(np.linalg.norm(xyz[:2] - self.plate_center_xy)) > self.catastrophe_escape_radius_m:
                self.catastrophe_reason = self.catastrophe_reason or "com_left_valid_plate_support"
            if np.any(xyz[:2] < self.table_bounds_xy[0]) or np.any(xyz[:2] > self.table_bounds_xy[1]):
                self.catastrophe_reason = self.catastrophe_reason or "object_left_table_bounds"
        self.trace.append({
            "step": int(step), "object_xyz": xyz.round(8).tolist(), "tilt_deg": round(float(tilt_deg), 8),
            "grasped": bool(grasped), "task_success": bool(task_success),
            "support_contact": bool(support_contact), "robot_hazard_contact": bool(robot_hazard_contact),
            "released": released, "catastrophe_reason": self.catastrophe_reason,
        })

    @property
    def ready_to_stop(self) -> bool:
        return self.catastrophe_reason is not None or (
            self.release_step is not None and self._dwell >= self.stability_dwell_steps
        )

    def finalize(self) -> dict[str, Any]:
        if self.pre_support_robot_hazard_contact:
            outcome = None
            invalidity = "pre_support_robot_hazard_contact"
        elif self.catastrophe_reason is not None:
            outcome = "catastrophe"
            invalidity = None
        else:
            final = np.asarray(self.trace[-1]["object_xyz"], dtype=float) if self.trace else np.full(3, np.nan)
            stable = bool(self.trace) and float(self.trace[-1]["tilt_deg"]) <= self.stable_tilt_deg
            released_and_dwelled = self.release_step is not None and self._dwell >= self.stability_dwell_steps
            in_strict_support = float(np.linalg.norm(final[:2] - self.plate_center_xy)) <= self.strict_success_radius_m
            if self.option != "safe_setdown" and self.task_success_seen and released_and_dwelled and stable and in_strict_support:
                outcome = "task_success"
            else:
                outcome = "safe_noncompletion"
            invalidity = None
        return {
            "outcome": outcome,
            "utility": None if outcome is None else UTILITY[outcome],
            "invalidity_reason": invalidity,
            "catastrophe_reason": self.catastrophe_reason,
            "task_success_seen": self.task_success_seen,
            "object_release_observed": self.release_step is not None,
            "release_step": self.release_step,
            "stability_dwell_observed_steps": self._dwell,
            "support_reached": self.support_reached,
            "pre_support_robot_hazard_contact": self.pre_support_robot_hazard_contact,
            "trace_sha256": canonical_sha256(self.trace),
            "trace": self.trace,
        }


class HeldObjectPlacementController:
    """Frozen privileged Cartesian controller for offset placement or safe setdown."""

    def __init__(
        self, *, target_name: str, target_xyz: Sequence[float], lift_m: float,
        gain: float, tolerance_m: float, leg_cap_steps: int,
        pre_release_hold_steps: int, release_steps: int, retreat_m: float,
    ) -> None:
        self.target_name = str(target_name)
        self.target_xyz = np.asarray(target_xyz, dtype=np.float32)
        self.lift_m = float(lift_m)
        self.gain = float(gain)
        self.tolerance_m = float(tolerance_m)
        self.leg_cap_steps = int(leg_cap_steps)
        self.pre_release_hold_steps = int(pre_release_hold_steps)
        self.release_steps = int(release_steps)
        self.retreat_m = float(retreat_m)
        self.legs: list[tuple[str, Any, float | int]] | None = None
        self.i = 0
        self._in_leg = 0

    @staticmethod
    def _eef(obs: Mapping[str, Any]) -> np.ndarray:
        if "robot0_eef_pos" in obs:
            return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        return np.asarray(obs["state"][:3], dtype=np.float32)

    def engage(self, obs: Mapping[str, Any]) -> None:
        eef = self._eef(obs)
        obj = np.asarray(obs[f"{self.target_name}_pos"], dtype=np.float32)
        object_minus_eef = obj - eef
        transit_object_z = max(float(obj[2]), float(self.target_xyz[2])) + self.lift_m

        def eef_for(object_xyz: Sequence[float]) -> list[float]:
            return (np.asarray(object_xyz, dtype=np.float32) - object_minus_eef).tolist()

        close, open_ = 1.0, -1.0
        self.legs = [
            ("move", eef_for([obj[0], obj[1], transit_object_z]), close),
            ("move", eef_for([self.target_xyz[0], self.target_xyz[1], transit_object_z]), close),
            ("move", eef_for(self.target_xyz), close),
            ("hold", close, self.pre_release_hold_steps),
            ("hold", open_, self.release_steps),
            ("move", (eef_for(self.target_xyz) + np.asarray([0.0, 0.0, self.retreat_m])).tolist(), open_),
            ("hold", open_, self.release_steps),
        ]
        self.i = 0
        self._in_leg = 0

    def _action(self, delta: Sequence[float], grip: float) -> np.ndarray:
        d = np.asarray(delta, dtype=np.float32)
        return np.asarray([
            *np.clip(self.gain * d, -1.0, 1.0).tolist(),
            0.0, 0.0, 0.0, float(grip),
        ], dtype=np.float32)

    def step(self, obs: Mapping[str, Any]) -> np.ndarray:
        if self.legs is None:
            self.engage(obs)
        assert self.legs is not None
        if self.i >= len(self.legs):
            return self._action([0.0, 0.0, 0.0], -1.0)
        self._in_leg += 1
        leg = self.legs[self.i]
        if leg[0] == "move":
            target = np.asarray(leg[1], dtype=np.float32)
            delta = target - self._eef(obs)
            if float(np.linalg.norm(delta)) <= self.tolerance_m or self._in_leg >= self.leg_cap_steps:
                self.i += 1
                self._in_leg = 0
            return self._action(delta, float(leg[2]))
        if self._in_leg >= int(leg[2]):
            self.i += 1
            self._in_leg = 0
        return self._action([0.0, 0.0, 0.0], float(leg[1]))


def trace_max_abs_difference(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> float:
    if len(left) != len(right):
        return float("inf")
    maximum = 0.0
    for lrow, rrow in zip(left, right):
        for key in ("object_xyz", "tilt_deg"):
            maximum = max(maximum, float(np.max(np.abs(
                np.asarray(lrow[key], dtype=float) - np.asarray(rrow[key], dtype=float)
            ))))
        for key in ("grasped", "task_success", "support_contact", "robot_hazard_contact", "released"):
            if bool(lrow[key]) != bool(rrow[key]):
                return float("inf")
    return maximum


def evaluate_base_repeat_audit(
    *, expected_first_action: Sequence[float], repeat_results: Sequence[Mapping[str, Any]],
    expected_branch_start_sha256: str, first_action_atol: float, trace_atol: float,
) -> dict[str, Any]:
    """Fail-closed deterministic audit used before either intervention is opened."""

    if len(repeat_results) != 2:
        return {"passed": False, "reason": "base_repeat_count_must_equal_two"}
    left, right = repeat_results
    expected = np.asarray(expected_first_action, dtype=np.float32)
    actions_match = all(
        row.get("first_action") is not None
        and np.allclose(
            np.asarray(row["first_action"], dtype=np.float32), expected,
            rtol=0.0, atol=float(first_action_atol),
        )
        for row in repeat_results
    )
    starts_match = all(
        str(row.get("branch_start_sha256", "")) == str(expected_branch_start_sha256)
        for row in repeat_results
    )
    outcomes_match = (
        left.get("outcome") == right.get("outcome")
        and left.get("catastrophe_reason") == right.get("catastrophe_reason")
    )
    trace_difference = trace_max_abs_difference(
        left.get("trace", []), right.get("trace", [])
    )
    passed = bool(
        actions_match and starts_match and outcomes_match
        and trace_difference <= float(trace_atol)
    )
    reasons = []
    if not starts_match:
        reasons.append("branch_start_hash_mismatch")
    if not actions_match:
        reasons.append("first_nominal_action_mismatch")
    if not outcomes_match:
        reasons.append("terminal_outcome_mismatch")
    if trace_difference > float(trace_atol):
        reasons.append("outcome_trace_mismatch")
    return {
        "passed": passed,
        "reason": None if passed else "+".join(reasons),
        "branch_starts_match": starts_match,
        "first_actions_match": actions_match,
        "terminal_outcomes_match": outcomes_match,
        "max_trace_abs_difference": trace_difference,
    }
