#!/usr/bin/env python3
"""Derive pre-outcome mechanism parameters from nominal preflight trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.envs import LiberoEnv
from crashbench.mechanisms.action_drift import validate_action_drift_pre_outcome
from crashbench.mechanisms.fragile_path_collision import (
    derive_fragile_injections,
    validate_fragile_pre_outcome,
)
from crashbench.mechanisms.narrow_clearance import (
    derive_corridor_from_path,
    validate_corridor_pre_outcome,
)
from crashbench.mechanisms.observation_staleness import validate_staleness_pre_outcome
from scripts.expansion.hash_tree_manifest import resolve_git_head


def derive_geometry_for_row(
    row: dict[str, Any],
    *,
    support_top_z: float | None = None,
    robot_envelope_width_m: float | None = None,
) -> dict[str, Any]:
    mechanism = str(row["mechanism_id"])
    path = np.asarray(row.get("eef_path_xyz", []), dtype=np.float64)
    base = {
        "attempt_id": row["attempt_id"],
        "mechanism_id": mechanism,
        "task_id": int(row["task_id"]),
        "candidate_index": int(row["candidate_index"]),
        "reset_seed": int(row["reset_seed"]),
        "source_state_sha256": row["source_state_sha256"],
        "option_outcomes_opened": 0,
    }
    if row.get("status") != "NOMINAL_COMPLETE" or path.ndim != 2 or len(path) < 2:
        return {**base, "status": "MECHANICAL_INVALID", "reason": "nominal_path_unavailable"}
    try:
        if mechanism == "fragile_path_collision_v2":
            if support_top_z is None:
                raise ValueError("fragile geometry requires live support_top_z")
            severities = {}
            for radius in (0.020, 0.025, 0.030):
                injections = derive_fragile_injections(
                    path,
                    path_fraction=0.5,
                    table_z=float(support_top_z),
                    radius_m=radius,
                    half_height_m=0.05,
                    offpath_offset_m=max(0.12, 4 * radius),
                )
                validity = validate_fragile_pre_outcome(path, injections["on_path"])
                severities[f"radius_{int(radius * 1000):03d}"] = {
                    "validity": asdict(validity),
                    "conditions": {
                        key: None if value is None else asdict(value)
                        for key, value in injections.items()
                    },
                }
        elif mechanism == "narrow_clearance_v1":
            if support_top_z is None or robot_envelope_width_m is None:
                raise ValueError("narrow-clearance geometry requires live support and robot width")
            severities = {}
            for extra in (0.005, 0.010, 0.015):
                gap = float(robot_envelope_width_m + extra)
                corridor = derive_corridor_from_path(
                    path,
                    path_fraction=0.5,
                    gap_m=gap,
                    length_m=0.20,
                    wall_thickness_m=0.02,
                    wall_height_m=0.18,
                    table_z=float(support_top_z),
                )
                wide_gap = float(robot_envelope_width_m + 0.08)
                validity = validate_corridor_pre_outcome(
                    corridor,
                    robot_envelope_width_m=float(robot_envelope_width_m),
                    maximum_extra_clearance_m=0.02,
                    wide_control_gap_m=wide_gap,
                )
                wide = derive_corridor_from_path(
                    path,
                    path_fraction=0.5,
                    gap_m=wide_gap,
                    length_m=0.20,
                    wall_thickness_m=0.02,
                    wall_height_m=0.18,
                    table_z=float(support_top_z),
                )
                severities[f"clearance_{int(extra * 1000):03d}"] = {
                    "validity": asdict(validity),
                    "conditions": {
                        "narrow": asdict(corridor),
                        "wide_control": asdict(wide),
                        "no_wall_control": None,
                    },
                }
        elif mechanism == "observation_staleness_v1":
            severities = {}
            for delay in (1, 3, 5):
                validity = validate_staleness_pre_outcome(
                    delay_steps=delay,
                    policy_action_period_ms=50.0,
                    opportunity_slack_ms=500.0,
                )
                severities[f"delay_{delay}"] = {
                    "validity": asdict(validity),
                    "conditions": {
                        "stale": {"delay_steps": delay},
                        "fresh_control": {"delay_steps": 0},
                        "matched_buffer_control": {"delay_steps": delay, "release_mode": "fresh_each_step"},
                    },
                }
        elif mechanism == "action_drift_v1":
            severities = {}
            for magnitude in (0.05, 0.10, 0.15):
                bias = np.array([magnitude, 0, 0, 0, 0, 0, 0], dtype=np.float64)
                validity = validate_action_drift_pre_outcome(
                    bias=bias,
                    action_low=np.full(7, -1.0),
                    action_high=np.full(7, 1.0),
                    max_translation_bias=0.15,
                )
                severities[f"bias_{int(magnitude * 100):03d}"] = {
                    "validity": asdict(validity),
                    "conditions": {
                        "drift": {"bias": bias.tolist()},
                        "zero_bias_control": {"bias": np.zeros(7).tolist()},
                        "orthogonal_bias_control": {
                            "bias": [0, magnitude, 0, 0, 0, 0, 0]
                        },
                    },
                }
        else:
            raise ValueError(f"unsupported or unauthorized mechanism: {mechanism}")
        valid = all(item["validity"]["valid"] for item in severities.values())
        return {
            **base,
            "status": "MECHANICAL_VALID" if valid else "MECHANICAL_INVALID",
            "reason": "all_severities_preoutcome_valid" if valid else "severity_preoutcome_invalid",
            "support_top_z": support_top_z,
            "robot_envelope_width_m": robot_envelope_width_m,
            "severities": severities,
        }
    except Exception as exc:
        return {
            **base,
            "status": "MECHANICAL_INVALID",
            "reason": f"{type(exc).__name__}:{exc}",
        }


def _path_anchor(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = min(len(path) // 2, len(path) - 2)
    point = (path[index] + path[index + 1]) / 2
    delta = path[index + 1, :2] - path[index, :2]
    if np.linalg.norm(delta) <= 0:
        raise ValueError("nominal path midpoint segment has zero XY length")
    tangent = delta / np.linalg.norm(delta)
    normal = np.array([-tangent[1], tangent[0]])
    return point, normal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-cell", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite mechanical preflight: {args.output}")
    nominal = json.loads(args.nominal_cell.read_text())
    mechanism = nominal["mechanism_id"]
    task_id = int(nominal["task_id"])
    if mechanism == "unstable_final_placement_v2":
        raise RuntimeError("unstable_final_placement_v2 requires separate user authorization")
    needs_live_geometry = mechanism in {"fragile_path_collision_v2", "narrow_clearance_v1"}
    env = LiberoEnv("libero_spatial", task_id, model_family="pi0", seed=0) if needs_live_geometry else None
    init_states = env.default_init_states() if env is not None else None
    rows = []
    for row in nominal["rows"]:
        support = None
        width = None
        if needs_live_geometry and row.get("status") == "NOMINAL_COMPLETE":
            path = np.asarray(row["eef_path_xyz"], dtype=np.float64)
            try:
                env.seed(int(row["reset_seed"]))
                env.reset_to(np.asarray(init_states[int(row["candidate_index"])]))
                point, normal = _path_anchor(path)
                support_row = env.sim_view.static_support_surface_at(
                    point[:2], below_z=float(point[2])
                )
                support = float(support_row["top_z"])
                if mechanism == "narrow_clearance_v1":
                    width = float(env.sim_view.distal_robot_envelope_width(normal)["width_m"])
            except Exception as exc:
                rows.append(
                    {
                        "attempt_id": row["attempt_id"],
                        "mechanism_id": mechanism,
                        "task_id": task_id,
                        "candidate_index": row["candidate_index"],
                        "reset_seed": row["reset_seed"],
                        "source_state_sha256": row["source_state_sha256"],
                        "option_outcomes_opened": 0,
                        "status": "MECHANICAL_INVALID",
                        "reason": f"live_geometry_resolution:{type(exc).__name__}:{exc}",
                    }
                )
                continue
        rows.append(
            derive_geometry_for_row(
                row,
                support_top_z=support,
                robot_envelope_width_m=width,
            )
        )
    valid = sum(row["status"] == "MECHANICAL_VALID" for row in rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crashbench_expansion_mechanical_preflight_cell",
        "mechanism_id": mechanism,
        "task_id": task_id,
        "nominal_cell": str(args.nominal_cell),
        "nominal_cell_sha256": hashlib.sha256(args.nominal_cell.read_bytes()).hexdigest(),
        "execution": {
            "git_commit": resolve_git_head(Path.cwd()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "attempted_sources": len(rows),
        "mechanically_valid_sources": valid,
        "all_attempts_accounted": len(rows) == 8,
        "option_outcomes_opened": 0,
        "rows": rows,
    }
    payload["cell_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "valid": valid, "attempted": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
