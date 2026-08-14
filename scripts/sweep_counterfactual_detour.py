#!/usr/bin/env python3
"""Select one frozen DetourComplete configuration on exposed exact states.

This is a controller-development gate, not counterfactual training data.  It
uses only previously exposed H=20 pair artifacts, never reads their per-state
oracle configuration, and applies every candidate configuration to every exact
state.  Repeated rollouts make simulator instability explicit before a single
common configuration is recommended for the protocol-correct smoke.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.counterfactual_router import classify_option_outcome
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import (
    array_sha256,
    canonical_sha256,
    read_placement_manifest,
)
from crashbench.provenance import repository_provenance
from crashbench.recovery import DetourComplete
from scripts.collect_counterfactual_option_rollouts import _run_structured_option
from scripts.collect_glass_recovery_pairs import (
    PLATE,
    TARGET,
    _branch_start_hashes,
    _controller_glass,
    _require_branch_start_hashes,
)


def build_configs(
    *,
    sides: Iterable[float],
    lane_margins: Iterable[float],
    lift_offsets: Iterable[float],
    descend_offsets: Iterable[float],
    grasp_xy_offsets: Iterable[Iterable[float]],
    departure_clearance: float,
) -> list[dict[str, Any]]:
    """Return the deterministic Cartesian controller grid."""

    configs = []
    for side, lane, lift, descend, grasp_xy in itertools.product(
        sides, lane_margins, lift_offsets, descend_offsets, grasp_xy_offsets
    ):
        config = {
            "side": float(side),
            "lane_margin": float(lane),
            "lift_offset": float(lift),
            "descend_offset": float(descend),
            "grasp_xy_offset": [float(value) for value in grasp_xy],
            "departure_clearance": float(departure_clearance),
            "orientation_target": None,
            "path_aligned": True,
        }
        if len(config["grasp_xy_offset"]) != 2:
            raise ValueError("each grasp xy offset must contain exactly two values")
        config["config_id"] = canonical_sha256(config)[:12]
        configs.append(config)
    return configs


def summarize_sweep(
    rows: Iterable[Mapping[str, Any]], configs: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Rank common configs without hiding state- or replicate-specific choices."""

    rows = [dict(row) for row in rows]
    configs = [dict(config) for config in configs]
    by_config_state: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        by_config_state[(str(row["config_id"]), str(row["state_id"]))].append(
            str(row["outcome"])
        )
    state_ids = sorted({str(row["state_id"]) for row in rows})
    summaries = []
    for config in configs:
        config_id = str(config["config_id"])
        outcomes = {}
        unstable_states = []
        for state_id in state_ids:
            values = by_config_state.get((config_id, state_id), [])
            if not values:
                raise ValueError(f"config {config_id} is missing state {state_id}")
            counts = dict(sorted(Counter(values).items()))
            outcomes[state_id] = counts
            if len(counts) != 1:
                unstable_states.append(state_id)
        stable_outcomes = {
            state_id: next(iter(counts))
            for state_id, counts in outcomes.items()
            if len(counts) == 1
        }
        summary = {
            "config_id": config_id,
            "config": {key: value for key, value in config.items() if key != "config_id"},
            "state_outcomes": outcomes,
            "unstable_states": unstable_states,
            "task_success_states": sum(
                outcome == "task_success" for outcome in stable_outcomes.values()
            ),
            "catastrophe_states": sum(
                outcome == "catastrophe" for outcome in stable_outcomes.values()
            ),
            "safe_noncompletion_states": sum(
                outcome == "safe_noncompletion" for outcome in stable_outcomes.values()
            ),
        }
        summary["eligible"] = bool(
            not unstable_states and summary["task_success_states"] >= 1
        )
        summaries.append(summary)
    eligible = [row for row in summaries if row["eligible"]]
    eligible.sort(key=lambda row: (
        -int(row["task_success_states"]),
        int(row["catastrophe_states"]),
        -int(row["safe_noncompletion_states"]),
        str(row["config_id"]),
    ))
    winner = eligible[0] if eligible else None
    return {
        "state_count": len(state_ids),
        "config_count": len(configs),
        "go": winner is not None,
        "decision": (
            f"freeze common config {winner['config_id']} and rerun multi-H smoke"
            if winner is not None
            else "no stable common config achieves task success; do not run full"
        ),
        "recommended_config": None if winner is None else winner,
        "configs": summaries,
    }


def _load_state_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    placements = {}
    for path in args.placements:
        manifest_rows, _ = read_placement_manifest(path)
        for placement in manifest_rows:
            previous = placements.setdefault(placement.placement_id, placement)
            if previous != placement:
                raise ValueError(f"conflicting placement {placement.placement_id}")
    specs = []
    for raw_root in args.pair_dir:
        root = Path(raw_root).resolve()
        pair = json.loads((root / "pair.json").read_text())
        nominal = next(
            row for row in pair["records"]
            if row["trajectory_kind"] == "nominal_catastrophe"
        )
        if nominal.get("crashed") is not True:
            raise ValueError(f"{root}: nominal branch is not a measured catastrophe")
        if int(nominal.get("trigger_horizon_actions", -1)) != args.required_horizon:
            raise ValueError(
                f"{root}: expected H={args.required_horizon}, got "
                f"{nominal.get('trigger_horizon_actions')}"
            )
        placement_id = str(nominal["placement_id"])
        if placement_id not in placements:
            raise ValueError(f"{root}: placement {placement_id} is not in the manifests")
        state = np.load(root / "precrash_onpath_state.npy", allow_pickle=False)
        expected = nominal["metadata"]["branch_start_hashes"]
        if array_sha256(state) != expected["simulator_state_sha256"]:
            raise ValueError(f"{root}: exact-state hash mismatch")
        with np.load(root / "controller_state.npz", allow_pickle=False) as archive:
            controller_state = {key: archive[key] for key in archive.files}
        specs.append({
            "state_id": placement_id,
            "root": root,
            "placement": placements[placement_id],
            "state": state,
            "model_xml": (root / "onpath_model.xml").read_text(),
            "controller_state": controller_state,
            "expected_hashes": expected,
        })
    if len({spec["state_id"] for spec in specs}) != len(specs):
        raise ValueError("pair-dir inputs must have distinct placement ids")
    return specs


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    repo = repository_provenance(Path(__file__).resolve().parents[1], require_clean=True)
    specs = _load_state_specs(args)
    configs = build_configs(
        sides=args.side,
        lane_margins=args.lane_margin,
        lift_offsets=args.lift_offset,
        descend_offsets=args.descend_offset,
        grasp_xy_offsets=args.grasp_xy,
        departure_clearance=args.departure_clearance,
    )
    rows = []
    envs = {}
    for spec in specs:
        placement = spec["placement"]
        env_key = (placement.task_suite, int(placement.task_id))
        if env_key not in envs:
            envs[env_key] = LiberoEnv(*env_key, seed=args.rollout_seed)
        env = envs[env_key]
        for config in configs:
            for replicate in range(args.replicates):
                env.seed(args.rollout_seed)
                obs = env.reset_to_exact(spec["state"], model_xml=spec["model_xml"])
                obs = env.restore_controller_state(spec["controller_state"])
                _require_branch_start_hashes(
                    _branch_start_hashes(env, obs), spec["expected_hashes"],
                    label=f"{spec['state_id']}:{config['config_id']}:{replicate}",
                )
                bowl = np.asarray(obs[f"{TARGET}_pos"], dtype=float)
                plate = np.asarray(obs[f"{PLATE}_pos"], dtype=float)
                controller = DetourComplete(
                    _controller_glass(placement.on_path_glass), bowl, plate,
                    side=config["side"], lane_margin=config["lane_margin"],
                    transit_z=float(bowl[2] + config["lift_offset"]),
                    descend_off=config["descend_offset"], leg_cap=args.leg_cap,
                    target_name=TARGET, orientation_target=None, path_aligned=True,
                    grasp_xy_offset=config["grasp_xy_offset"],
                    departure_clearance=config["departure_clearance"],
                )
                result = _run_structured_option(
                    env, obs, [placement.on_path_glass], controller,
                    max_steps=args.max_steps,
                )
                row = {
                    "state_id": spec["state_id"],
                    "source_state_sha256": placement.source_state_sha256,
                    "required_horizon_actions": args.required_horizon,
                    "config_id": config["config_id"],
                    "replicate": replicate,
                    "outcome": classify_option_outcome(
                        crashed=bool(result["crashed"]),
                        succeeded=bool(result["succeeded"]),
                    ),
                    **result,
                }
                rows.append(row)
                with (output / "sweep_rows.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_sweep(rows, configs)
    manifest = {
        "schema_version": 1,
        "kind": "counterfactual_detour_development_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo,
        "development_only": True,
        "per_state_oracle_configs_used": False,
        "required_horizon_actions": args.required_horizon,
        "replicates": args.replicates,
        "states": [spec["state_id"] for spec in specs],
        "summary": summary,
    }
    (output / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if summary["recommended_config"] is not None:
        (output / "frozen_detour_config.json").write_text(
            json.dumps(summary["recommended_config"]["config"], indent=2, sort_keys=True)
            + "\n"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", action="append", required=True)
    parser.add_argument("--pair-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-horizon", type=int, default=20)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--leg-cap", type=int, default=140)
    parser.add_argument("--side", type=float, nargs="+", default=[-1.0, 1.0])
    parser.add_argument("--lane-margin", type=float, nargs="+", default=[0.12, 0.18])
    parser.add_argument("--lift-offset", type=float, nargs="+", default=[0.30, 0.38])
    parser.add_argument("--descend-offset", type=float, nargs="+", default=[0.018, 0.04])
    parser.add_argument(
        "--grasp-xy", type=float, nargs=2, action="append",
        default=[[0.009, -0.04], [-0.003, -0.05], [0.013, -0.06]],
    )
    parser.add_argument("--departure-clearance", type=float, default=0.06)
    args = parser.parse_args()
    if args.replicates < 2 or args.max_steps < 1 or args.leg_cap < 1:
        raise SystemExit("replicates must be >=2 and step caps must be positive")
    run(args)


if __name__ == "__main__":
    main()
