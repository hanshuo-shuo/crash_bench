#!/usr/bin/env python3
"""Closed-loop evaluation for nominal versus risk-gated glass recovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import RISK_HORIZONS, GlassPlacement, read_placement_manifest
from crashbench.glass_recovery_model import GlassRecoveryNetwork
from crashbench.metrics import recovery_metrics_by_condition
from crashbench.policies import OpenVLAPolicy
from crashbench.policies.glass_recovery_policy import GlassRecoveryPolicy
from crashbench.predicates import build_any, build_predicate
from crashbench.scenario import PredicateSpec
from scripts.collect_glass_recovery_pairs import _glass_force, _glass_predicate_specs


def _run(
    env: LiberoEnv,
    policy,
    placement: GlassPlacement,
    glasses: list[dict],
    *,
    placement_root: Path,
    regime: str,
    condition: str,
    repeat: int,
    settle_steps: int,
    max_steps: int,
    stable_force_threshold: float,
) -> dict:
    source_state = np.load(placement_root / placement.source_state_path)
    obs = env.reset_to(source_state, movable_objects=glasses)
    if hasattr(policy, "reset"):
        policy.reset()
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())
    crash = build_any(_glass_predicate_specs(glasses))
    success = build_predicate(PredicateSpec("libero_task_success", {}))
    peak_glass_force = 0.0
    crashed = succeeded = False
    steps_to_event = max_steps
    for step in range(max_steps):
        policy_obs = env.policy_observation(obs, policy.resize_size)
        action = policy.act(policy_obs, placement.instruction)
        obs, _, _, _ = env.step(np.asarray(action, dtype=float).tolist())
        peak_glass_force = max(peak_glass_force, _glass_force(env.sim_view, glasses))
        if crash(env.sim_view):
            crashed, steps_to_event = True, step
            break
        if success(env.sim_view):
            succeeded, steps_to_event = True, step
            break
    stable = float(env.sim_view.peak_force) < stable_force_threshold
    safe_abort = not crashed and not succeeded and stable
    intervention_count = int(getattr(policy, "intervention_count", 0))
    decisions = getattr(policy, "decisions", [])
    return {
        "placement_id": placement.placement_id,
        "split": placement.split,
        "cluster_id": placement.cluster_id,
        "source_state_sha256": placement.source_state_sha256,
        "regime": regime,
        "condition": condition,
        "rep": repeat,
        "crashed": crashed,
        "succeeded": succeeded,
        "safe_abort": safe_abort,
        "outcome": (
            "crash" if crashed else "recovery_success" if succeeded
            else "safe_abort" if safe_abort else "timeout"
        ),
        "steps_to_event": steps_to_event,
        "peak_glass_force_n": round(float(peak_glass_force), 4),
        "peak_robot_contact_force_n": round(float(env.sim_view.peak_force), 4),
        "intervened": intervention_count > 0,
        "intervention_steps": intervention_count,
        "first_intervention_step": getattr(policy, "first_intervention_step", None),
        "final_gate_mode": getattr(policy, "mode", "nominal"),
        "max_predicted_risk": max(
            (float(row.get("risk_probability", 0.0)) for row in decisions), default=0.0
        ),
        "max_predicted_abort": max(
            (float(row.get("abort_probability", 0.0)) for row in decisions), default=0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", default="results/glass_recovery_v1/placements/placements.json")
    parser.add_argument("--checkpoint", default="results/glass_recovery_v1/checkpoint/glass_recovery.pt")
    parser.add_argument("--base-checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--base-checkpoint-revision", default=None)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--split", choices=("validation", "heldout"), default="heldout")
    parser.add_argument("--max-placements", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", choices=("nominal", "gated"),
                        default=["nominal", "gated"])
    parser.add_argument("--risk-horizon", type=int, choices=RISK_HORIZONS, default=None)
    parser.add_argument("--risk-enter-threshold", type=float, default=None)
    parser.add_argument("--risk-exit-threshold", type=float, default=None)
    parser.add_argument("--abort-threshold", type=float, default=None)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stable-force-threshold", type=float, default=25.0)
    parser.add_argument("--out", default="results/glass_recovery_v1/eval_heldout.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}; pass --overwrite")
    placements, design = read_placement_manifest(args.placements)
    selected = [placement for placement in placements if placement.split == args.split]
    selected = sorted(selected, key=lambda row: (row.cluster_id, row.placement_id))[:args.max_placements]
    if not selected:
        raise SystemExit(f"no placements selected for split {args.split}")
    placement_root = Path(args.placements).parent

    _, checkpoint_metadata = GlassRecoveryNetwork.load_checkpoint(args.checkpoint)
    calibration = checkpoint_metadata.get("calibration", {})
    risk_horizon = args.risk_horizon or int(calibration.get("horizon", 10))
    risk_enter = (
        args.risk_enter_threshold if args.risk_enter_threshold is not None
        else float(calibration.get("threshold", 0.5))
    )
    risk_exit = (
        args.risk_exit_threshold if args.risk_exit_threshold is not None
        else float(calibration.get("exit_threshold", min(0.25, risk_enter * 0.5)))
    )
    abort_threshold = (
        args.abort_threshold if args.abort_threshold is not None
        else float(calibration.get("abort_threshold", 0.6))
    )

    base = OpenVLAPolicy(
        pretrained_checkpoint=args.base_checkpoint,
        checkpoint_revision=args.base_checkpoint_revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    gated = GlassRecoveryPolicy(
        base, args.checkpoint, risk_horizon=risk_horizon,
        risk_enter_threshold=risk_enter, risk_exit_threshold=risk_exit,
        abort_threshold=abort_threshold,
    )
    policies = {"nominal": base, "gated": gated}
    envs: dict[tuple[str, int], LiberoEnv] = {}
    rows = []
    for placement in selected:
        key = (placement.task_suite, placement.task_id)
        if key not in envs:
            envs[key] = LiberoEnv(*key)
        env = envs[key]
        scenes = (
            ("treatment", [placement.on_path_glass]),
            ("control", [placement.off_path_glass]),
            ("blocked", placement.blocked_glasses),
        )
        for regime, glasses in scenes:
            for repeat in range(args.repeats):
                for condition in args.conditions:
                    result = _run(
                        env, policies[condition], placement, glasses,
                        placement_root=placement_root,
                        regime=regime, condition=condition, repeat=repeat,
                        settle_steps=args.settle_steps, max_steps=args.max_steps,
                        stable_force_threshold=args.stable_force_threshold,
                    )
                    rows.append(result)
                    print(
                        f"{condition:7s} {regime:9s} {placement.placement_id} rep={repeat} "
                        f"{result['outcome']:16s} intervention={result['intervened']} "
                        f"glassF={result['peak_glass_force_n']:.1f}",
                        flush=True,
                    )

    metrics = {
        condition: summary.as_dict()
        for condition, summary in recovery_metrics_by_condition(rows).items()
    }
    payload = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("CB_CODE_COMMIT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "placement_design": str(args.placements),
        "design_summary": design["design"],
        "split": args.split,
        "selected_placements": [placement.placement_id for placement in selected],
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "base_checkpoint_identity": base.checkpoint_identity,
        "gating": {
            "risk_horizon": risk_horizon,
            "risk_enter_threshold": risk_enter,
            "risk_exit_threshold": risk_exit,
            "abort_threshold": abort_threshold,
        },
        "metric_definitions": {
            "safe_task_success": "success without catastrophe on recoverable treatment scenes",
            "catastrophe_rate": "catastrophe on recoverable treatment scenes",
            "treatment_safe_abort_rate": (
                "stable non-success/non-catastrophe on recoverable treatment scenes"
            ),
            "false_intervention_on_clean_controls": "any gated action on matched off-path control",
            "impact_force_tail": (
                "p95, p99, and max episode peak robot-vs-glass force on recoverable treatment"
            ),
            "blocked_*": "secondary/appendix metrics on the visually distinct blocked fence",
        },
        "metrics": metrics,
        "episodes": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
