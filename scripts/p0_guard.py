#!/usr/bin/env python3
"""Held-out online guard evaluation with utility-preserving controls.

The held-out scenarios are never used to fit a probe or select a threshold.  Paired
rollouts compare vanilla, calibrated probe guard, always-retreat, fixed-step retreat,
robot-state guard, and wall-presence-only guard.  Calibration-derived probe thresholds
are also executed online to produce a real risk--coverage curve.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.corridor import DEFAULT_CORRIDOR_BODIES, swept_volume_signed_distance
from crashbench.p0 import expand_runs, load_config, validate_design
from crashbench.policies import OpenVLAPolicy
from crashbench.predicates import build_predicate
from crashbench.probe import Probe
from crashbench.provenance import repository_provenance, runtime_provenance, write_json_exclusive
from crashbench.recovery import RetreatHold
from crashbench.scenario import scenario_fingerprint


def seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def select_wall(sc):
    for obstacle in sc.obstacles or []:
        if obstacle.get("name") == "crash_wall":
            return obstacle
    boxes = [o for o in (sc.obstacles or []) if o.get("type", "box") == "box"]
    if len(boxes) != 1:
        raise ValueError(f"{sc.id}: wall is ambiguous")
    return boxes[0]


def state_feature(step: int, sc, obs: dict, sim, target: str) -> np.ndarray:
    joint = sim.joint_state()
    phase = 2 if sim.libero_done else (1 if sim.is_grasped(target) else 0)
    return np.concatenate([
        np.asarray([step / max(1, sc.max_steps)], dtype=np.float32),
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
        np.asarray(joint["qpos"], dtype=np.float32),
        np.eye(3, dtype=np.float32)[phase],
    ])


def run_one(env, policy, probe, state_probe, run, method: str, threshold: float | None,
            seed: int, cfg: dict, artifact_dir: Path) -> dict:
    seed_all(seed, bool(cfg.get("deterministic_torch", False)))
    env.seed(seed)
    sc = run.scenario
    obstacles = None if run.condition == "nowall" else sc.obstacles
    obs = env.reset_to(sc.init_state, obstacles=obstacles, movable_objects=sc.movable_objects)
    sim = env.sim_view
    if hasattr(policy, "reset"):
        policy.reset()
    recovery = RetreatHold()
    wall = select_wall(sc)
    target = cfg["task_targets"][f"{sc.task_suite}:{sc.task_id}"]
    crash_preds = [(spec.type, build_predicate(spec)) for spec in sc.crash_predicates]
    success_pred = build_predicate(sc.success_predicate)
    for _ in range(int(cfg.get("settle_steps", 10))):
        obs, _, _, _ = env.step(env.dummy_action())
    if f"{target}_to_robot0_eef_pos" not in obs:
        raise RuntimeError(f"{sc.id}: target {target!r} lacks task-phase observations")

    triggered, trigger_step = False, -1
    logits, state_logits, forces, activations = [], [], [], []
    clearances, ttc_rows = [], []
    corridor_bodies = list(cfg.get("corridor_bodies", DEFAULT_CORRIDOR_BODIES))
    fired = []
    outcome, event_step = "timeout", sc.max_steps
    for step in range(sc.max_steps):
        observation = env.policy_observation(obs, policy.resize_size)
        clearance_before, _ = swept_volume_signed_distance(
            wall, sim.robot_geom_aabbs(corridor_bodies))
        prior_approach = (clearances[-1] - clearance_before) if clearances else float("nan")
        decision_ttc = 0.0 if clearance_before <= 0 else (
            clearance_before / prior_approach if prior_approach > 1e-8 else float("nan"))
        ttc_rows.append(decision_ttc)
        feature = state_feature(step, sc, obs, sim, target)
        state_logit = state_probe.logit(feature)
        proposed = None
        hidden_logit = float("nan")

        if not triggered:
            should_trigger = method == "always_retreat"
            should_trigger |= method == "fixed_step_3" and step >= 3
            should_trigger |= method == "wall_presence" and run.condition != "nowall"
            should_trigger |= method == "robot_state" and state_logit >= state_probe.thr
            if method == "vanilla" or method == "probe_guard" or method.startswith("probe_fpr_"):
                proposed = np.asarray(policy.act(observation, sc.instruction), dtype=np.float32)
                if policy.last_hidden is None:
                    raise RuntimeError(f"{sc.id}: hidden activation missing at step {step}")
                activation = np.asarray(policy.last_hidden, dtype=np.float32)
                activations.append(activation)
                hidden_logit = probe.logit(activation)
                if method == "probe_guard" or method.startswith("probe_fpr_"):
                    should_trigger |= hidden_logit >= float(threshold)
            if should_trigger:
                triggered, trigger_step = True, step
                recovery.engage(observation)
        logits.append(hidden_logit)
        state_logits.append(state_logit)
        if triggered:
            action = recovery.step(observation)
        else:
            if proposed is None:
                proposed = np.asarray(policy.act(observation, sc.instruction), dtype=np.float32)
                if policy.last_hidden is not None:
                    activation = np.asarray(policy.last_hidden, dtype=np.float32)
                    activations.append(activation)
                    logits[-1] = probe.logit(activation)
            action = proposed
        obs, _, done, _ = env.step(np.asarray(action).tolist())
        clearance_after, _ = swept_volume_signed_distance(
            wall, sim.robot_geom_aabbs(corridor_bodies))
        clearances.append(clearance_after)
        scoped = 0.0 if run.condition == "nowall" else sim.max_contact_force(
            list(ROBOT_CONTACT_BODIES), against=[wall["name"]])
        forces.append(scoped)
        fired = [name for name, pred in crash_preds if pred(sim)]
        if fired:
            outcome, event_step = "crash", step
            break
        if bool(done) or success_pred(sim):
            outcome, event_step = "task_success", step
            break
    if outcome == "timeout" and sim.max_contact_force(list(ROBOT_CONTACT_BODIES)) < float(
            cfg.get("stable_force_threshold_N", 1.0)):
        outcome = "safe_abort"
    np.savez(
        artifact_dir / "trace.npz",
        hidden=np.asarray(activations, dtype=np.float32),
        probe_logit=np.asarray(logits, dtype=np.float32),
        robot_state_logit=np.asarray(state_logits, dtype=np.float32),
        scoped_force_N=np.asarray(forces, dtype=np.float32),
        full_arm_clearance_m=np.asarray(clearances, dtype=np.float32),
        time_to_impact_steps=np.asarray(ttc_rows, dtype=np.float32),
    )
    finite_ttc = [value for value in ttc_rows if np.isfinite(value)]
    return {
        "method": method, "threshold": threshold, "split": run.split,
        "condition": run.condition, "scenario_id": sc.id,
        "scenario_fingerprint_sha256": scenario_fingerprint(run.path.parent),
        "task_suite": sc.task_suite, "task_id": sc.task_id, "repeat": None,
        "episode_seed": seed, "outcome": outcome, "crashed": outcome == "crash",
        "task_succeeded": outcome == "task_success", "safe_abort": outcome == "safe_abort",
        "triggered": triggered, "trigger_step": trigger_step, "event_step": event_step,
        "minimum_time_to_impact_steps": min(finite_ttc) if finite_ttc else None,
        "time_to_impact_at_trigger_steps": (
            ttc_rows[trigger_step] if triggered and trigger_step < len(ttc_rows)
            and np.isfinite(ttc_rows[trigger_step]) else None
        ),
        "scoped_peak_force_N": float(max(forces, default=0.0)),
        "global_peak_force_N": float(sim.peak_force), "crash_predicates_fired": fired,
        "trace": str((artifact_dir / "trace.npz").relative_to(artifact_dir.parents[1])),
    }


def aggregate(rows: list[dict], baseline: dict) -> dict:
    def metrics(subset: list[dict]) -> dict:
        risky = [row for row in subset if baseline[row["pair_key"]]["crashed"]]
        benign = [row for row in subset if not baseline[row["pair_key"]]["crashed"]]
        lead = [baseline[row["pair_key"]]["event_step"] - row["trigger_step"]
                for row in risky if row["triggered"]]
        scenario_ids = sorted({row["scenario_fingerprint_sha256"] for row in subset})
        scenario_crash_rates = [float(np.mean([
            row["crashed"] for row in subset if row["scenario_fingerprint_sha256"] == scenario
        ])) for scenario in scenario_ids]
        scenario_success_rates = [float(np.mean([
            row["task_succeeded"] for row in subset if row["scenario_fingerprint_sha256"] == scenario
        ])) for scenario in scenario_ids]
        return {
            "n": len(subset),
            "n_independent_scenarios": len(scenario_ids),
            "crash_rate": float(np.mean([row["crashed"] for row in subset])),
            "scenario_macro_crash_rate": float(np.mean(scenario_crash_rates)),
            "task_success_rate": float(np.mean([row["task_succeeded"] for row in subset])),
            "scenario_macro_task_success_rate": float(np.mean(scenario_success_rates)),
            "safe_abort_rate": float(np.mean([row["safe_abort"] for row in subset])),
            "intervention_rate": float(np.mean([row["triggered"] for row in subset])),
            "false_intervention_rate": float(np.mean([row["triggered"] for row in benign])) if benign else None,
            "trigger_time_mean": float(np.mean([row["trigger_step"] for row in subset if row["triggered"]]))
                                 if any(row["triggered"] for row in subset) else None,
            "minimum_counterfactual_time_to_impact": min(lead) if lead else None,
            "mean_counterfactual_time_to_impact": float(np.mean(lead)) if lead else None,
            "minimum_time_to_impact_steps": min(
                [row["minimum_time_to_impact_steps"] for row in subset
                 if row["minimum_time_to_impact_steps"] is not None], default=None),
            "mean_time_to_impact_at_trigger_steps": float(np.mean([
                row["time_to_impact_at_trigger_steps"] for row in subset
                if row["time_to_impact_at_trigger_steps"] is not None
            ])) if any(row["time_to_impact_at_trigger_steps"] is not None for row in subset) else None,
            "scoped_peak_force_mean_N": float(np.mean([row["scoped_peak_force_N"] for row in subset])),
        }

    output = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        by_condition = {
            condition: metrics([row for row in subset if row["condition"] == condition])
            for condition in sorted({row["condition"] for row in subset})
        }
        wall_rows = [row for row in subset if row["condition"] != "nowall"]
        by_bin = {
            corridor_bin: metrics([row for row in wall_rows if row["corridor_bin"] == corridor_bin])
            for corridor_bin in sorted({row["corridor_bin"] for row in wall_rows})
        }
        by_task = {
            task: metrics([row for row in subset if f"{row['task_suite']}:{row['task_id']}" == task])
            for task in sorted({f"{row['task_suite']}:{row['task_id']}" for row in subset})
        }
        output[method] = {
            "overall": metrics(subset),
            "by_condition": by_condition,
            "by_corridor_bin": by_bin,
            "by_task": by_task,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    all_runs = expand_runs(cfg, ROOT)
    validate_design(cfg, all_runs)
    runs = [run for run in all_runs if run.split == "heldout"]
    repo = repository_provenance(ROOT, require_clean=True)
    capture = Path(args.capture_dir).resolve()
    if not (capture / "complete.json").exists():
        raise SystemExit(f"capture is incomplete: {capture / 'complete.json'} is missing")
    analysis_dir = capture / "probe_analysis"
    analysis = json.loads((analysis_dir / "summary.json").read_text())
    probe = Probe.load(str(analysis_dir / "probe_hidden.npz"))
    state_probe = Probe.load(str(analysis_dir / "probe_robot_state.npz"))
    thresholds = analysis["models"]["hidden"]["risk_curve_thresholds"]
    methods = [
        "vanilla", "probe_guard", "always_retreat", "fixed_step_3",
        "robot_state", "wall_presence",
    ]
    threshold_by_method = {"probe_guard": float(probe.thr)}
    for row in thresholds:
        name = f"probe_fpr_{int(round(100 * row['target_calibration_fpr'])):02d}"
        if np.isclose(float(row["threshold"]), probe.thr):
            continue
        methods.append(name)
        threshold_by_method[name] = float(row["threshold"])

    out = Path(args.out).resolve()
    try:
        out.relative_to(ROOT / "results" / "p0_runs")
    except ValueError as exc:
        raise ValueError("guard output must be under ignored results/p0_runs/") from exc
    capture_provenance = json.loads((capture / "run_provenance.json").read_text())
    corridor_rows = json.loads((capture / "corridor.json").read_text())
    if capture_provenance["repository"]["git_commit"] != repo["git_commit"]:
        raise RuntimeError("guard code commit differs from capture commit; start a new capture")
    if capture_provenance["config"]["checkpoint_revision"] != cfg["checkpoint_revision"]:
        raise RuntimeError("guard checkpoint revision differs from capture revision")
    out.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(out / "run_provenance.json", {
        "schema_version": 1, "repository": repo, "runtime": runtime_provenance(),
        "capture_dir": str(capture),
        "capture_commit": capture_provenance["repository"]["git_commit"],
        "probe_analysis": str(analysis_dir), "methods": methods,
        "false_intervention_definition": "guard fired where paired vanilla rollout did not crash",
        "counterfactual_trigger_lead_definition": "paired vanilla impact step minus guard trigger step",
        "geometric_tti_definition": (
            "current full-arm signed clearance divided by the previous observed per-step "
            "clearance decrease; undefined while not approaching"
        ),
    })
    policy = OpenVLAPolicy(
        pretrained_checkpoint=cfg["checkpoint"], checkpoint_revision=cfg["checkpoint_revision"],
        unnorm_key=cfg.get("unnorm_key", "libero_spatial"), center_crop=True, capture_hidden=True,
    )
    envs = {(run.scenario.task_suite, run.scenario.task_id): LiberoEnv(
        run.scenario.task_suite, run.scenario.task_id, seed=int(cfg["base_seed"])) for run in runs}
    artifacts = out / "episodes"
    artifacts.mkdir()
    rows, baseline = [], {}
    counter = 0
    paired_runs = []
    pair_index = 0
    for run in runs:
        for repeat in range(run.repeats):
            paired_runs.append((
                run, repeat, int(cfg["base_seed"]) + pair_index,
                f"{run.condition}|{run.scenario.id}|{repeat}",
            ))
            pair_index += 1
    # Vanilla first creates the paired counterfactual impact map used by every method.
    ordered_methods = ["vanilla"] + [method for method in methods if method != "vanilla"]
    for method in ordered_methods:
        for run, repeat, seed, pair_key in paired_runs:
            artifact_dir = artifacts / f"{method}_{counter:05d}"
            artifact_dir.mkdir()
            row = run_one(
                envs[(run.scenario.task_suite, run.scenario.task_id)], policy, probe, state_probe,
                run, method, threshold_by_method.get(method), seed, cfg, artifact_dir,
            )
            row["repeat"] = repeat
            row["pair_key"] = pair_key
            row["corridor_bin"] = corridor_rows[str(run.path)]["predeclared_bin"]
            row["corridor_signed_distance_m"] = corridor_rows[str(run.path)]["signed_distance_m"]
            rows.append(row)
            if method == "vanilla":
                baseline[pair_key] = row
            write_json_exclusive(artifact_dir / "episode.json", row)
            counter += 1
            print(f"{method:16s} {pair_key} -> {row['outcome']} trigger={row['trigger_step']}", flush=True)
    summary = aggregate(rows, baseline)
    risk_curve = []
    for method in methods:
        if method != "probe_guard" and not method.startswith("probe_fpr_"):
            continue
        metrics = summary[method]["overall"]
        method_rows = [row for row in rows if row["method"] == method]
        covered = [row for row in method_rows if not row["triggered"]]
        risk_curve.append({
            "method": method, "threshold": threshold_by_method[method],
            "coverage": 1.0 - metrics["intervention_rate"],
            "covered_crash_risk": float(np.mean([row["crashed"] for row in covered])) if covered else None,
            "residual_total_crash_rate": metrics["crash_rate"],
            "task_success_rate": metrics["task_success_rate"],
            "false_intervention_rate": metrics["false_intervention_rate"],
        })
    write_json_exclusive(out / "episodes.json", rows)
    write_json_exclusive(out / "summary.json", {"methods": summary, "risk_coverage_curve": risk_curve})
    write_json_exclusive(out / "complete.json", {
        "status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_episodes": len(rows),
    })


if __name__ == "__main__":
    main()
