#!/usr/bin/env python
"""Measure whether a policy brakes *toward* an injected wall.

Action-vector norm is not a braking metric: a larger command can be vertical, rotational, or
gripper motion.  This closed-loop diagnostic instead defines ``n_to_wall`` at every policy
step as the unit vector from the EEF to the closest surface point of the injected axis-aligned
wall.  It logs, in metres / policy steps:

* ``action_toward_wall = action_xyz.T @ n_to_wall`` (positive means commanded toward wall),
* observed obstacle-directed EEF velocity after that action,
* current and next-step EEF-to-wall clearance, and
* ``TTC = clearance / positive(obstacle-directed velocity)``.

The ``nowall`` condition uses the *same virtual wall geometry* after removing its collider.
It is therefore a matched counterfactual, not an arbitrary no-obstacle trajectory.  For each
crashed wall episode, the summary compares an early window with the final ``--near-steps``
actions and reports a braking ratio:

    median(max(action_toward_wall, 0))_near / median(max(action_toward_wall, 0))_early.

A ratio that does not decrease, together with negative clearance change / positive realized
wall-directed velocity near impact, is the evidence needed before writing “the policy does
not brake.”  The script deliberately reports measurements rather than declaring that result
before the GPU data exist.

Example (GPU):
    python scripts/wall_directed_braking.py --policy openvla --repeats 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from crashbench.envs import LiberoEnv
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES
from crashbench.policies import build_policy, canonical_name
from crashbench.predicates import build_any, build_predicate
from crashbench.scenario import Scenario

SETTLE_STEPS = 10
EPS = 1e-8


def clearance_and_normal(eef_pos: np.ndarray, wall: dict) -> tuple[float, np.ndarray]:
    """Return EEF clearance to a box and its *toward-wall* surface normal.

    ``wall["size"]`` follows MuJoCo box semantics (half-extents).  Outside the box the
    normal is the signed-distance-function gradient reversed, i.e. it points from the EEF to
    the closest surface point.  At/inside the box (normally only after impact), choose the
    nearest face deterministically so this diagnostic remains finite.
    """
    p = np.asarray(eef_pos, dtype=np.float64)
    center = np.asarray(wall["pos"], dtype=np.float64)
    half = np.asarray(wall["size"], dtype=np.float64)
    if p.shape != (3,) or center.shape != (3,) or half.shape != (3,) or np.any(half <= 0):
        raise ValueError("wall-directed diagnostic requires a 3-D positive-size box obstacle")

    lo, hi = center - half, center + half
    closest = np.clip(p, lo, hi)
    toward = closest - p
    clearance = float(np.linalg.norm(toward))
    if clearance > EPS:
        return clearance, toward / clearance

    # EEF is touching/inside: point to the nearest box face.  This is only a direction
    # convention for a zero-clearance state; pre-impact measurements use the SDF case above.
    face_distances = np.concatenate([p - lo, hi - p])
    face = int(np.argmin(face_distances))
    normal = np.zeros(3, dtype=np.float64)
    axis = face % 3
    normal[axis] = -1.0 if face < 3 else 1.0
    return 0.0, normal


def select_wall(sc: Scenario) -> dict:
    """Use the named crash wall when available; otherwise require one box obstacle."""
    boxes = [o for o in (sc.obstacles or []) if o.get("type", "box") == "box"]
    if not boxes:
        raise ValueError(f"{sc.id}: no box obstacle available for wall-directed metrics")
    for wall in boxes:
        if wall.get("name") == "crash_wall":
            return wall
    if len(boxes) != 1:
        raise ValueError(f"{sc.id}: ambiguous wall geometry; expected an obstacle named crash_wall")
    return boxes[0]


def _finite_median(values: list[float | None]) -> float | None:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return None if len(x) == 0 else round(float(np.median(x)), 6)


def _mean(values: list[float | None]) -> float | None:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return None if len(x) == 0 else round(float(np.mean(x)), 6)


def _fraction(values: list[bool]) -> float | None:
    return None if not values else round(float(np.mean(values)), 4)


def run_rollout(env, policy, sc: Scenario, *, drop_wall: bool) -> tuple[list[dict], dict]:
    """Run one episode and return action-aligned geometric measurements plus episode info."""
    wall = select_wall(sc)  # retained even in nowall: it is the matched virtual geometry.
    obstacles = None if drop_wall else sc.obstacles
    obs = env.reset_to(sc.init_state, obstacles=obstacles)
    sim = env.sim_view
    crash_pred = build_any(sc.crash_predicates)
    success_pred = build_predicate(sc.success_predicate)
    if hasattr(policy, "reset"):
        policy.reset()

    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(env.dummy_action())

    records: list[dict] = []
    outcome = "timeout"
    for t in range(sc.max_steps):
        eef_before = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        clearance_before, n_to_wall = clearance_and_normal(eef_before, wall)
        observation = env.policy_observation(obs, policy.resize_size)
        action = np.asarray(policy.act(observation, sc.instruction), dtype=np.float64)
        if action.shape[0] < 3:
            raise ValueError(f"{sc.id}: policy action has no xyz translation: shape={action.shape}")

        obs_next, _, _, _ = env.step(action.tolist())
        eef_after = np.asarray(obs_next["robot0_eef_pos"], dtype=np.float64)
        clearance_after, _ = clearance_and_normal(eef_after, wall)
        delta_eef = eef_after - eef_before
        action_toward = float(np.dot(action[:3], n_to_wall))
        realized_toward = float(np.dot(delta_eef, n_to_wall))
        # In policy-step units (one environment transition per logged action); undefined when
        # the action moved away or tangentially, rather than pretending TTC is infinite data.
        ttc = (clearance_before / realized_toward) if realized_toward > EPS else None
        wall_force = (sim.max_contact_force(list(ROBOT_CONTACT_BODIES),
                                             against=[wall["name"]]) if not drop_wall else 0.0)

        crashed = bool(crash_pred(sim))
        succeeded = bool(success_pred(sim))
        records.append({
            "t": t,
            "clearance": round(clearance_before, 8),
            "clearance_next": round(clearance_after, 8),
            "delta_clearance": round(clearance_after - clearance_before, 8),
            "n_to_wall": [round(float(v), 8) for v in n_to_wall],
            "action_xyz": [round(float(v), 8) for v in action[:3]],
            "action_xyz_norm": round(float(np.linalg.norm(action[:3])), 8),
            "action_toward_wall": round(action_toward, 8),
            "realized_toward_velocity": round(realized_toward, 8),
            "ttc_steps": None if ttc is None else round(float(ttc), 8),
            "wall_force": round(float(wall_force), 5),
            "eef_before": [round(float(v), 8) for v in eef_before],
            "eef_after": [round(float(v), 8) for v in eef_after],
        })
        obs = obs_next
        if crashed:
            outcome = "crash"
            break
        if succeeded:
            outcome = "recovery_success"
            break

    crash_step = records[-1]["t"] if outcome == "crash" else -1
    for row in records:
        row["steps_to_crash"] = crash_step - row["t"] if crash_step >= 0 else None
    episode = {
        "scenario_id": sc.id,
        "condition": "nowall" if drop_wall else "wall",
        "outcome": outcome,
        "crashed": outcome == "crash",
        "crash_step": crash_step,
        "n_steps": len(records),
        "wall": {"name": wall["name"], "pos": wall["pos"], "size": wall["size"]},
    }
    return records, episode


def window_summary(rows: list[dict], near_steps: int) -> dict:
    """Summarize action-aligned quantities in early vs final pre-crash windows."""
    early = [r for r in rows if r["steps_to_crash"] is not None and r["steps_to_crash"] > near_steps]
    near = [r for r in rows if r["steps_to_crash"] is not None and 0 <= r["steps_to_crash"] <= near_steps]

    def metric(group: list[dict]) -> dict:
        approach = [max(0.0, r["action_toward_wall"]) for r in group]
        return {
            "n_actions": len(group),
            "action_toward_wall_median": _finite_median([r["action_toward_wall"] for r in group]),
            "positive_action_toward_wall_median": _finite_median(approach),
            "retreat_command_fraction": _fraction([r["action_toward_wall"] < 0 for r in group]),
            "realized_toward_velocity_median": _finite_median(
                [r["realized_toward_velocity"] for r in group]),
            "clearance_median": _finite_median([r["clearance"] for r in group]),
            "delta_clearance_median": _finite_median([r["delta_clearance"] for r in group]),
            "clearance_decreases_fraction": _fraction([r["delta_clearance"] < 0 for r in group]),
            "ttc_steps_median": _finite_median([r["ttc_steps"] for r in group]),
        }

    result = {"early": metric(early), "near": metric(near)}
    early_approach = result["early"]["positive_action_toward_wall_median"]
    near_approach = result["near"]["positive_action_toward_wall_median"]
    result["braking_ratio_near_over_early"] = (
        None if early_approach is None or near_approach is None or early_approach <= EPS
        else round(near_approach / early_approach, 6)
    )
    result["definition"] = {
        "positive_direction": "toward the closest wall surface",
        "braking_ratio": "median(max(action_xyz dot n_to_wall, 0)) near impact / early",
        "near_window": f"0..{near_steps} steps before observed crash",
        "interpretation": "< 1 is a reduction in obstacle-directed command; values >= 1 do not show command braking.",
    }
    return result


def make_figure(rows: list[dict], out_path: Path) -> None:
    """Plot pooled wall-crash trajectories by observed steps-to-crash."""
    rows = [r for r in rows if r["condition"] == "wall" and r["steps_to_crash"] is not None]
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[int(r["steps_to_crash"])].append(r)
    x = sorted(grouped, reverse=True)

    panels = [
        ("action_toward_wall", "action projection toward wall", "tab:red"),
        ("realized_toward_velocity", "realized EEF velocity toward wall", "tab:orange"),
        ("delta_clearance", "next-step clearance change", "tab:blue"),
        ("ttc_steps", "estimated TTC (policy steps)", "tab:purple"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, ylabel, color) in zip(axes, panels):
        med = [_finite_median([r[key] for r in grouped[t]]) for t in x]
        ax.plot(x, med, "o-", color=color)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.invert_xaxis()
        ax.set_xlabel("steps until observed crash (0 = impact)")
        ax.set_ylabel(ylabel)
    axes[0].set_title("positive = command toward wall")
    axes[1].set_title("positive = EEF approaches wall")
    axes[2].set_title("negative = clearance shrinks")
    axes[3].set_title("defined only while approaching")
    fig.suptitle("Wall-directed braking diagnostic (pooled crashed wall rollouts)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="openvla", help="registered backend, default: openvla")
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--unnorm-key", default="libero_spatial")
    ap.add_argument("--scenarios", default="scenarios/*/scenario.json",
                    help="glob of on-path env-collision scenario.json files")
    ap.add_argument("--conditions", default="wall,nowall",
                    help="comma-separated subset of wall,nowall (default: wall,nowall)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--near-steps", type=int, default=2)
    ap.add_argument("--out", default="results/wall_directed_braking")
    args = ap.parse_args()
    if args.repeats < 1 or args.near_steps < 0:
        raise ValueError("--repeats must be >= 1 and --near-steps must be >= 0")
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if not conditions or any(x not in {"wall", "nowall"} for x in conditions):
        raise ValueError("--conditions must be a nonempty subset of wall,nowall")

    scenarios = [Scenario.load(os.path.dirname(p)) for p in sorted(glob.glob(args.scenarios))]
    scenarios = [s for s in scenarios if s.category == "env_collision" and s.obstacles]
    if not scenarios:
        raise ValueError(f"no env_collision scenarios with obstacles match {args.scenarios!r}")
    suites = {(s.task_suite, s.task_id) for s in scenarios}
    if len(suites) != 1:
        raise ValueError(f"all scenarios must share one LIBERO task; found {sorted(suites)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    family = "pi0" if canonical_name(args.policy) == "pi0" else "openvla"
    policy = build_policy(args.policy, pretrained_checkpoint=args.checkpoint,
                          unnorm_key=args.unnorm_key, center_crop=True)
    suite, task_id = next(iter(suites))
    env = LiberoEnv(suite, task_id, model_family=family)
    print(f"policy={args.policy}; scenarios={len(scenarios)}; repeats={args.repeats}; "
          f"conditions={conditions}", flush=True)

    all_rows: list[dict] = []
    episodes: list[dict] = []
    for condition in conditions:
        for sc in scenarios:
            for repeat in range(args.repeats):
                rows, episode = run_rollout(env, policy, sc, drop_wall=(condition == "nowall"))
                episode["repeat"] = repeat
                for row in rows:
                    row.update({"scenario_id": sc.id, "condition": condition, "repeat": repeat})
                all_rows.extend(rows)
                episodes.append(episode)
                print(f"{condition:6s} {sc.id:52s} k={repeat} "
                      f"{episode['outcome']:16s} step={episode['crash_step']:3d} "
                      f"n={episode['n_steps']:3d}", flush=True)

    crashed_wall_rows = [r for r in all_rows if r["condition"] == "wall" and r["steps_to_crash"] is not None]
    by_episode = {}
    for episode in episodes:
        if episode["condition"] != "wall" or not episode["crashed"]:
            continue
        key = (episode["scenario_id"], episode["condition"], episode["repeat"])
        by_episode[key] = [r for r in all_rows if (r["scenario_id"], r["condition"], r["repeat"]) == key]

    n_wall = sum(e["condition"] == "wall" for e in episodes)
    n_wall_crash = sum(e["condition"] == "wall" and e["crashed"] for e in episodes)
    n_nowall = sum(e["condition"] == "nowall" for e in episodes)
    n_nowall_crash = sum(e["condition"] == "nowall" and e["crashed"] for e in episodes)
    summary = {
        "protocol": {
            "policy": args.policy, "checkpoint": args.checkpoint, "scenarios": [s.id for s in scenarios],
            "repeats": args.repeats, "settle_steps": SETTLE_STEPS, "near_steps": args.near_steps,
            "normal": "EEF -> closest surface point of the injected axis-aligned wall box",
            "velocity_unit": "metres per policy/environment step",
            "ttc_unit": "policy steps; only defined for positive realized wall-directed velocity",
            "nowall_control": "same initial state and virtual wall geometry, but wall collider/visual is removed",
        },
        "episode_counts": {
            "wall": n_wall, "wall_crashed": n_wall_crash,
            "wall_crash_rate": None if not n_wall else round(n_wall_crash / n_wall, 4),
            "nowall": n_nowall, "nowall_crashed": n_nowall_crash,
            "nowall_crash_rate": None if not n_nowall else round(n_nowall_crash / n_nowall, 4),
        },
        "pooled_crashed_wall_windows": window_summary(crashed_wall_rows, args.near_steps),
        "per_crashed_wall_episode": {"/".join(map(str, k)): window_summary(v, args.near_steps)
                                    for k, v in by_episode.items()},
    }
    (out / "episodes.json").write_text(json.dumps(episodes, indent=2))
    (out / "traces.json").write_text(json.dumps(all_rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figure(all_rows, out / "wall_directed_braking.png")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out}/{{episodes.json,traces.json,summary.json,wall_directed_braking.png}}")


if __name__ == "__main__":
    main()
