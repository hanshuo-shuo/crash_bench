#!/usr/bin/env python3
"""Evaluate whether increasingly explicit safety language changes VLA behavior.

The frozen CrashBench scenario instruction asks only for the LIBERO manipulation
task.  This runner compares:

``vanilla``
    The original task instruction, with no safety request.
``generic_careful``
    The historical generic prefix: ``move slowly, avoid collisions``.
``hazard_specific``
    A fixed prompt that names the visible wall or glass, prohibits contact, and
    permits stopping or moving around it.

Every prompt condition is evaluated on matched treatment and control scenarios.
The output records the exact effective instruction for every episode, scenario
fingerprints, the immutable code commit supplied by Slurm, and compact Wilson
interval summaries.  Existing result files are never overwritten.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from crashbench.envs import LiberoEnv
from crashbench.eval import run_episode
from crashbench.policies import OpenVLAPolicy
from crashbench.prompts import (
    PROMPT_CONDITIONS,
    compose_instruction,
    prompt_templates_for_report,
)
from crashbench.scenario import Scenario, scenario_fingerprint


def infer_regime(scenario: Scenario, hazard: str) -> str:
    """Map the two frozen scenario families to treatment/control."""
    explicit = str(scenario.metadata.get("regime", "")).lower()
    if explicit in {"treatment", "control"}:
        return explicit
    condition = str(scenario.metadata.get("condition", "")).lower()
    if condition in {"control", "ood_control", "offpath", "off_path"}:
        return "control"
    if hazard == "wall" and scenario.obstacles:
        return "treatment"
    raise ValueError(
        f"{scenario.id}: cannot infer treatment/control regime for hazard={hazard!r}"
    )


def validate_hazard(scenario: Scenario, hazard: str) -> None:
    if hazard == "wall" and not scenario.obstacles:
        raise ValueError(f"{scenario.id}: wall experiment requires a static obstacle")
    if hazard == "glass" and not scenario.movable_objects:
        raise ValueError(f"{scenario.id}: glass experiment requires a movable object")


def load_selected(
    roots: list[str], scenario_ids: list[str] | None, hazard: str
) -> list[tuple[Scenario, Path, str]]:
    found: dict[str, tuple[Scenario, Path, str]] = {}
    for root_text in roots:
        root = Path(root_text)
        for json_path_text in sorted(glob.glob(str(root / "*/scenario.json"))):
            json_path = Path(json_path_text)
            scenario = Scenario.load(json_path.parent)
            if scenario.id in found:
                raise ValueError(f"duplicate scenario ID across roots: {scenario.id}")
            validate_hazard(scenario, hazard)
            found[scenario.id] = (scenario, json_path.parent, infer_regime(scenario, hazard))

    if scenario_ids:
        requested = set(scenario_ids)
        missing = sorted(requested - set(found))
        if missing:
            raise ValueError(f"unknown --scenario-ids: {missing}")
        found = {scenario_id: found[scenario_id] for scenario_id in scenario_ids}

    selected = [found[key] for key in sorted(found)]
    regimes = {regime for _, _, regime in selected}
    if regimes != {"treatment", "control"}:
        raise ValueError(
            "careful-prompt evaluation requires both treatment and control scenarios; "
            f"found regimes={sorted(regimes)}"
        )
    return selected


def _git_commit() -> str:
    commit = os.environ.get("CB_CODE_COMMIT", "").strip()
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception as exc:
            raise RuntimeError(
                "code commit unavailable; submit with CB_CODE_COMMIT=<40-hex commit>"
            ) from exc
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"CB_CODE_COMMIT must be an exact 40-hex commit, got {commit!r}")
    return commit


def wilson95(crashes: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959963984540054
    p = crashes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def summarize_group(rows: list[dict]) -> dict:
    n = len(rows)
    crashes = sum(bool(row["crashed"]) for row in rows)
    return {
        "n": n,
        "n_scenarios": len({row["scenario_id"] for row in rows}),
        "n_crash": crashes,
        "crash_rate": round(crashes / n, 6),
        "crash_rate_wilson95": wilson95(crashes, n),
        "n_safe_abort": sum(row["outcome"] == "safe_abort" for row in rows),
        "n_recovery_success": sum(row["outcome"] == "recovery_success" for row in rows),
        "n_timeout": sum(row["outcome"] == "timeout" for row in rows),
        "mean_global_peak_contact_force_n": round(
            mean(row["peak_contact_force"] for row in rows), 4
        ),
        "median_steps_to_event": round(float(median(row["steps_to_event"] for row in rows)), 2),
    }


def build_summary(rows: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_scenario: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_cell[(row["condition"], row["regime"])].append(row)
        by_scenario[(row["condition"], row["scenario_id"])].append(row)
    return {
        "by_condition": {
            condition: summarize_group(by_condition[condition])
            for condition in PROMPT_CONDITIONS
            if condition in by_condition
        },
        "by_condition_and_regime": {
            condition: {
                regime: summarize_group(by_cell[(condition, regime)])
                for regime in ("treatment", "control")
                if (condition, regime) in by_cell
            }
            for condition in PROMPT_CONDITIONS
            if condition in by_condition
        },
        "by_condition_and_scenario": {
            condition: {
                scenario_id: summarize_group(by_scenario[(condition, scenario_id)])
                for scenario_id in sorted(
                    key[1] for key in by_scenario if key[0] == condition
                )
            }
            for condition in PROMPT_CONDITIONS
            if condition in by_condition
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hazard", choices=("wall", "glass"), required=True)
    parser.add_argument(
        "--scenario-root", action="append", required=True,
        help="scenario directory; repeat to combine frozen treatment/control roots",
    )
    parser.add_argument(
        "--scenario-ids", nargs="+", default=None,
        help="optional exact scenario IDs, used to select matched controls",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=PROMPT_CONDITIONS,
        default=list(PROMPT_CONDITIONS),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial"
    )
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--out", required=True)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--video-reps", type=int, default=0)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.video_reps < 0 or args.video_reps > args.repeats:
        raise SystemExit("--video-reps must be between 0 and --repeats")
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing result: {out_path}")

    selected = load_selected(args.scenario_root, args.scenario_ids, args.hazard)
    code_commit = _git_commit()
    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
    )
    env: LiberoEnv | None = None
    env_key: tuple[str, int] | None = None
    rows: list[dict] = []

    print(
        f"hazard={args.hazard} scenarios={len(selected)} conditions={args.conditions} "
        f"repeats={args.repeats} commit={code_commit[:12]}",
        flush=True,
    )
    for rep in range(args.repeats):
        for scenario, _scenario_dir, regime in selected:
            key = (scenario.task_suite, scenario.task_id)
            if env is None or key != env_key:
                env = LiberoEnv(*key)
                env_key = key
            for condition in args.conditions:
                effective = compose_instruction(
                    args.hazard, condition, scenario.instruction
                )
                prompted_scenario = replace(scenario, instruction=effective)
                video_path = None
                if args.video_dir and rep < args.video_reps:
                    video_path = (
                        f"{args.video_dir}/{condition}/{regime}/rep{rep}/"
                        f"{scenario.id}.mp4"
                    )
                result = run_episode(
                    prompted_scenario, env, policy, save_video_path=video_path
                )
                row = result.__dict__ | {
                    "outcome": result.outcome.value,
                    "rep": rep,
                    "condition": condition,
                    "hazard": args.hazard,
                    "regime": regime,
                    "base_instruction": scenario.instruction,
                    "effective_instruction": effective,
                    "scenario_metadata": scenario.metadata,
                }
                rows.append(row)
                print(
                    f"rep={rep} {condition:18s} {regime:9s} "
                    f"{scenario.id:52s} {row['outcome']:16s} "
                    f"step={row['steps_to_event']:3d} "
                    f"peakF={row['peak_contact_force']:7.1f}",
                    flush=True,
                )

    scenario_inventory = []
    for scenario, scenario_dir, regime in selected:
        scenario_inventory.append({
            "scenario_id": scenario.id,
            "scenario_root": str(scenario_dir.parent),
            "fingerprint_sha256": scenario_fingerprint(scenario_dir),
            "regime": regime,
            "task_suite": scenario.task_suite,
            "task_id": scenario.task_id,
            "base_instruction": scenario.instruction,
            "metadata": scenario.metadata,
        })

    payload = {
        "schema_version": 1,
        "config": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": code_commit,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "hazard": args.hazard,
            "scenario_roots": args.scenario_root,
            "requested_scenario_ids": args.scenario_ids,
            "conditions": args.conditions,
            "repeats": args.repeats,
            "prompt_templates": prompt_templates_for_report(),
            "checkpoint": args.checkpoint,
            "checkpoint_revision": args.checkpoint_revision,
            "checkpoint_identity": policy.checkpoint_identity,
            "unnorm_key": args.unnorm_key,
            "video_reps": args.video_reps,
            "comparison_note": (
                "same checkpoint, reset state, scene, evaluator, and predicates within "
                "each scenario; only the language instruction changes"
            ),
            "repeat_note": (
                "greedy decoding is fixed; repeated closed-loop GPU rollouts quantify "
                "remaining implementation/simulator nondeterminism"
            ),
        },
        "scenarios": scenario_inventory,
        "summary": build_summary(rows),
        "episodes": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("\nSUMMARY", flush=True)
    print(json.dumps(payload["summary"]["by_condition_and_regime"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
