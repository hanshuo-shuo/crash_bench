#!/usr/bin/env python3
"""Run the fresh matched online branch cohort for the frozen outcome router.

Each accepted source placement contributes glass, off-path, and no-glass
episodes.  Base establishes the actual on-path collision time; all structured
methods branch from the exact matched T-20 state, while the fixed hazard prompt
runs independently from the same episode reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crashbench.counterfactual_router import (
    FrozenOutcomeRouter,
    OPTIONS,
    classify_option_outcome,
    validate_decision_rows,
)
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import (
    array_sha256,
    canonical_sha256,
    read_placement_manifest,
)
from crashbench.predicates import build_any
from crashbench.prompts import compose_instruction
from crashbench.provenance import repository_provenance, require_checkpoint_revision
from crashbench.recovery import RetreatHold
from scripts.capture_glass_detector_placements import _condition_glasses, _seed_everything
from scripts.collect_counterfactual_option_rollouts import (
    CONDITIONS,
    _apply_frozen_detour_config,
    _branch_start_catastrophic,
    _detour_controller,
    _expected_hashes,
    _restore_anchor,
    _run_base_continue,
    _run_structured_option,
    _scan_condition,
)
from scripts.collect_glass_recovery_pairs import (
    CandidateRejected,
    _glass_force,
    _glass_predicate_specs,
    _prime_glass_predicates,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _run_prompt_from_reset(
    env: LiberoEnv,
    policy,
    source_state: np.ndarray,
    placement,
    condition: str,
    *,
    settle_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    glasses = _condition_glasses(placement, condition)
    obs = env.reset_to(source_state, movable_objects=glasses or None)
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())
    crash = None
    if glasses:
        crash = build_any(_glass_predicate_specs(glasses))
        _prime_glass_predicates(crash, env.sim_view)
    instruction = compose_instruction("glass", "hazard_specific", placement.instruction)
    policy.reset()
    peak_force = 0.0
    for step in range(max_steps):
        if env.episode_terminated():
            return {
                "crashed": False, "succeeded": False, "steps": step,
                "peak_force_n": float(peak_force),
                "termination": "robosuite_episode_horizon",
                "effective_instruction": instruction,
            }
        policy_obs = env.policy_observation(obs, policy.resize_size)
        action = np.asarray(policy.act(policy_obs, instruction), dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())
        force = 0.0 if not glasses else _glass_force(env.sim_view, glasses)
        peak_force = max(peak_force, force)
        if crash is not None and crash(env.sim_view):
            return {
                "crashed": True, "succeeded": False, "steps": step + 1,
                "peak_force_n": float(peak_force),
                "effective_instruction": instruction,
            }
        if done:
            return {
                "crashed": False, "succeeded": True, "steps": step + 1,
                "peak_force_n": float(peak_force),
                "effective_instruction": instruction,
            }
    return {
        "crashed": False, "succeeded": False, "steps": max_steps,
        "peak_force_n": float(peak_force),
        "effective_instruction": instruction,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    frozen_detour = _apply_frozen_detour_config(args)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    repo_root = Path(__file__).resolve().parents[1]
    repo = repository_provenance(repo_root, require_clean=True)
    declared = os.environ.get("CB_CODE_COMMIT")
    if declared is not None and declared != repo["git_commit"]:
        raise SystemExit("CB_CODE_COMMIT does not match checked-out source")
    revision = require_checkpoint_revision(args.checkpoint_revision)
    placements_path = Path(args.placements).resolve()
    placements, placement_payload = read_placement_manifest(placements_path)
    placements = placements[args.skip_placements:]
    if args.max_placements is not None:
        placements = placements[:args.max_placements]
    router_path = Path(args.router_model).resolve()
    router = FrozenOutcomeRouter.load(router_path)
    exposed_sources = set(router.manifest["all_capture_source_state_sha256"])
    overlap = sorted(
        exposed_sources & {placement.source_state_sha256 for placement in placements}
    )
    if overlap:
        raise RuntimeError(f"fresh source cohort overlaps router capture: {overlap}")

    from crashbench.policies import OpenVLAPolicy
    policy = OpenVLAPolicy(
        pretrained_checkpoint=args.checkpoint,
        checkpoint_revision=revision,
        unnorm_key=args.unnorm_key,
        center_crop=True,
        capture_hidden=True,
    )
    envs: dict[tuple[str, int], LiberoEnv] = {}
    decision_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    valid_placements = 0
    attempted_placements = 0

    for placement in placements:
        if (
            args.target_valid_placements is not None
            and valid_placements >= args.target_valid_placements
        ):
            break
        attempted_placements += 1
        source_path = placements_path.parent / placement.source_state_path
        source_state = np.load(source_path, allow_pickle=False)
        if array_sha256(source_state) != placement.source_state_sha256:
            raise ValueError(f"{placement.placement_id}: source-state hash mismatch")
        env_key = (placement.task_suite, int(placement.task_id))
        if env_key not in envs:
            envs[env_key] = LiberoEnv(*env_key, seed=args.rollout_seed)
        env = envs[env_key]
        scans = {}
        try:
            for condition in CONDITIONS:
                _seed_everything(args.rollout_seed)
                env.seed(args.rollout_seed)
                policy.reset()
                scans[condition] = _scan_condition(
                    env, policy, source_state, placement, condition,
                    settle_steps=args.settle_steps, max_steps=args.scan_steps,
                )
        except CandidateRejected as exc:
            exclusion = {
                "placement_id": placement.placement_id,
                "source_state_sha256": placement.source_state_sha256,
                "reason": f"scan_{exc.reason}",
            }
            exclusions.append(exclusion)
            _append_jsonl(output / "progress.jsonl", exclusion)
            continue
        glass_scan = scans["glass"]
        if not glass_scan["crashed"]:
            exclusion = {
                "placement_id": placement.placement_id,
                "source_state_sha256": placement.source_state_sha256,
                "reason": "onpath_no_catastrophe",
            }
            exclusions.append(exclusion)
            _append_jsonl(output / "progress.jsonl", exclusion)
            continue
        collision_step = int(glass_scan["collision_step"])
        anchor_index = collision_step - int(args.horizon) + 1
        if anchor_index < 0:
            exclusion = {
                "placement_id": placement.placement_id,
                "source_state_sha256": placement.source_state_sha256,
                "reason": "catastrophe_before_horizon",
                "collision_step": collision_step,
            }
            exclusions.append(exclusion)
            _append_jsonl(output / "progress.jsonl", exclusion)
            continue
        expected_by_condition = {}
        invalid_condition = None
        for condition in CONDITIONS:
            scan = scans[condition]
            if anchor_index >= len(scan["states"]):
                invalid_condition = f"{condition}_terminal_before_matched_anchor"
                break
            expected = _expected_hashes(scan, anchor_index)
            if _branch_start_catastrophic(
                env, scan, anchor_index, expected,
                label=f"{placement.placement_id}:{condition}:preflight",
            ):
                invalid_condition = f"{condition}_branch_start_catastrophic"
                break
            expected_by_condition[condition] = expected
        if invalid_condition is not None:
            exclusion = {
                "placement_id": placement.placement_id,
                "source_state_sha256": placement.source_state_sha256,
                "reason": invalid_condition,
            }
            exclusions.append(exclusion)
            _append_jsonl(output / "progress.jsonl", exclusion)
            continue

        placement_decisions = []
        placement_options = []
        placement_prompts = []
        for condition in CONDITIONS:
            scan = scans[condition]
            expected = expected_by_condition[condition]
            feature = scan["rows"][anchor_index]
            prediction = router.predict(
                feature["hidden"], feature["robot_state"], feature["nominal_action"]
            )
            decision_id = canonical_sha256({
                "source_state_sha256": placement.source_state_sha256,
                "placement_id": placement.placement_id,
                "condition": condition,
                "horizon_actions": int(args.horizon),
                "anchor_state_sha256": expected["simulator_state_sha256"],
            })
            decision = {
                "decision_id": decision_id,
                "placement_id": placement.placement_id,
                "author_split": placement.split,
                "split": "fresh_evaluation",
                "source_state_sha256": placement.source_state_sha256,
                "source_state_index": placement.metadata.get("source_state_index"),
                "condition": condition,
                "horizon_actions": int(args.horizon),
                "matched_scan_index": int(anchor_index),
                "onpath_collision_step": collision_step,
                "episode_seed": int(args.rollout_seed),
                "branch_start_hashes": expected,
                "option_outcome_probabilities": np.asarray(
                    prediction["option_outcome_probabilities"]
                ).tolist(),
                "base_catastrophe_probability": float(
                    prediction["base_catastrophe_probability"]
                ),
            }
            placement_decisions.append(decision)

            for option in OPTIONS:
                _seed_everything(args.rollout_seed)
                env.seed(args.rollout_seed)
                obs = _restore_anchor(
                    env, scan, anchor_index, expected,
                    label=f"{decision_id}:{option}",
                )
                if option == "base_continue":
                    policy.reset()
                    result = _run_base_continue(
                        env, policy, obs, placement.instruction,
                        list(scan["glasses"]), max_steps=args.base_steps,
                    )
                elif option == "detour_complete":
                    controller_glass = (
                        placement.on_path_glass if condition == "noglass"
                        else scan["glasses"][0]
                    )
                    result = _run_structured_option(
                        env, obs, list(scan["glasses"]),
                        _detour_controller(obs, controller_glass, args),
                        max_steps=args.detour_steps,
                    )
                else:
                    result = _run_structured_option(
                        env, obs, list(scan["glasses"]),
                        RetreatHold(back=args.retreat_back, up=args.retreat_up),
                        max_steps=args.retreat_steps,
                    )
                placement_options.append({
                    **{key: decision[key] for key in (
                        "decision_id", "placement_id", "split",
                        "source_state_sha256", "condition", "horizon_actions",
                        "episode_seed",
                    )},
                    "option": option,
                    "outcome": classify_option_outcome(
                        crashed=bool(result["crashed"]),
                        succeeded=bool(result["succeeded"]),
                    ),
                    **result,
                })

            _seed_everything(args.rollout_seed)
            env.seed(args.rollout_seed)
            prompt_result = _run_prompt_from_reset(
                env, policy, source_state, placement, condition,
                settle_steps=args.settle_steps, max_steps=args.scan_steps,
            )
            placement_prompts.append({
                **{key: decision[key] for key in (
                    "decision_id", "placement_id", "split",
                    "source_state_sha256", "condition", "episode_seed",
                )},
                "method": "hazard_specific_prompt_from_reset",
                "outcome": classify_option_outcome(
                    crashed=bool(prompt_result["crashed"]),
                    succeeded=bool(prompt_result["succeeded"]),
                ),
                **prompt_result,
            })

        decision_rows.extend(placement_decisions)
        option_rows.extend(placement_options)
        prompt_rows.extend(placement_prompts)
        for row in placement_options:
            _append_jsonl(output / "option_rollouts.jsonl", row)
        for row in placement_prompts:
            _append_jsonl(output / "prompt_rollouts.jsonl", row)
        valid_placements += 1
        _append_jsonl(output / "progress.jsonl", {
            "event": "placement_complete",
            "placement_id": placement.placement_id,
            "source_state_sha256": placement.source_state_sha256,
            "valid_placements": valid_placements,
        })
        print(
            f"placement={placement.placement_id} source={placement.source_state_sha256[:10]} "
            f"anchor={anchor_index} complete",
            flush=True,
        )

    validation = validate_decision_rows(option_rows)
    (output / "fresh_decisions.json").write_text(
        json.dumps(decision_rows, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "kind": "fresh_matched_counterfactual_router_online_capture",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": revision,
        "checkpoint_identity": policy.checkpoint_identity,
        "rollout_seed": int(args.rollout_seed),
        "horizon_actions": int(args.horizon),
        "methods": [
            "base", "hazard_specific_prompt_from_reset",
            "binary_risk_to_retreat", "always_detour", "always_retreat",
            "frozen_counterfactual_outcome_router", "counterfactual_oracle",
        ],
        "matched_protocol": (
            "same source state, scene placement, seed, and Base prefix through the "
            "actual on-path T-20 anchor; prompt baseline starts from the same reset"
        ),
        "placements": str(placements_path),
        "placements_sha256": _sha256(placements_path),
        "placement_design_metadata": placement_payload.get("metadata", {}),
        "router_model": str(router_path),
        "router_model_sha256": _sha256(router_path),
        "router_artifact_sha256": router.manifest["artifact_npz_sha256"],
        "frozen_detour_config": frozen_detour,
        "attempted_placements": attempted_placements,
        "valid_placements": valid_placements,
        "excluded_placements": exclusions,
        "validation": validation,
        "statistical_unit": "source_state_sha256",
        "artifacts": {
            "decisions": "fresh_decisions.json",
            "options": "option_rollouts.jsonl",
            "prompt": "prompt_rollouts.jsonl",
        },
    }
    (output / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output": str(output),
        "attempted_placements": attempted_placements,
        "valid_placements": valid_placements,
        "validation": validation,
    }, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", required=True)
    parser.add_argument("--router-model", required=True)
    parser.add_argument("--detour-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--scan-steps", type=int, default=220)
    parser.add_argument("--base-steps", type=int, default=220)
    parser.add_argument("--detour-steps", type=int, default=900)
    parser.add_argument("--retreat-steps", type=int, default=80)
    parser.add_argument("--skip-placements", type=int, default=0)
    parser.add_argument("--max-placements", type=int)
    parser.add_argument("--target-valid-placements", type=int)
    parser.add_argument("--detour-side", type=float, default=1.0)
    parser.add_argument("--detour-lane-margin", type=float, default=0.12)
    parser.add_argument("--detour-lift-offset", type=float, default=0.30)
    parser.add_argument("--detour-descend-offset", type=float, default=0.018)
    parser.add_argument("--detour-leg-cap", type=int, default=140)
    parser.add_argument("--detour-departure-clearance", type=float, default=0.06)
    parser.add_argument("--detour-grasp-xy-offset", type=float, nargs=2, default=[0, 0])
    parser.add_argument("--retreat-back", type=float, default=0.14)
    parser.add_argument("--retreat-up", type=float, default=0.10)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
