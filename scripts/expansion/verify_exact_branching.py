#!/usr/bin/env python3
"""Live D1 exact-branching conformance on an exposed engineering source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from crashbench.branching.state import capture_exact_state, restore_exact_state
from crashbench.data.source_registry import ExposureRegistry, SourceIdentity
from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import array_sha256
from crashbench.policies import build_policy, canonical_name


OPENVLA_CHECKPOINT = "openvla/openvla-7b-finetuned-libero-spatial"
OPENVLA_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
PI0_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_libero"


def action_sha256(action: Any) -> str:
    return array_sha256(np.asarray(action))


def repeat_gate(rows: list[Mapping[str, Any]], expected_repeats: int) -> dict[str, Any]:
    start_ids = {row["branch_start_bundle_id"] for row in rows}
    start_hashes = {
        json.dumps(row["branch_start_component_hashes"], sort_keys=True) for row in rows
    }
    action_traces = {tuple(row["action_sha256"]) for row in rows}
    state_traces = {tuple(row["state_sha256"]) for row in rows}
    terminals = {row["terminal_signature"] for row in rows}
    criteria = {
        "repeat_count_exact": len(rows) == expected_repeats,
        "branch_start_bundle_identical": len(start_ids) == 1,
        "branch_start_components_identical": len(start_hashes) == 1,
        "next_action_trace_identical": len(action_traces) == 1,
        "state_trace_identical": len(state_traces) == 1,
        "terminal_signature_identical": len(terminals) == 1,
    }
    return {
        "criteria": criteria,
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "classification": "DETERMINISTIC_EXACT" if all(criteria.values()) else "UNCLASSIFIED_FAIL_CLOSED",
    }


def select_exposed_init_state(
    states: np.ndarray, registry: ExposureRegistry, requested_index: int
) -> tuple[int, np.ndarray, str]:
    candidates = range(len(states)) if requested_index < 0 else (requested_index,)
    for index in candidates:
        state = np.asarray(states[index])
        source_hash = array_sha256(state)
        if registry.exposure_reasons(SourceIdentity(source_state_sha256=source_hash)):
            return index, state, source_hash
    raise RuntimeError("no requested default init state is present in the frozen exposure registry")


def terminal_signature(actions: list[str], states: list[str], dones: list[bool]) -> str:
    payload = {"actions": actions, "states": states, "dones": dones}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("openvla", "pi0"), required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-index", type=int, default=-1)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--warmup-policy-steps", type=int, default=1)
    parser.add_argument("--trace-policy-steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D1 result: {args.output}")
    if min(args.repeats, args.trace_policy_steps) < 1:
        raise ValueError("repeats and trace-policy-steps must be positive")

    backend = canonical_name(args.policy)
    env_family = "pi0" if backend == "pi0" else "openvla"
    env = LiberoEnv(args.suite, args.task_id, model_family=env_family, seed=args.seed)
    registry = ExposureRegistry.load(args.exposure_registry)
    init_index, init_state, source_hash = select_exposed_init_state(
        env.default_init_states(), registry, args.init_index
    )

    policy_kwargs: dict[str, Any]
    if backend == "openvla":
        policy_kwargs = {
            "pretrained_checkpoint": OPENVLA_CHECKPOINT,
            "checkpoint_revision": OPENVLA_REVISION,
            "unnorm_key": "libero_spatial",
            "center_crop": True,
        }
    else:
        policy_kwargs = {
            "pretrained_checkpoint": PI0_CHECKPOINT,
            "config_name": "pi0_libero",
            "num_open_loop_steps": 5,
        }
    policy = build_policy(backend, **policy_kwargs)
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset_to(init_state)
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(env.dummy_action())
    for _ in range(args.warmup_policy_steps):
        policy_obs = env.policy_observation(obs, policy.resize_size)
        action = policy.act(policy_obs, env.task_description)
        obs, _, _, _ = env.step(np.asarray(action).tolist())

    capture_started = time.monotonic()
    bundle = capture_exact_state(
        env,
        policy,
        identity={
            "source_id": source_hash,
            "policy_id": backend,
            "mechanism_id": "nominal_engineering_conformance_v1",
            "task_id": f"{args.suite}:{args.task_id}",
            "trajectory_id": f"default_init:{init_index}:seed:{args.seed}",
            "anchor_id": f"after_policy_steps:{args.warmup_policy_steps}",
        },
        provenance={
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "checkpoint": policy_kwargs["pretrained_checkpoint"],
            "checkpoint_revision": policy_kwargs.get("checkpoint_revision"),
            "source_role": "EXPOSED_ENGINEERING_ONLY",
        },
        declared_branch_seed=args.seed,
    )
    capture_seconds = time.monotonic() - capture_started

    rows = []
    for repeat in range(args.repeats):
        started = time.monotonic()
        obs = restore_exact_state(bundle, env, policy)
        actions: list[str] = []
        states: list[str] = []
        dones: list[bool] = []
        for _ in range(args.trace_policy_steps):
            policy_obs = env.policy_observation(obs, policy.resize_size)
            action = policy.act(policy_obs, env.task_description)
            actions.append(action_sha256(action))
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            states.append(array_sha256(env.flat_state()))
            dones.append(bool(done))
        rows.append(
            {
                "repeat_index": repeat,
                "branch_start_bundle_id": bundle.bundle_id,
                "branch_start_component_hashes": dict(bundle.component_hashes),
                "action_sha256": actions,
                "state_sha256": states,
                "done": dones,
                "terminal_signature": terminal_signature(actions, states, dones),
                "restore_and_trace_seconds": time.monotonic() - started,
            }
        )

    gate = repeat_gate(rows, args.repeats)
    payload = {
        "schema_version": 1,
        "kind": "crashbench_expansion_d1_exact_branching_conformance",
        "backend": backend,
        "source": {
            "suite": args.suite,
            "task_id": args.task_id,
            "default_init_index": init_index,
            "source_state_sha256": source_hash,
            "role": "EXPOSED_ENGINEERING_ONLY",
        },
        "configuration": vars(args) | {"output": str(args.output), "exposure_registry": str(args.exposure_registry)},
        "bundle": {
            "bundle_id": bundle.bundle_id,
            "component_hashes": dict(bundle.component_hashes),
            "rng_sources": {
                "numpy_generators": sorted(bundle.rng.numpy_generators),
                "numpy_random_states": sorted(bundle.rng.numpy_random_states or {}),
                "torch_cpu": bundle.rng.torch_cpu_state is not None,
                "torch_cuda_devices": len(bundle.rng.torch_cuda_states),
                "policy_continuation_sha256": bundle.policy_continuation.sha256(),
            },
            "capture_seconds": capture_seconds,
        },
        "repeats": rows,
        "gate": gate,
        "claim_boundary": "Engineering conformance only; no scientific option outcome was opened.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "backend": backend, **gate}, sort_keys=True))
    if gate["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
