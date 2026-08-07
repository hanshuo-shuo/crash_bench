#!/usr/bin/env python3
"""Simulator-only probe for exact glass branch runtime-state capture/restore."""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
faulthandler.enable()

from crashbench.envs import LiberoEnv
from crashbench.glass_recovery_data import read_placement_manifest
from scripts.collect_glass_recovery_pairs import _controller_state_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", required=True)
    parser.add_argument("--placement-id", required=True)
    parser.add_argument("--settle-steps", type=int, default=10)
    args = parser.parse_args()

    placements, _ = read_placement_manifest(args.placements)
    placement = next(row for row in placements if row.placement_id == args.placement_id)
    source = np.load(Path(args.placements).parent / placement.source_state_path)
    print("PROBE create_env", flush=True)
    env = LiberoEnv(placement.task_suite, placement.task_id)
    print("PROBE reset_to", flush=True)
    obs = env.reset_to(source, movable_objects=[placement.on_path_glass])
    for index in range(args.settle_steps):
        print(f"PROBE settle {index}", flush=True)
        obs, _, _, _ = env.step(env.dummy_action())
    print("PROBE flat_state", flush=True)
    flat = env.flat_state()
    print("PROBE controller_state", flush=True)
    runtime = env.controller_state()
    print(
        f"PROBE captured flat={flat.shape} keys={sorted(runtime)} "
        f"hash={_controller_state_sha256(runtime)}",
        flush=True,
    )
    print("PROBE reset_to_exact", flush=True)
    env.reset_to_exact(flat, movable_objects=[placement.on_path_glass])
    print("PROBE restore_controller_state", flush=True)
    env.restore_controller_state(runtime)
    restored = env.controller_state()
    restored_hash = _controller_state_sha256(restored)
    print(f"PROBE restored hash={restored_hash}", flush=True)
    if restored_hash != _controller_state_sha256(runtime):
        raise SystemExit("runtime-state hash changed after restore")
    print("PROBE PASS", flush=True)


if __name__ == "__main__":
    main()
