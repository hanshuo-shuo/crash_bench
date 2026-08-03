#!/usr/bin/env python3
"""Author a split-safe glass recovery placement design.

The default design contains 100 train, 20 validation, and 40 clustered held-out
placements.  Splits use disjoint LIBERO source initial states; a state hash is
never shared across splits.  Each placement describes the four scene branches
used later by ``collect_glass_recovery_pairs.py``.

This step needs LIBERO/MuJoCo but does not load OpenVLA.  It therefore runs much
faster than paired trajectory collection.
"""

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
from crashbench.glass_recovery_data import GlassPlacement, array_sha256, write_placement_manifest


TARGET = "akita_black_bowl_1"
SETTLE_STEPS = 10
RGBA = [0.55, 0.78, 0.95, 0.55]


GEOMETRY = {
    "train": (
        ("nominal", [0.030, 0.060], 400.0, [0.50, 0.56, 0.62, 0.68, 0.72]),
    ),
    "validation": (
        ("validation_tall", [0.028, 0.067], 360.0, [0.53, 0.62, 0.71]),
    ),
    # Held-out placements are grouped by a scene-level geometry family.  No
    # family identifier is reused by train or validation.
    "heldout": (
        ("tall_narrow", [0.024, 0.078], 330.0, [0.56, 0.64, 0.72]),
        ("wide_glass", [0.040, 0.060], 440.0, [0.58, 0.67, 0.76]),
        ("late_approach", [0.030, 0.060], 400.0, [0.74, 0.79, 0.83]),
    ),
}


def _glass(name: str, xy: np.ndarray, table_top: float, size: list[float], density: float) -> dict:
    return {
        "name": name,
        "type": "cylinder",
        "size": [round(float(size[0]), 4), round(float(size[1]), 4)],
        "pos": [round(float(xy[0]), 4), round(float(xy[1]), 4),
                round(float(table_top + size[1]), 4)],
        "rgba": RGBA,
        "density": round(float(density), 2),
    }


def _quotas(total: int, state_indices: list[int]) -> list[int]:
    if total < 1 or not state_indices:
        raise ValueError("each split needs positive placements and at least one source state")
    base, remainder = divmod(total, len(state_indices))
    return [base + int(index < remainder) for index in range(len(state_indices))]


def _state_layout(
    available: int,
    train_states: int,
    validation_states: int,
    heldout_states: int,
) -> dict[str, list[int]]:
    requested = train_states + validation_states + heldout_states
    if requested > available:
        raise ValueError(
            f"requested {requested} disjoint source states, but task exposes only {available}"
        )
    order = list(range(requested))
    return {
        "train": order[:train_states],
        "validation": order[train_states:train_states + validation_states],
        "heldout": order[train_states + validation_states:],
    }


def author(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    state_root = output / "states"

    env = LiberoEnv(args.suite, args.task_id)
    states = np.asarray(env.default_init_states())
    layout = _state_layout(
        len(states), args.train_states, args.validation_states, args.heldout_states
    )
    requested_counts = {
        "train": args.train_placements,
        "validation": args.validation_placements,
        "heldout": args.heldout_placements,
    }

    placements: list[GlassPlacement] = []
    for split, indices in layout.items():
        quotas = _quotas(requested_counts[split], indices)
        geometry_families = GEOMETRY[split]
        split_counter = 0
        for state_slot, (source_index, quota) in enumerate(zip(indices, quotas)):
            state = np.asarray(states[source_index], dtype=np.float64)
            state_rel = Path("states") / split / f"libero_state_{source_index:03d}.npy"
            state_path = output / state_rel
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(state_path, state)
            state_hash = array_sha256(state)

            obs = env.reset_to(state)
            for _ in range(args.settle_steps):
                obs, _, _, _ = env.step(env.dummy_action())
            home = np.asarray(obs["robot0_eef_pos"], dtype=float)[:2]
            bowl = np.asarray(obs[f"{TARGET}_pos"], dtype=float)
            table_top = float(bowl[2] - args.bowl_rest_offset)
            direction = bowl[:2] - home
            norm = float(np.linalg.norm(direction))
            if norm < 0.10:
                raise RuntimeError(f"source state {source_index} has degenerate home-to-bowl path")
            direction /= norm
            perpendicular = np.asarray([-direction[1], direction[0]])

            for local_index in range(quota):
                family, size, density, fractions = geometry_families[
                    (state_slot + local_index) % len(geometry_families)
                ]
                fraction = float(fractions[(state_slot + local_index) % len(fractions)])
                # Small deterministic along-path variation increases train coverage;
                # split separation is still guaranteed by the source-state hash.
                along_jitter = ((local_index % 3) - 1) * args.along_jitter
                anchor = home + np.clip(fraction + along_jitter, 0.05, 0.95) * (bowl[:2] - home)
                on_path = _glass("glass_1", anchor, table_top, size, density)
                side = -1.0 if (source_index + local_index) % 2 else 1.0
                off_xy = anchor + side * args.control_offset * perpendicular
                off_path = _glass("glass_1", off_xy, table_top, size, density)

                # A dense line of fragile glasses spans the declared detour
                # corridor.  Collection does not blindly trust this declaration:
                # it searches the fixed oracle-controller class and only accepts a
                # blocked record if every configured recovery attempt fails while
                # RetreatHold remains collision-free.
                blocked = []
                barrier_offsets = np.linspace(
                    -args.blocked_half_width, args.blocked_half_width, args.blocked_glasses
                )
                for barrier_index, barrier_offset in enumerate(barrier_offsets):
                    # Keep the central on-path glass identical in height to the
                    # measured recoverable scene.  Only the lateral detour lanes
                    # are fenced by tall glasses; otherwise replacing the cup at
                    # a T-20 robot state can create an invalid initial overlap.
                    blocked_size = (
                        size if abs(float(barrier_offset)) < 1e-9
                        else [args.blocked_radius, args.blocked_half_height]
                    )
                    blocked.append(_glass(
                        f"glass_block_{barrier_index}",
                        anchor + barrier_offset * perpendicular,
                        table_top,
                        blocked_size,
                        density,
                    ))
                placement_id = f"glass_recovery_{split}_{split_counter:04d}"
                cluster_id = f"{split}/{family}"
                placements.append(GlassPlacement(
                    placement_id=placement_id,
                    split=split,
                    cluster_id=cluster_id,
                    task_suite=args.suite,
                    task_id=args.task_id,
                    instruction=env.task_description,
                    source_state_path=state_rel.as_posix(),
                    source_state_sha256=state_hash,
                    on_path_glass=on_path,
                    off_path_glass=off_path,
                    blocked_glasses=blocked,
                    nominal_fraction=fraction,
                    metadata={
                        "source_state_index": source_index,
                        "geometry_family": family,
                        "home_xy": home.round(6).tolist(),
                        "bowl_xyz": bowl.round(6).tolist(),
                        "path_direction_xy": direction.round(6).tolist(),
                        "off_path_offset_m": args.control_offset,
                        "blocked_corridor_half_width_m": args.blocked_half_width,
                        "blocked_glass_half_height_m": args.blocked_half_height,
                        "blocked_controller_class": (
                            "GlassDetourComplete sides={-1,+1}, declared lane margins and transit heights"
                        ),
                    },
                ))
                split_counter += 1

    payload = write_placement_manifest(
        output / "placements.json",
        placements,
        metadata={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": os.environ.get("CB_CODE_COMMIT"),
            "suite": args.suite,
            "task_id": args.task_id,
            "settle_steps": args.settle_steps,
            "requested_counts": requested_counts,
            "source_state_indices": layout,
            "policy": (
                "source initial states are disjoint across train/validation/heldout; "
                "heldout geometry families are clustered and absent from train"
            ),
        },
    )
    print(json.dumps(payload["design"], indent=2), flush=True)
    print(f"wrote {output / 'placements.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/glass_recovery_v1/placements")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--train-placements", type=int, default=100)
    parser.add_argument("--validation-placements", type=int, default=20)
    parser.add_argument("--heldout-placements", type=int, default=40)
    parser.add_argument("--train-states", type=int, default=30)
    parser.add_argument("--validation-states", type=int, default=8)
    parser.add_argument("--heldout-states", type=int, default=12)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--bowl-rest-offset", type=float, default=0.005)
    parser.add_argument("--control-offset", type=float, default=0.20)
    parser.add_argument("--along-jitter", type=float, default=0.012)
    parser.add_argument("--blocked-half-width", type=float, default=0.28)
    parser.add_argument("--blocked-glasses", type=int, default=9)
    parser.add_argument("--blocked-radius", type=float, default=0.032)
    parser.add_argument("--blocked-half-height", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.blocked_glasses < 3 or args.blocked_glasses % 2 == 0:
        raise SystemExit("blocked scene needs an odd number of at least three glasses")
    author(args)


if __name__ == "__main__":
    main()
