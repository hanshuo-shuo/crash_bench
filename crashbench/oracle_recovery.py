"""Utilities for the oracle-stop recovery fine-tuning baseline.

OpenVLA is trained on action tokens in the normalized Open-X convention, while
CrashBench stores the actions actually executed by LIBERO.  In particular, the
gripper sign is inverted at inference time.  Keeping these conversions here makes
the dataset build auditable and unit-testable without importing MuJoCo or torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_TRAIN_SCENARIOS = (
    "env_collision__T5__libero_spatial_t0_wall_d62",
    "env_collision__T5__libero_spatial_t0_wall_d70",
    "env_collision__T5__libero_spatial_t0_wall_d78",
)
DEFAULT_HELDOUT_SCENARIOS = (
    "env_collision__T1__libero_spatial_t0_wall_wide",
    "env_collision__T5__libero_spatial_t0_wall_d85",
)


def env_action_to_openvla_raw(action: Sequence[float]) -> np.ndarray:
    """Undo CrashBench's OpenVLA-to-LIBERO gripper conversion.

    The first six dimensions pass through unchanged.  OpenVLA predicts gripper in
    ``[0, 1]`` (0=closed, 1=open); the evaluation wrapper maps it to ``[-1, 1]``
    and flips the sign for LIBERO.  Therefore ``raw_gripper = (1-env)/2``.
    """

    out = np.asarray(action, dtype=np.float32).copy()
    if out.shape != (7,):
        raise ValueError(f"expected a 7-D action, got {out.shape}")
    out[-1] = (1.0 - out[-1]) / 2.0
    return out


def normalize_openvla_action(
    raw_action: Sequence[float], action_stats: Mapping[str, Sequence[float]]
) -> np.ndarray:
    """Apply OpenVLA's ``BOUNDS_Q99`` normalization used before tokenization."""

    action = np.asarray(raw_action, dtype=np.float32)
    q01 = np.asarray(action_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    mask = np.asarray(action_stats.get("mask", np.ones_like(action, dtype=bool)), dtype=bool)
    if action.shape != (7,) or q01.shape != action.shape or q99.shape != action.shape:
        raise ValueError("action, q01, and q99 must all be 7-D")
    if mask.shape != action.shape:
        raise ValueError("normalization mask must be 7-D")
    span = q99 - q01
    if np.any(mask & (span <= 0)):
        raise ValueError("q99 must be greater than q01 for every normalized dimension")
    normalized = action.copy()
    normalized[mask] = 2.0 * (action[mask] - q01[mask]) / span[mask] - 1.0
    normalized[mask] = np.clip(normalized[mask], -1.0, 1.0)
    return normalized.astype(np.float32)


def validate_scenario_split(
    available: Iterable[str], train: Iterable[str], heldout: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate and canonicalize a disjoint train/held-out scenario split."""

    available_set = set(available)
    train_ids = tuple(train)
    heldout_ids = tuple(heldout)
    if not train_ids or not heldout_ids:
        raise ValueError("train and held-out splits must both be non-empty")
    if len(set(train_ids)) != len(train_ids) or len(set(heldout_ids)) != len(heldout_ids):
        raise ValueError("scenario IDs must be unique within each split")
    overlap = set(train_ids) & set(heldout_ids)
    if overlap:
        raise ValueError(f"train and held-out scenarios overlap: {sorted(overlap)}")
    missing = (set(train_ids) | set(heldout_ids)) - available_set
    if missing:
        raise ValueError(f"unknown scenario IDs: {sorted(missing)}")
    return train_ids, heldout_ids
