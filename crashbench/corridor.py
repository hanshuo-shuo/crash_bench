"""Full-arm nominal swept-volume measurements for axis-aligned obstacles.

The swept volume is represented as the union of world-frame AABBs of every MuJoCo
collision geom attached to links 5--7, the hand, and the gripper/fingers, sampled
along a nominal no-obstacle rollout.  This is conservative for rotated geoms but,
unlike EEF clearance or wall x-position, includes the full distal arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_CORRIDOR_BODIES = (
    "robot0_link5", "robot0_link6", "robot0_link7", "robot0_right_hand",
    "gripper0_right_gripper", "gripper0_leftfinger", "gripper0_rightfinger",
    "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip",
)


@dataclass(frozen=True)
class ClearanceBins:
    """Predeclared signed-distance intervals in metres.

    Intrusion is negative overlap, boundary is the near-corridor nonnegative band,
    and clear is at or beyond ``clear_min``.  The two fixed cuts cover exactly three
    intervals; no sample can be relabeled as a post-hoc "transition" group.
    """

    intrusion_max: float = 0.0
    clear_min: float = 0.10

    def __post_init__(self) -> None:
        if self.intrusion_max > self.clear_min:
            raise ValueError("corridor cut points must satisfy intrusion_max <= clear_min")

    def classify(self, signed_distance_m: float) -> str:
        if signed_distance_m < self.intrusion_max:
            return "intrusion"
        if signed_distance_m < self.clear_min:
            return "boundary"
        return "clear"

    def as_dict(self) -> dict[str, float]:
        return {
            "intrusion_max_m": self.intrusion_max,
            "clear_min_m": self.clear_min,
        }


def aabb_signed_distance(
    first_lo: np.ndarray, first_hi: np.ndarray,
    second_lo: np.ndarray, second_hi: np.ndarray,
) -> float:
    """Euclidean separation when disjoint; negative minimum overlap when intersecting."""
    a0, a1 = np.asarray(first_lo, float), np.asarray(first_hi, float)
    b0, b1 = np.asarray(second_lo, float), np.asarray(second_hi, float)
    if any(x.shape != (3,) for x in (a0, a1, b0, b1)):
        raise ValueError("AABB bounds must be three-vectors")
    if np.any(a1 < a0) or np.any(b1 < b0):
        raise ValueError("AABB upper bounds must not be below lower bounds")
    separation = np.maximum(np.maximum(a0 - b1, b0 - a1), 0.0)
    if np.any(separation > 0):
        return float(np.linalg.norm(separation))
    overlap = np.minimum(a1, b1) - np.maximum(a0, b0)
    return -float(np.min(overlap))


def box_bounds(obstacle: dict) -> tuple[np.ndarray, np.ndarray]:
    if obstacle.get("type", "box") != "box":
        raise ValueError("full-arm corridor currently requires axis-aligned box obstacles")
    center = np.asarray(obstacle["pos"], dtype=float)
    half = np.asarray(obstacle["size"], dtype=float)
    if center.shape != (3,) or half.shape != (3,) or np.any(half <= 0):
        raise ValueError("box obstacle requires 3-D pos and positive MuJoCo half-extents")
    return center - half, center + half


def swept_volume_signed_distance(obstacle: dict, sampled_aabbs: Iterable[dict]) -> tuple[float, dict]:
    """Distance from obstacle to a union of sampled full-arm geom AABBs."""
    obstacle_lo, obstacle_hi = box_bounds(obstacle)
    best_distance = float("inf")
    best: dict | None = None
    count = 0
    for row in sampled_aabbs:
        count += 1
        distance = aabb_signed_distance(obstacle_lo, obstacle_hi, row["lo"], row["hi"])
        if distance < best_distance:
            best_distance = distance
            best = row
    if best is None:
        raise ValueError("nominal swept volume has no sampled robot geoms")
    return best_distance, {
        "closest_body": best.get("body"),
        "closest_geom": best.get("geom"),
        "closest_step": int(best.get("step", -1)),
        "n_sampled_geom_aabbs": count,
    }
