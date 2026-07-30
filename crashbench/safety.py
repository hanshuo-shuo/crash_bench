"""Small, auditable safety primitives for the OpenVLA baseline comparison.

These helpers intentionally separate two objections:

* :func:`project_signed_distance_cbf` is a classical, EEF-only action shield.  It
  projects just the obstacle-directed Cartesian component and otherwise preserves
  the VLA action.
* :func:`full_arm_signed_distance` exposes simulator-oracle full-arm geometry.  The
  experiment uses it for a latched stop proxy for recovery fine-tuning; it is not a
  learned policy and must be reported as an oracle upper bound.
"""

from __future__ import annotations

import re

import numpy as np

from crashbench.corridor import box_bounds, swept_volume_signed_distance
from crashbench.envs.libero_adapter import ROBOT_CONTACT_BODIES


EPS = 1e-9


def point_box_clearance_and_toward(
    point: np.ndarray, obstacle: dict,
) -> tuple[float, np.ndarray]:
    """Unsigned point-to-box clearance and unit vector pointing toward the box."""
    p = np.asarray(point, dtype=float)
    lo, hi = box_bounds(obstacle)
    if p.shape != (3,):
        raise ValueError("point must be a three-vector")
    closest = np.clip(p, lo, hi)
    toward = closest - p
    clearance = float(np.linalg.norm(toward))
    if clearance > EPS:
        return clearance, toward / clearance

    # At/inside the box, choose the closest face deterministically.  The clearance
    # is already zero; this direction only defines which action component to remove.
    face_distances = np.concatenate([p - lo, hi - p])
    face = int(np.argmin(face_distances))
    normal = np.zeros(3, dtype=float)
    axis = face % 3
    normal[axis] = -1.0 if face < 3 else 1.0
    return 0.0, normal


def project_signed_distance_cbf(
    action: np.ndarray,
    eef_pos: np.ndarray,
    obstacle: dict,
    *,
    margin_m: float = 0.08,
    action_scale_m: float = 0.05,
    alpha: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Project an action to satisfy a one-step signed-distance CBF approximation.

    With ``h = clearance - margin``, the constraint is

    ``action_scale * a_toward <= alpha * max(h, 0)``.

    Only the Cartesian component toward the obstacle is reduced.  Tangential and
    retreat motion, rotation, and the gripper command pass through unchanged.
    ``action_scale_m`` is the documented LIBERO OSC displacement approximation for
    a unit normalized Cartesian command, not a simulator look-ahead.
    """
    if margin_m < 0 or action_scale_m <= 0 or not 0 < alpha <= 1:
        raise ValueError("require margin_m>=0, action_scale_m>0, and 0<alpha<=1")
    proposed = np.asarray(action, dtype=float)
    if proposed.ndim != 1 or proposed.size < 3:
        raise ValueError("action must be a flat vector with xyz translation")

    clearance, toward = point_box_clearance_and_toward(eef_pos, obstacle)
    toward_before = float(np.dot(proposed[:3], toward))
    h = clearance - margin_m
    max_toward = alpha * max(h, 0.0) / action_scale_m
    excess = max(0.0, toward_before - max_toward)
    executed = proposed.copy()
    if excess > 0:
        executed[:3] -= excess * toward
    toward_after = float(np.dot(executed[:3], toward))
    return executed, {
        "intervened": bool(excess > 0),
        "clearance_m": clearance,
        "barrier_h_m": h,
        "toward_before": toward_before,
        "toward_after": toward_after,
        "max_toward": max_toward,
        "correction_norm": float(np.linalg.norm(executed[:3] - proposed[:3])),
    }


def full_arm_signed_distance(sim_view, obstacle: dict) -> tuple[float, dict]:
    """Current full distal-arm AABB signed distance to an axis-aligned obstacle."""
    rows = sim_view.robot_geom_aabbs(list(ROBOT_CONTACT_BODIES))
    return swept_volume_signed_distance(obstacle, rows)


def parse_binary_collision_answer(text: str) -> tuple[bool, str]:
    """Parse a Qwen answer; malformed/ambiguous responses fail closed to collision."""
    tokens = re.findall(r"\b(?:YES|NO)\b", str(text).upper())
    unique = set(tokens)
    if tokens and unique == {"YES"}:
        return True, "yes"
    if tokens and unique == {"NO"}:
        return False, "no"
    return True, "ambiguous_fail_closed"
