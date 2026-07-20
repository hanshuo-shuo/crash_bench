"""Geometry-aware behavioral metrics for obstacle-directed motion.

The policy action norm mixes translation directions (and, in the full action,
rotation/gripper channels).  These helpers keep the directional part explicit:
positive ``toward_obstacle`` means motion toward the obstacle, while a negative
clearance delta means the obstacle is getting closer.

Obstacle geometry currently comes from the injected axis-aligned MuJoCo boxes.
The functions are deliberately NumPy-only so they can also be used by offline
analysis and unit tests without LIBERO or a GPU.
"""

from __future__ import annotations

import numpy as np


def _box_arrays(obstacle: dict) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(obstacle["pos"], dtype=np.float64)
    half_size = np.asarray(obstacle["size"], dtype=np.float64)
    if center.shape != (3,) or half_size.shape != (3,):
        raise ValueError("axis-aligned obstacle must have 3D pos and size")
    return center, half_size


def aabb_clearance(point: np.ndarray, obstacle: dict) -> float:
    """Euclidean distance from a point to an axis-aligned box (0 if inside)."""
    center, half_size = _box_arrays(obstacle)
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError("point must be a 3-vector")
    outside = np.maximum(np.abs(p - center) - half_size, 0.0)
    return float(np.linalg.norm(outside))


def obstacle_direction(point: np.ndarray, obstacle: dict) -> np.ndarray:
    """Return a unit vector pointing from ``point`` toward the box.

    For an outside point this is the shortest point-to-box direction.  If the
    point is already inside the box (or exactly on it), fall back to the vector
    toward the box center.  Thus ``dot(v, obstacle_direction) > 0`` always has
    the intuitive meaning "the command moves into the obstacle".
    """
    center, half_size = _box_arrays(obstacle)
    p = np.asarray(point, dtype=np.float64)
    closest = np.clip(p, center - half_size, center + half_size)
    toward = closest - p
    norm = float(np.linalg.norm(toward))
    if norm < 1e-12:
        toward = center - p
        norm = float(np.linalg.norm(toward))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return toward / norm


def project_toward_obstacle(vector: np.ndarray, direction: np.ndarray) -> float:
    """Signed projection of a 3D vector onto the obstacle-directed unit vector."""
    v = np.asarray(vector, dtype=np.float64)
    n = np.asarray(direction, dtype=np.float64)
    return float(np.dot(v, n))


def time_to_contact(clearance: float, toward_velocity: float, eps: float = 1e-8) -> float:
    """One-step-velocity TTC proxy in environment control steps.

    TTC is infinite when the observed end-effector displacement is not toward
    the obstacle.  This is an end-effector TTC proxy; a tall wall may be struck
    by another arm link before the end-effector reaches the box.
    """
    if clearance <= 0.0:
        return 0.0
    if toward_velocity <= eps:
        return float("inf")
    return float(clearance / toward_velocity)
