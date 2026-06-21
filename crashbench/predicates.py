"""Crash + success predicates (PLAN.md §3, README §4.5).

Predicates are objective functions of *sim state*, never vision. They are built
from a serializable `PredicateSpec` so scenarios round-trip through JSON.

Each predicate is a callable `fn(sim: SimView) -> bool`. The `SimView` is a thin
read-only interface over the LIBERO/robosuite env (implemented in
`crashbench.envs.libero_adapter.LiberoSimView`) so predicates never touch
robosuite internals directly and can be unit-tested with a fake.

For the pilot we ship the 3 easiest, most reliable predicates (PLAN.md §3):
  - contact_force : collision crashes
  - object_fell   : fell-off-table crashes
  - grasp_dropped : grasp-instability crashes
plus libero_task_success for the success side. Joint/force-limit and penetration
predicates are Phase 2.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class SimView(Protocol):
    """Read-only view of the simulator that predicates query. Implemented by the env adapter."""

    @property
    def libero_done(self) -> bool:
        """Whether the underlying LIBERO task reports success (env.step `done`)."""
        ...

    def max_contact_force(self, bodies: list[str], against: list[str] | None = None) -> float:
        """Max magnitude of contact force (N) on any of the named bodies. If `against`
        is given, only count contacts whose other body is in `against` (isolates an
        env-collision from normal grasp contacts)."""
        ...

    def object_z(self, object_name: str) -> float:
        """World z (m) of the named object's COM/body."""
        ...

    def is_grasped(self, object_name: str) -> bool:
        """Whether the gripper currently holds the object (in-contact + above a small height)."""
        ...


Predicate = Callable[[SimView], bool]


# ---- individual predicate builders -----------------------------------------

def _contact_force(bodies: list[str], threshold: float,
                   against: list[str] | None = None) -> Predicate:
    """CRASH if contact force on `bodies` exceeds `threshold` Newtons.

    `against` (optional) restricts to contacts with a specific obstacle/wall body, so a
    normal gripper-on-object grasp (~20-70 N) doesn't false-positive as a collision.
    """
    def fn(sim: SimView) -> bool:
        return sim.max_contact_force(bodies, against) > threshold
    return fn


def _object_fell(object_name: str, table_z: float, margin: float = 0.05) -> Predicate:
    """CRASH if object COM falls below table surface (fell off the table)."""
    def fn(sim: SimView) -> bool:
        return sim.object_z(object_name) < (table_z - margin)
    return fn


def _grasp_dropped(object_name: str, init_z: float, drop: float = 0.10) -> Predicate:
    """CRASH if a held object is no longer grasped AND has dropped below its initial height."""
    def fn(sim: SimView) -> bool:
        return (not sim.is_grasped(object_name)) and (sim.object_z(object_name) < init_z - drop)
    return fn


def _libero_task_success() -> Predicate:
    """SUCCESS when the underlying LIBERO task predicate fires."""
    def fn(sim: SimView) -> bool:
        return sim.libero_done
    return fn


_BUILDERS: dict[str, Callable[..., Predicate]] = {
    "contact_force": _contact_force,
    "object_fell": _object_fell,
    "grasp_dropped": _grasp_dropped,
    "libero_task_success": _libero_task_success,
}


def build_predicate(spec) -> Predicate:
    """Turn a PredicateSpec into a callable. `spec` is a crashbench.scenario.PredicateSpec."""
    if spec.type not in _BUILDERS:
        raise ValueError(f"unknown predicate type {spec.type!r}; known: {list(_BUILDERS)}")
    return _BUILDERS[spec.type](**spec.params)


def build_any(specs) -> Predicate:
    """Combine a list of crash specs with OR (any crash fires -> crash)."""
    preds = [build_predicate(s) for s in specs]
    def fn(sim: SimView) -> bool:
        return any(p(sim) for p in preds)
    return fn
