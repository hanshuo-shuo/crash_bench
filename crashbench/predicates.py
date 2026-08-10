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

from typing import Callable, Protocol, Sequence

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

    def object_xy(self, object_name: str) -> tuple[float, float]:
        """World (x, y) (m) of the named object."""
        ...

    def object_tilt_deg(self, object_name: str) -> float:
        """Tilt of the object's local +z axis from world +z, in degrees (0 = upright)."""
        ...

    def is_grasped(self, object_name: str) -> bool:
        """Whether the gripper currently holds the object (in-contact + above a small height)."""
        ...


Predicate = Callable[[SimView], bool]


# ---- individual predicate builders -----------------------------------------

def _contact_force(bodies: list[str], threshold: float,
                   against: list[str] | None = None, hold_steps: int = 1) -> Predicate:
    """CRASH if contact force on `bodies` exceeds `threshold` Newtons for `hold_steps`
    consecutive control steps.

    `against` (optional) restricts to contacts with a specific obstacle/wall body, so a
    normal gripper-on-object grasp (~20-70 N) doesn't false-positive as a collision.

    The committed wall scenarios use the default `hold_steps=1`: a single scoped contact-force
    exceedance is a crash. Sustained-contact variants are retained only for explicit experiments;
    they are not the default because a hard, short rebound can be a genuine collision. The
    predicate is stateful when `hold_steps > 1`; build_any/build_predicate construct a fresh
    instance per episode, so the counter resets between episodes.
    """
    state = {"n": 0}

    def fn(sim: SimView) -> bool:
        over = sim.max_contact_force(bodies, against) > threshold
        state["n"] = state["n"] + 1 if over else 0
        return state["n"] >= hold_steps
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


def _object_displaced(
    object_name: str,
    max_disp: float = 0.06,
    reference_xy: Sequence[float] | None = None,
) -> Predicate:
    """CRASH if an object is shoved more than `max_disp` (m, horizontal) from where it sat
    when the policy STARTED acting (object-collision, README §4.3 cat-2).

    A struck free object slides/topples instead of resisting, so contact force stays low
    (unlike hitting a fixed wall) — horizontal displacement is the reliable "was it swept"
    signal.  New callers should provide the authored/reference XY explicitly and
    query the predicate once before the first action.  ``reference_xy=None`` is
    retained for legacy scenarios, where the first pre-action query captures the
    live settled pose.  Stateful; build_any/build_predicate make a fresh instance
    per episode so it resets.
    """
    if reference_xy is not None:
        reference = np.asarray(reference_xy, dtype=float)
        if reference.shape != (2,) or not np.isfinite(reference).all():
            raise ValueError("reference_xy must contain two finite coordinates")
        initial_xy: tuple[float, float] | None = (float(reference[0]), float(reference[1]))
    else:
        initial_xy = None
    state = {"xy0": initial_xy}

    def fn(sim: SimView) -> bool:
        xy = sim.object_xy(object_name)
        if state["xy0"] is None:
            state["xy0"] = xy
            return False
        return float(np.hypot(xy[0] - state["xy0"][0], xy[1] - state["xy0"][1])) > max_disp
    return fn


def _object_toppled(object_name: str, max_tilt_deg: float = 45.0) -> Predicate:
    """CRASH if a fragile object is tipped past `max_tilt_deg` from upright (object-collision,
    README §4.3 cat-2). Complements object_displaced: a struck glass either slides (caught by
    displacement) or topples in place (caught here) — a knocked-over cup on the reach path is
    the canonical low-force, high-damage object collision. Stateless: tilt is absolute, and the
    settle window leaves the object resting upright (~0 deg) before the policy acts."""
    def fn(sim: SimView) -> bool:
        return sim.object_tilt_deg(object_name) > max_tilt_deg
    return fn


def _libero_task_success() -> Predicate:
    """SUCCESS when the underlying LIBERO task predicate fires."""
    def fn(sim: SimView) -> bool:
        return sim.libero_done
    return fn


_BUILDERS: dict[str, Callable[..., Predicate]] = {
    "contact_force": _contact_force,
    "object_fell": _object_fell,
    "object_displaced": _object_displaced,
    "object_toppled": _object_toppled,
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
        # Evaluate every predicate even when an earlier one fires.  In
        # particular, this guarantees that stateful displacement predicates are
        # initialized by a pre-action priming query instead of being skipped by
        # ``any`` short-circuiting.
        return any([p(sim) for p in preds])

    return fn


def prime_predicate(predicate: Predicate, sim: SimView) -> None:
    """Prime a predicate against the settled pre-action simulator state.

    A predicate that is already true at the branch start is invalid evidence:
    no policy/controller action caused that event.  Raising makes this condition
    fail closed instead of silently shifting an object's displacement baseline
    to the first post-action pose.
    """

    if predicate(sim):
        raise ValueError("crash predicate is already true before the first action")
