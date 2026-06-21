"""The core data structure: a CrashBench Scenario (PLAN.md §2).

A scenario is a LIBERO task + a perturbed pre-crash initial state + predicates.
Everything is serializable so scenarios can live on disk as
`scenarios/<id>/{scenario.json, init_state.npy, witness.npy?}`.

Predicates are stored as *specs* (type + params), not Python callables, so they
round-trip through JSON. `crashbench.predicates.build_predicate` turns a spec into
a callable at eval time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

HORIZONS = ("T-1", "T-5", "T-20")
CATEGORIES = (
    "env_collision",        # gripper drifting into wall / plunging at table / shelf
    "object_collision",     # about to knock over / sweep objects
    "self_collision",       # self / dual-arm
    "joint_force_limit",    # joint / force-limit violation
    "grasp_instability",    # held object slipping / tilted / about to drop
    "unsafe_terminal",      # pushing object off table edge / toppling stack
    "constraint_violation", # peg about to bind / door about to slam
)


@dataclass
class PredicateSpec:
    """Serializable description of a predicate. Interpreted by predicates.build_predicate.

    Examples:
        PredicateSpec("contact_force", {"bodies": ["robot0_link5"], "threshold": 20.0})
        PredicateSpec("object_fell", {"object_name": "akita_black_bowl_1", "table_z": 0.41})
        PredicateSpec("grasp_dropped", {"object_name": "akita_black_bowl_1", "init_z": 0.95})
        PredicateSpec("libero_task_success", {})
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, "params": self.params}

    @staticmethod
    def from_dict(d: dict) -> "PredicateSpec":
        return PredicateSpec(type=d["type"], params=d.get("params", {}))


@dataclass
class Scenario:
    """One CrashBench trial. See PLAN.md §2."""

    id: str                                  # e.g. "env_collision_table_plunge__T5__003"
    category: str                            # one of CATEGORIES
    horizon: str                             # one of HORIZONS
    task_suite: str                          # LIBERO suite, e.g. "libero_spatial"
    task_id: int                             # index of the task within the suite
    instruction: str                         # language string given to the VLA

    # pre-crash initial state: a robosuite/LIBERO flat sim-state vector (qpos/qvel/...)
    # passed to env.set_init_state(). Stored alongside as init_state.npy.
    init_state: np.ndarray

    crash_predicates: list[PredicateSpec]    # ANY true -> CRASH
    success_predicate: PredicateSpec         # true -> task completed (default: LIBERO done)

    max_steps: int = 220                     # rollout horizon cap (suite-dependent)
    witness: Optional[np.ndarray] = None     # oracle recovery trajectory (Phase 2), proves recoverability
    # static obstacles injected into the scene for env-collision scenarios (README §4.3 cat-1).
    # Each: {"name": str, "pos": [x,y,z], "size": [sx,sy,sz], "type": "box"(default), "rgba": [...]?}.
    # Static (jointless) bodies -> they add geoms but NO qpos/qvel DOF, so init_state stays valid.
    obstacles: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.category in CATEGORIES, f"bad category {self.category}"
        assert self.horizon in HORIZONS, f"bad horizon {self.horizon}"

    # ---- serialization -----------------------------------------------------
    def save(self, root: str | Path) -> Path:
        """Write to <root>/<id>/ as scenario.json + init_state.npy (+ witness.npy)."""
        d = Path(root) / self.id
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "init_state.npy", self.init_state)
        if self.witness is not None:
            np.save(d / "witness.npy", self.witness)
        meta = {
            "id": self.id,
            "category": self.category,
            "horizon": self.horizon,
            "task_suite": self.task_suite,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "crash_predicates": [p.to_dict() for p in self.crash_predicates],
            "success_predicate": self.success_predicate.to_dict(),
            "max_steps": self.max_steps,
            "obstacles": self.obstacles,
            "has_witness": self.witness is not None,
            "metadata": self.metadata,
        }
        (d / "scenario.json").write_text(json.dumps(meta, indent=2))
        return d

    @staticmethod
    def load(scenario_dir: str | Path) -> "Scenario":
        d = Path(scenario_dir)
        meta = json.loads((d / "scenario.json").read_text())
        init_state = np.load(d / "init_state.npy")
        witness = np.load(d / "witness.npy") if (d / "witness.npy").exists() else None
        return Scenario(
            id=meta["id"],
            category=meta["category"],
            horizon=meta["horizon"],
            task_suite=meta["task_suite"],
            task_id=meta["task_id"],
            instruction=meta["instruction"],
            init_state=init_state,
            crash_predicates=[PredicateSpec.from_dict(p) for p in meta["crash_predicates"]],
            success_predicate=PredicateSpec.from_dict(meta["success_predicate"]),
            max_steps=meta.get("max_steps", 220),
            witness=witness,
            obstacles=meta.get("obstacles", []),
            metadata=meta.get("metadata", {}),
        )


def load_all(root: str | Path) -> list[Scenario]:
    """Load every scenario under `root` (each in its own subdir)."""
    root = Path(root)
    return [Scenario.load(p.parent) for p in sorted(root.glob("*/scenario.json"))]
