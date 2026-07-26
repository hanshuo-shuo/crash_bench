"""Configuration and fail-closed preflight for the provenance-complete P0 study."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crashbench.provenance import require_checkpoint_revision, scenario_provenance
from crashbench.scenario import Scenario


SPLIT_MINIMUMS = {"train": 3, "calibration": 3, "heldout": 5}
VALID_CONDITIONS = {"wall", "nowall", "offpath"}


@dataclass(frozen=True)
class ScenarioRun:
    split: str
    condition: str
    path: Path
    scenario: Scenario
    repeats: int


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError("P0 config schema_version must be 1")
    cfg["_config_path"] = str(path)
    return cfg


def expand_runs(cfg: dict[str, Any], root: str | Path) -> list[ScenarioRun]:
    root = Path(root).resolve()
    runs: list[ScenarioRun] = []
    for group in cfg.get("scenario_groups", []):
        split = group["split"]
        if split not in SPLIT_MINIMUMS:
            raise ValueError(f"unknown split {split!r}; expected {sorted(SPLIT_MINIMUMS)}")
        repeats = int(group.get("repeats", cfg.get("repeats", 0)))
        if repeats < 2:
            raise ValueError(f"{split} requires at least two independent rollouts per scenario")
        conditions = group.get("conditions", [])
        if not conditions or any(c not in VALID_CONDITIONS for c in conditions):
            raise ValueError(f"invalid conditions {conditions!r}; expected subset of {sorted(VALID_CONDITIONS)}")
        pattern = str(root / group["glob"])
        paths = [Path(p).resolve() for p in sorted(glob.glob(pattern))]
        if not paths:
            raise ValueError(f"scenario glob matched nothing: {group['glob']!r}")
        for path in paths:
            scenario = Scenario.load(path.parent)
            for condition in conditions:
                if condition in {"wall", "offpath"} and not scenario.obstacles:
                    raise ValueError(f"{scenario.id}: condition {condition} requires an obstacle")
                runs.append(ScenarioRun(split, condition, path, scenario, repeats))
    if not runs:
        raise ValueError("scenario_groups is empty")
    return runs


def validate_design(cfg: dict[str, Any], runs: list[ScenarioRun]) -> dict[str, Any]:
    revision = require_checkpoint_revision(cfg.get("checkpoint_revision"))
    if not cfg.get("checkpoint"):
        raise ValueError("checkpoint is required")
    if not cfg.get("output_dir"):
        raise ValueError("output_dir is required and must be new for every run")

    # Scenario independence is based on exact scenario bytes, not duplicated condition rows.
    provenance = scenario_provenance(
        (r.split, r.condition, r.path, r.scenario) for r in runs
    )
    split_ids: dict[str, set[str]] = {key: set() for key in SPLIT_MINIMUMS}
    split_fingerprints: dict[str, set[str]] = {key: set() for key in SPLIT_MINIMUMS}
    scenario_conditions: dict[tuple[str, str], set[str]] = {}
    tasks = set()
    wall_geometries: dict[str, set[tuple[float, ...]]] = {key: set() for key in SPLIT_MINIMUMS}
    for row in provenance:
        split_ids[row["split"]].add(row["scenario_id"])
        split_fingerprints[row["split"]].add(row["fingerprint_sha256"])
        scenario_conditions.setdefault((row["split"], row["fingerprint_sha256"]), set()).add(
            row["condition"])
        tasks.add((row["task_suite"], row["task_id"]))
    for run in runs:
        for obstacle in run.scenario.obstacles or []:
            if obstacle.get("name") == "crash_wall" or obstacle.get("type", "box") == "box":
                signature = tuple(float(x) for x in obstacle["pos"] + obstacle["size"])
                wall_geometries[run.split].add(signature)
                break
    for split, minimum in SPLIT_MINIMUMS.items():
        if len(split_fingerprints[split]) < minimum:
            raise ValueError(
                f"{split} has {len(split_fingerprints[split])} independent scenarios; minimum is {minimum}"
            )
    for key, conditions_for_scenario in scenario_conditions.items():
        if "nowall" not in conditions_for_scenario or not ({"wall", "offpath"} & conditions_for_scenario):
            raise ValueError(
                f"{key} must have a paired obstacle-present and nowall rollout; "
                f"found {sorted(conditions_for_scenario)}"
            )
    if len(tasks) < 2:
        raise ValueError(f"P0 probe design requires at least two tasks; found {sorted(tasks)}")
    for a, b in (("train", "calibration"), ("train", "heldout"), ("calibration", "heldout")):
        overlap = split_fingerprints[a] & split_fingerprints[b]
        if overlap:
            raise ValueError(f"scenario fingerprint leakage between {a} and {b}: {sorted(overlap)}")
    seen_wall_geometry = wall_geometries["train"] | wall_geometries["calibration"]
    reused_geometry = seen_wall_geometry & wall_geometries["heldout"]
    if reused_geometry:
        raise ValueError(
            "heldout must use new wall positions/sizes; reused geometries: "
            f"{sorted(reused_geometry)}"
        )
    targets = cfg.get("task_targets", {})
    missing_targets = [f"{suite}:{task_id}" for suite, task_id in sorted(tasks)
                       if f"{suite}:{task_id}" not in targets]
    if missing_targets:
        raise ValueError(f"task_targets lacks {missing_targets}; task-phase baseline cannot be computed")

    conditions = {r.condition for r in runs}
    if "wall" not in conditions or not ({"nowall", "offpath"} & conditions):
        raise ValueError("design requires wall treatment and at least one nowall/offpath negative condition")
    return {
        "checkpoint": cfg["checkpoint"],
        "checkpoint_revision": revision,
        "tasks": [{"task_suite": s, "task_id": t} for s, t in sorted(tasks)],
        "unique_scenarios_by_split": {k: len(v) for k, v in split_fingerprints.items()},
        "episode_count": sum(r.repeats for r in runs),
        "scenario_provenance": provenance,
    }
