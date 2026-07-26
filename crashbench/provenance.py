"""Strict run provenance for new, paper-facing CrashBench experiments.

Historical manifests document what is known about old outputs.  This module is for
new runs: it fails closed when the repository is dirty, a checkpoint revision is not
explicit, scenarios cannot be fingerprinted, or an output directory already exists.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from crashbench.scenario import Scenario, scenario_fingerprint


def _git(root: Path, *args: str) -> str:
    executable = os.environ.get("CB_GIT_EXECUTABLE", "git")
    try:
        proc = subprocess.run(
            [executable, *args], cwd=root, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"git executable not found: {executable!r}; set CB_GIT_EXECUTABLE to an absolute path"
        ) from exc
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def repository_provenance(root: str | Path, *, require_clean: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    dirty_lines = [line for line in _git(root, "status", "--porcelain=v1").splitlines() if line]
    if require_clean and dirty_lines:
        preview = ", ".join(line[3:] for line in dirty_lines[:8])
        raise RuntimeError(
            "paper-facing runs require a clean checkout; commit/stash changes first. "
            f"Dirty paths: {preview}"
        )
    return {
        "git_commit": commit,
        "git_commit_short": commit[:12],
        "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty_lines),
        "git_dirty_entries": dirty_lines,
    }


def require_checkpoint_revision(revision: str | None) -> str:
    """Require an explicit immutable revision rather than silently using a moving tag."""
    if revision is None or not revision.strip():
        raise ValueError("checkpoint_revision is required for P0 runs")
    revision = revision.strip()
    if revision.lower() in {"main", "master", "latest", "default", "unknown"}:
        raise ValueError(
            f"checkpoint_revision={revision!r} is mutable/ambiguous; use an immutable commit revision"
        )
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is None:
        raise ValueError(
            "checkpoint_revision must be the exact 40--64 character hexadecimal commit/digest, "
            f"not a tag: {revision!r}"
        )
    return revision


def scenario_provenance(rows: Iterable[tuple[str, str, Path, Scenario]]) -> list[dict[str, Any]]:
    """Return exact scenario bytes plus the predeclared split/condition."""
    output = []
    seen: dict[str, tuple[str, str]] = {}
    for split, condition, path, scenario in rows:
        path = Path(path).resolve()
        fingerprint = scenario_fingerprint(path.parent)
        previous = seen.get(fingerprint)
        current = (split, condition)
        if previous is not None and previous[0] != split:
            raise ValueError(
                f"scenario leakage: {scenario.id} fingerprint occurs in {previous} and {current}"
            )
        seen[fingerprint] = current
        output.append({
            "scenario_id": scenario.id,
            "scenario_json": str(path),
            "fingerprint_sha256": fingerprint,
            "split": split,
            "condition": condition,
            "task_suite": scenario.task_suite,
            "task_id": scenario.task_id,
        })
    return output


def runtime_provenance() -> dict[str, Any]:
    env_keys = (
        "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST",
        "CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM",
    )
    packages: dict[str, str] = {}
    for name in ("numpy", "torch", "transformers", "mujoco", "robosuite"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "not-exposed"))
        except Exception as exc:  # package inventory must not make CPU preflight import GPU stacks
            packages[name] = f"unavailable: {type(exc).__name__}"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "argv": sys.argv,
        "environment": {key: os.environ[key] for key in env_keys if key in os.environ},
        "packages": packages,
    }


def write_json_exclusive(path: str | Path, value: Any) -> None:
    """Create a JSON artifact without overwriting an earlier run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    with path.open("x", encoding="utf-8") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
