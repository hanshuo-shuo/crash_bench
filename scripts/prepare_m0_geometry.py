#!/usr/bin/env python3
"""M0 hygiene: separate the lowered d62 detour scenario from the original wall benchmark.

The task-completing d62 recovery deliberately lowered the wall and replaced the safe-abort
witness. Keeping that variant under ``scenarios/`` would make later shield/baseline numbers
incomparable with the frozen capture data. This script is login-node only and uses the known
pre-detour Git commit to restore the original scenario files exactly.

Run from the repository root:
    python3 scripts/prepare_m0_geometry.py

It is intentionally conservative: an existing ``scenarios_detour_lowwall/<d62>`` directory is not
overwritten unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ID = "env_collision__T5__libero_spatial_t0_wall_d62"
CURRENT = ROOT / "scenarios" / SCENARIO_ID
DETOUR_ROOT = ROOT / "scenarios_detour_lowwall"
DETOUR_ID = f"{SCENARIO_ID}__lowwall_detour_v1"
DETOUR = DETOUR_ROOT / DETOUR_ID
ORIGINAL_COMMIT = "02895e6"
RELATIVE = Path("scenarios") / SCENARIO_ID


def git_bytes(path: Path) -> bytes:
    spec = f"{ORIGINAL_COMMIT}:{path.as_posix()}"
    return subprocess.check_output(["git", "show", spec], cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="replace an existing scenarios_detour_lowwall/d62 copy")
    args = ap.parse_args()

    if not CURRENT.is_dir():
        raise SystemExit(f"missing current scenario directory: {CURRENT}")
    current_json = CURRENT / "scenario.json"
    current_data = json.loads(current_json.read_text())
    current_half_z = float(current_data["obstacles"][0]["size"][2])
    if abs(current_half_z - 0.12) > 1e-6:
        raise SystemExit("current d62 does not look like the lowered detour variant; refusing to guess")

    if DETOUR.exists() and not args.force:
        raise SystemExit(f"already exists: {DETOUR} (use --force only after checking it)")
    if DETOUR.exists():
        shutil.rmtree(DETOUR)
    DETOUR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CURRENT, DETOUR)

    # A geometry-changing derivative must never retain the treatment ID. Preserve
    # the existing witness and add explicit provenance before restoring CURRENT.
    detour_json = DETOUR / "scenario.json"
    detour_data = json.loads(detour_json.read_text())
    detour_data["id"] = DETOUR_ID
    detour_data.setdefault("metadata", {})["detour_variant"] = {
        "parent_scenario_id": SCENARIO_ID,
        "parent_git_commit": ORIGINAL_COMMIT,
        "original_wall_size": [0.025, 0.08, 0.22],
        "modified_wall_size": detour_data["obstacles"][0]["size"],
        "modification_reason": "Full-arm OSC detour could not clear the tall wall; the low wall is retained only as a task-completion existence demo.",
        "openvla_validity_rerun": "Recorded in the pre-existing low-wall metadata as validity-checked; no rerun is performed by this migration.",
        "witness_type": "task_detour",
        "recovery_limitations": "Task completion is demonstrated only for this low-wall geometry and recorded start state; tall-wall d70/d78/d85 detours remain negative results.",
    }
    detour_data.setdefault("metadata", {}).pop("wall_lowered", None)
    detour_json.write_text(json.dumps(detour_data, indent=2) + "\n")

    # Restore the exact pre-detour JSON and binary state/witness from Git history.
    for name in ("scenario.json", "init_state.npy", "witness.npy"):
        (CURRENT / name).write_bytes(git_bytes(RELATIVE / name))

    restored = json.loads((CURRENT / "scenario.json").read_text())
    restored_half_z = float(restored["obstacles"][0]["size"][2])
    if abs(restored_half_z - 0.22) > 1e-6:
        raise SystemExit("restore verification failed: expected original d62 wall geometry")
    print(f"saved lowered task-completion variant: {DETOUR}")
    print(f"restored original tall-wall benchmark: {CURRENT}")
    print("next: run the nominal gate, then baseline/shield jobs")


if __name__ == "__main__":
    main()
