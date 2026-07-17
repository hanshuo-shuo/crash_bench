#!/usr/bin/env python3
"""Regenerate the tracked scenario fingerprint manifest without touching scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashbench.scenario import scenario_fingerprint

SCENARIO_ROOTS = (
    "scenarios",
    "scenarios_control",
    "scenarios_glass",
    "scenarios_detour_lowwall",
)
OUT = ROOT / "results" / "scenario_fingerprints.json"


def build() -> dict:
    rows = []
    for root_name in SCENARIO_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.glob("*/scenario.json")):
            data = json.loads(path.read_text())
            scenario_dir = path.parent
            rows.append({
                "id": data["id"],
                "root": root_name,
                "path": str(scenario_dir.relative_to(ROOT)),
                "fingerprint_sha256": scenario_fingerprint(scenario_dir),
                "schema_version": data.get("schema_version", 1),
            })
    return {
        "schema_version": 1,
        "algorithm": "sha256(filename + NUL + exact file bytes for scenario.json, init_state.npy, witness.npy when present)",
        "generated_by": "scripts/update_scenario_manifest.py",
        "scenarios": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the existing tracked fingerprint manifest")
    args = parser.parse_args()
    if OUT.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {OUT}; pass --overwrite after review")
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
