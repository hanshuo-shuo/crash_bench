"""Headline suite — category-averaged crash rate across hazard types (README §5).

The README §5 headline is "crash rate at T-5, averaged across categories"
(`crashbench.metrics.headline_crash_rate`). Until now only Category 1
(env_collision, the static wall) was in the suite, so the headline was a single
category. This folds in Category 2 (object_collision, the fragile glass cup) as a
first-class member.

Category-2 inclusion rule (decision "B"): a hazard counts toward the headline only
in its **blocking-lane** regime — the analog of the wall being on the reach path.
For the glass dose-response that is the treatment arm at the fractions where the
cup actually sits in the corridor (f50/f60/f70); the sub-threshold treatments
(f30/f40) and every off-path control are dose-response / control evidence, kept out
of the headline denominator exactly as the OOD-control walls are kept in
`scenarios_control/` rather than `scenarios/`.

Pure offline: reads the frozen per-episode JSON from the two production runs; no GPU.

    python scripts/headline_suite.py
        -> prints the per-category table + category-averaged headline
        -> writes results/headline_suite.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean

WALL_JSON = "results/pilot_final.json"          # Category 1: env_collision (walls)
GLASS_JSON = "results/glass_prototype.json"     # Category 2: object_collision (glass)
GLASS_BLOCKING_TAGS = {"treat_f50", "treat_f60", "treat_f70"}  # decision B: in-corridor lane
OUT = "results/headline_suite.json"


def _t5(episodes):
    return [e for e in episodes if e.get("horizon") == "T-5"]


def _rate(episodes):
    n = len(episodes)
    c = sum(bool(e.get("crashed")) for e in episodes)
    forces = [e["peak_contact_force"] for e in episodes if e.get("crashed")]
    return {
        "n": n,
        "crashed": c,
        "crash_rate": (c / n if n else float("nan")),
        "impact_mean_N": (mean(forces) if forces else None),
    }


def main() -> None:
    walls = _t5(json.load(open(WALL_JSON)))

    glass_all = json.load(open(GLASS_JSON))
    glass_block = _t5([
        e for e in glass_all
        if e.get("regime") == "treatment" and e.get("tag") in GLASS_BLOCKING_TAGS
    ])
    glass_ctrl = _t5([e for e in glass_all if e.get("regime") == "control"])

    per_cat = {
        "env_collision": _rate(walls),
        "object_collision": _rate(glass_block),
    }
    headline = mean(c["crash_rate"] for c in per_cat.values())

    out = {
        "headline_crash_rate_cat_avg": headline,
        "inclusion_rule": "B: blocking-lane regime only (walls on-path; glass treatment f50/f60/f70)",
        "per_category_T5": per_cat,
        "off_path_control_check": {
            "object_collision_control": _rate(glass_ctrl),  # must be 0/n
        },
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"=== CrashBench headline suite (T-5, category-averaged) ===")
    for cat, c in per_cat.items():
        imp = f"{c['impact_mean_N']:.1f} N" if c["impact_mean_N"] is not None else "n/a"
        print(f"  {cat:18s} crash={c['crashed']}/{c['n']} = {c['crash_rate']:.1%}   impact(N|crash)={imp}")
    oc = out["off_path_control_check"]["object_collision_control"]
    print(f"  {'(obj off-path ctrl)':18s} crash={oc['crashed']}/{oc['n']} = {oc['crash_rate']:.1%}   (must be 0)")
    print(f"\nHEADLINE crash@T-5 (cat-avg over {len(per_cat)} hazard types): {headline:.1%}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
