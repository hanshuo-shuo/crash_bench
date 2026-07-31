#!/usr/bin/env python3
"""Combine completed wall/glass careful-prompt runs into report-ready artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


LABELS = {
    "vanilla": "Original task only",
    "generic_careful": "Generic careful",
    "hazard_specific": "Hazard-specific",
}


def load_run(path: Path, expected_hazard: str) -> dict:
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    if config.get("hazard") != expected_hazard:
        raise ValueError(
            f"{path}: expected hazard={expected_hazard!r}, got {config.get('hazard')!r}"
        )
    if not payload.get("episodes"):
        raise ValueError(f"{path}: no episodes")
    return payload


def cell(payload: dict, condition: str, regime: str) -> dict:
    return payload["summary"]["by_condition_and_regime"][condition][regime]


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def interval(value: list[float] | None) -> str:
    if value is None:
        return "n/a"
    return f"[{100.0 * value[0]:.1f}, {100.0 * value[1]:.1f}]%"


def interpretation(hazard: str, payload: dict) -> list[str]:
    vanilla = cell(payload, "vanilla", "treatment")
    generic = cell(payload, "generic_careful", "treatment")
    specific = cell(payload, "hazard_specific", "treatment")
    control = cell(payload, "hazard_specific", "control")
    lines = []
    if specific["crash_rate"] < vanilla["crash_rate"]:
        delta = vanilla["crash_rate"] - specific["crash_rate"]
        lines.append(
            f"- **{hazard.title()}:** naming the hazard reduced treatment crash rate "
            f"by {pct(delta)} relative to the original task-only instruction."
        )
    else:
        lines.append(
            f"- **{hazard.title()}:** naming the hazard did not reduce treatment "
            "crash rate relative to the original task-only instruction."
        )
    if specific["crash_rate"] < generic["crash_rate"]:
        lines.append(
            f"  It also outperformed the generic careful wording "
            f"({pct(generic['crash_rate'])} → {pct(specific['crash_rate'])})."
        )
    else:
        lines.append(
            f"  The generic and hazard-specific treatment rates were "
            f"{pct(generic['crash_rate'])} and {pct(specific['crash_rate'])}, respectively."
        )
    lines.append(
        f"  Under the hazard-specific prompt, matched controls had "
        f"{control['n_crash']}/{control['n']} crashes, "
        f"{control['n_recovery_success']}/{control['n']} task successes, and "
        f"{control['n_safe_abort']}/{control['n']} safe aborts."
    )
    return lines


def build_markdown(wall: dict, glass: dict) -> str:
    lines = [
        "# Careful-prompt follow-up",
        "",
        "**Scope audit.** The frozen vanilla scenarios supplied only the original LIBERO "
        "manipulation instruction. They did not ask OpenVLA to avoid the added wall or "
        "glass. Therefore, vanilla crashes measure unprompted safety behavior; they are "
        "not evidence that the model disobeyed an explicit safety request.",
        "",
        "The follow-up changes only the language instruction within each frozen scenario. "
        "It compares the original task, the historical generic careful wording, and a "
        "hazard-specific instruction on matched on-path treatment and off-path control "
        "scenes. Each cell uses K=3 closed-loop repeats.",
        "",
        "| Hazard | Prompt condition | Regime | Crashes | Crash rate (Wilson 95%) | "
        "Task success | Safe abort | Mean global robot-contact peak |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for hazard, payload in (("wall", wall), ("glass", glass)):
        for condition in ("vanilla", "generic_careful", "hazard_specific"):
            for regime in ("treatment", "control"):
                summary = cell(payload, condition, regime)
                lines.append(
                    f"| {hazard} | {LABELS[condition]} | {regime} | "
                    f"{summary['n_crash']}/{summary['n']} | "
                    f"{pct(summary['crash_rate'])} "
                    f"({interval(summary['crash_rate_wilson95'])}) | "
                    f"{summary['n_recovery_success']}/{summary['n']} | "
                    f"{summary['n_safe_abort']}/{summary['n']} | "
                    f"{summary['mean_global_peak_contact_force_n']:.1f} N |"
                )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(interpretation("wall", wall))
    lines.extend(interpretation("glass", glass))
    lines.extend([
        "",
        "A lower crash rate is not automatically task-level success. Safe aborts and "
        "matched-control task success must be read alongside collision outcomes. With "
        "five treatment and five control scenarios per hazard, scenario clustering is "
        "also more important than treating all 15 episode repeats as independent scenes.",
        "",
        "## Exact prompts",
        "",
    ])
    templates = wall["config"]["prompt_templates"]
    for condition in ("vanilla", "generic_careful", "hazard_specific"):
        lines.append(f"- **{LABELS[condition]} / wall:** `{templates[condition]['wall']}`")
        lines.append(f"- **{LABELS[condition]} / glass:** `{templates[condition]['glass']}`")
    lines.extend([
        "",
        "## Provenance",
        "",
        f"- Wall job: `{wall['config'].get('slurm_job_id')}`; "
        f"commit `{wall['config']['git_commit']}`.",
        f"- Glass job: `{glass['config'].get('slurm_job_id')}`; "
        f"commit `{glass['config']['git_commit']}`.",
        "- Full scenario SHA-256 fingerprints, effective per-episode instructions, "
        "checkpoint identity, and raw outcomes are stored in the two source JSON files.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall", default="results/careful_prompt/wall_prompt_matrix.json")
    parser.add_argument("--glass", default="results/careful_prompt/glass_prompt_matrix.json")
    parser.add_argument(
        "--out-json", default="results/careful_prompt/combined_summary.json"
    )
    parser.add_argument("--out-md", default="results/ANALYSIS_careful_prompt.md")
    args = parser.parse_args()
    wall = load_run(Path(args.wall), "wall")
    glass = load_run(Path(args.glass), "glass")
    wall_commit = wall["config"].get("git_commit")
    glass_commit = glass["config"].get("git_commit")
    if wall_commit != glass_commit:
        raise ValueError(
            f"wall/glass commit mismatch: {wall_commit!r} != {glass_commit!r}"
        )
    if wall["config"].get("prompt_templates") != glass["config"].get("prompt_templates"):
        raise ValueError("wall/glass prompt-template metadata differ")

    combined = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": wall_commit,
        "source_files": [args.wall, args.glass],
        "jobs": {
            "wall": wall["config"].get("slurm_job_id"),
            "glass": glass["config"].get("slurm_job_id"),
            "analysis": __import__("os").environ.get("SLURM_JOB_ID"),
        },
        "prompt_templates": wall["config"]["prompt_templates"],
        "hazards": {
            "wall": wall["summary"],
            "glass": glass["summary"],
        },
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    for path in (out_json, out_md):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing analysis: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("x", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, sort_keys=True)
        handle.write("\n")
    out_md.write_text(build_markdown(wall, glass) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
