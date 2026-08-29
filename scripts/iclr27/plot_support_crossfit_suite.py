#!/usr/bin/env python3
"""Plot compact natural/strict/base-preservation summaries for Phase 2.5B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot(input_dir: Path) -> list[Path]:
    input_dir = input_dir.resolve()
    rows = list(csv.DictReader((input_dir / "overall_metrics.csv").open()))
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    roles = manifest["method_roles"]
    shown = [
        row for row in rows
        if roles.get(row["method"]) in {"scalar_risk", "learned", "diagnostic_only"}
    ]
    methods = [row["method"] for row in shown]
    colors = [
        "#999999" if roles[name] == "scalar_risk"
        else ("#d58b33" if roles[name] == "diagnostic_only" else "#3973ac")
        for name in methods
    ]
    positions = np.arange(len(methods))
    figures = input_dir / "figures"
    figures.mkdir(exist_ok=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, max(5.5, 0.34 * len(methods))))
    utility = [float(row["source_macro_utility"]) for row in shown]
    axes[0].barh(positions, utility, color=colors)
    axes[0].axvline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel("OOF source-macro utility")
    axes[0].set_yticks(positions, methods, fontsize=8)

    detour = [float(row["strict_detour_recall"]) for row in shown]
    retreat = [float(row["strict_retreat_recall"]) for row in shown]
    axes[1].scatter(detour, positions, label="Detour", color="#2f78b7")
    axes[1].scatter(retreat, positions, label="Retreat", color="#b94b4b")
    axes[1].axvline(0.55, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_xlabel("Strict-choice recall")
    axes[1].set_yticks([])
    axes[1].legend(loc="lower right", fontsize=8)

    base = [float(row["strict_base_recall"]) for row in shown]
    intervention = [float(row["intervention_rate"]) for row in shown]
    axes[2].scatter(intervention, base, c=colors)
    for x, y, method in zip(intervention, base, methods):
        axes[2].annotate(method, (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6)
    axes[2].axhline(0.80, color="black", linestyle="--", linewidth=0.8)
    axes[2].axvline(0.60, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_xlim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("OOF intervention rate")
    axes[2].set_ylabel("Strict Base recall")

    for axis in axes:
        axis.grid(axis="x", alpha=0.18)
    axes[0].invert_yaxis()
    fig.suptitle("Phase 2.5B source-cross-fitted support rescue")
    fig.tight_layout()
    png = figures / "support_crossfit_summary.png"
    pdf = figures / "support_crossfit_summary.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    manifest["artifact_sha256"].update({
        str(path.relative_to(input_dir)): _sha256(path) for path in (png, pdf)
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = plot(args.input_dir)
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
