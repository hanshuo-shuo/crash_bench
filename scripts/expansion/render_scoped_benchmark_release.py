#!/usr/bin/env python3
"""Render the scoped D10 support figure with explicit effective source counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-gate-a", type=Path, required=True)
    parser.add_argument("--d8-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite D10 figure root: {args.output_dir}")
    d5 = json.loads(args.corrected_gate_a.read_text())
    d8 = json.loads(args.d8_analysis.read_text())
    if d5.get("calibration_rows_read") != 0 or d8.get("method_superiority_evaluated") is not False:
        raise ValueError("scoped figure inputs cross the frozen information boundary")
    categories = ("B=0", "B=1", "Refresh strict", "Safe-stop strict", "Same-risk flip")
    d5_values = (
        d5["benefit_zero_sources"], d5["benefit_one_sources"],
        d5["strict_support_sources"]["observation_refresh"],
        d5["strict_support_sources"]["safe_stop"], d5["same_risk_flip_sources"],
    )
    primary = d8["primary"]
    d8_values = (
        primary["benefit_zero_sources"], primary["benefit_one_sources"],
        primary["strict_support_sources"]["observation_refresh"],
        primary["strict_support_sources"]["safe_stop"], primary["same_risk_flip_sources"],
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True)
    x = np.arange(len(categories))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    train_bars = axis.bar(x - width / 2, d5_values, width, label="D5 train+development (n=36)", color="#31688e")
    test_bars = axis.bar(x + width / 2, d8_values, width, label="D8 confirmatory test (n=32)", color="#35b779")
    axis.bar_label(train_bars, padding=2, fontsize=8)
    axis.bar_label(test_bars, padding=2, fontsize=8)
    axis.set_ylabel("Independent physical sources")
    axis.set_xticks(x, categories, rotation=15, ha="right")
    axis.set_title("Scoped staleness realized-option heterogeneity support")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = args.output_dir / "figure_scoped_support.png"
    pdf = args.output_dir / "figure_scoped_support.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    caption = (
        "**Scoped realized-option support.** Counts use independent physical sources, not anchors or branches. "
        "D5 includes train+development only (effective n=36; all 12 calibration sources excluded). "
        "D8 is the once-opened confirmatory split (effective n=32; 16 per task). B indicates whether any "
        "deployable non-Base option has positive realized U0 advantage; strict winners require the frozen "
        "0.10 utility gap. This single-policy, single-staleness panel does not support multi-mechanism or "
        "learned-method superiority claims."
    )
    caption_path = args.output_dir / "figure_scoped_support_caption.md"
    caption_path.write_text(caption + "\n")
    manifest = {
        "schema_version": 1,
        "kind": "crashbench_expansion_scoped_release_figure_manifest",
        "effective_n": {"D5_train_development": 36, "D8_confirmatory_test": 32},
        "roles": {"D5": ["train", "development"], "D8": ["confirmatory_id_test"]},
        "source_counts": {"D5": dict(zip(categories, d5_values)), "D8": dict(zip(categories, d8_values))},
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (png, pdf, caption_path)
        },
        "method_superiority_depicted": False,
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir), "figures": 1, "effective_n": [36, 32]}, sort_keys=True))


if __name__ == "__main__":
    main()
