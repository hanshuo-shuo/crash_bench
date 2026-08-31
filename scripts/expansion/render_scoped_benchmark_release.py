#!/usr/bin/env python3
"""Render the scoped D10 support figure with explicit effective source counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
    args.output_dir.mkdir(parents=True)
    width, height = 900, 480
    left, right, top, bottom = 80, 100, 70, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(max(d5_values), max(d8_values), 1)
    group_width = plot_width / len(categories)
    bar_width = 34
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Scoped staleness realized-option heterogeneity support</title>',
        '<desc id="desc">Grouped source-count bars compare D5 train plus development with the once-opened D8 confirmatory test.</desc>',
        '<rect width="900" height="480" fill="white"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">Scoped staleness realized-option heterogeneity support</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
        f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">Independent physical sources</text>',
    ]
    for tick in range(0, maximum + 1, 5):
        y = top + plot_height - tick / maximum * plot_height
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}</text>',
            ]
        )
    for index, (category, d5_value, d8_value) in enumerate(zip(categories, d5_values, d8_values)):
        center = left + group_width * (index + 0.5)
        for offset, value, color in (
            (-bar_width / 2 - 3, d5_value, "#31688e"),
            (bar_width / 2 + 3, d8_value, "#35b779"),
        ):
            bar_height = value / maximum * plot_height
            x = center + offset - bar_width / 2
            y = top + plot_height - bar_height
            elements.extend(
                [
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{color}"/>',
                    f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value}</text>',
                ]
            )
        elements.append(
            f'<text x="{center:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="11">{category}</text>'
        )
    elements.extend(
        [
            '<rect x="210" y="442" width="14" height="14" fill="#31688e"/>',
            '<text x="232" y="454" font-family="sans-serif" font-size="12">D5 train+development (n=36)</text>',
            '<rect x="500" y="442" width="14" height="14" fill="#35b779"/>',
            '<text x="522" y="454" font-family="sans-serif" font-size="12">D8 confirmatory test (n=32)</text>',
            '</svg>',
        ]
    )
    svg = args.output_dir / "figure_scoped_support.svg"
    svg.write_text("\n".join(elements) + "\n")
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
            for path in (svg, caption_path)
        },
        "method_superiority_depicted": False,
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir), "figures": 1, "effective_n": [36, 32]}, sort_keys=True))


if __name__ == "__main__":
    main()
