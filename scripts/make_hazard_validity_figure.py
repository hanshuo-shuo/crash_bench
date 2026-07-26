#!/usr/bin/env python3
"""Render the hazard-validity/generalization outcome figure used in REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/p0_core_20260726_retry1/summary.json"
OUTPUT = ROOT / "setup/figures/fig_hazard_validity_generalization.png"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    outcomes = data["capture"]["outcome_counts"]
    online = data["online_guard"]["headline_metrics"]

    canvas = Image.new("RGB", (1400, 700), "white")
    draw = ImageDraw.Draw(canvas)
    title = font(34, bold=True)
    heading = font(25, bold=True)
    label = font(21)
    small = font(18)

    draw.text(
        (55, 32),
        "Hazard validity and environment generalization: no held-out crashes",
        fill="#20242b",
        font=title,
    )

    # Left: capture wall outcomes.
    draw.text((55, 105), "Capture: wall-present episodes", fill="#20242b", font=heading)
    splits = (("Train", "train", 9), ("Calibration", "calibration", 9), ("Held-out", "heldout", 25))
    colors = {"crash": "#d64545", "task_success": "#3b9b6d", "safe_abort": "#e4a83b"}
    x0, bar_w = 90, 115
    chart_top, chart_bottom = 175, 555
    for index, (display, key, total) in enumerate(splits):
        x = x0 + index * 190
        values = {
            name: int(outcomes.get(f"{key}|wall|{name}", 0))
            for name in ("crash", "task_success", "safe_abort")
        }
        y = chart_bottom
        for name in ("crash", "task_success", "safe_abort"):
            count = values[name]
            height = round((chart_bottom - chart_top) * count / total)
            if height:
                draw.rectangle((x, y - height, x + bar_w, y), fill=colors[name])
                draw.text((x + bar_w / 2, y - height / 2), str(count), fill="white", font=heading, anchor="mm")
            y -= height
        draw.text((x + bar_w / 2, chart_bottom + 24), display, fill="#20242b", font=label, anchor="ma")
        draw.text((x + bar_w / 2, chart_bottom + 52), f"n={total}", fill="#59616c", font=small, anchor="ma")

    legend_x, legend_y = 55, 625
    for name, text_value in (("crash", "Crash"), ("task_success", "Task success"), ("safe_abort", "Safe abort")):
        draw.rectangle((legend_x, legend_y, legend_x + 20, legend_y + 20), fill=colors[name])
        draw.text((legend_x + 30, legend_y - 2), text_value, fill="#303740", font=small)
        legend_x += 155

    # Divider and right: online task success vs false intervention.
    draw.line((700, 105, 700, 640), fill="#d5d8dc", width=2)
    draw.text((750, 105), "Held-out online test (50 episodes per method)", fill="#20242b", font=heading)
    methods = (
        ("Bare OpenVLA", "vanilla"),
        ("Probe guard", "probe_guard"),
        ("Wall presence", "wall_presence"),
        ("Conservative probe", "probe_fpr_00"),
    )
    scale = 250 / 50
    y = 195
    for display, key in methods:
        row = online[key]
        success = round(float(row["task_success_rate"]) * 50)
        false_stop = round(float(row["false_intervention_rate"]) * 50)
        draw.text((750, y), display, fill="#20242b", font=label)
        draw.rectangle((965, y, 965 + success * scale, y + 24), fill="#3b9b6d")
        draw.text((975 + success * scale, y + 12), f"{success}/50 success", fill="#276c4b", font=small, anchor="lm")
        draw.rectangle((965, y + 34, 965 + false_stop * scale, y + 58), fill="#d64545")
        draw.text((975 + false_stop * scale, y + 46), f"{false_stop}/50 false stops", fill="#9f3030", font=small, anchor="lm")
        y += 108

    draw.text((750, 625), "All methods: 0/50 crashes", fill="#20242b", font=heading)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)


if __name__ == "__main__":
    main()
