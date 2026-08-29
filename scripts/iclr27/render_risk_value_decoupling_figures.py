#!/usr/bin/env python3
"""Render the three PIVOT-0 figures with ReportLab.

This dependency-light renderer is invoked only when the analysis runtime does
not provide matplotlib.  It reads already-computed, model-free tables and does
not import CrashBench code, fit a model, or alter any scientific value.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen.canvas import Canvas


PAGE = landscape(letter)
BACKGROUND = HexColor("#f7f4ed")
INK = HexColor("#17323a")
MUTED = HexColor("#51666b")
GRID = HexColor("#cad1cf")
ORANGE = HexColor("#d68155")
TEAL = HexColor("#398783")
LIGHT_TEAL = HexColor("#70b8ad")
GRAY = HexColor("#687b83")
RED = HexColor("#b86969")


def _canvas(path: Path, title: str) -> Canvas:
    canvas = Canvas(str(path), pagesize=PAGE, pageCompression=1)
    canvas.setTitle(title)
    canvas.setAuthor("CrashBench PIVOT-0 evidence synthesis")
    canvas.setFillColor(BACKGROUND)
    canvas.rect(0, 0, PAGE[0], PAGE[1], fill=1, stroke=0)
    return canvas


def _text(
    canvas: Canvas,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 10,
    color=INK,
    bold: bool = False,
    align: str = "left",
) -> None:
    canvas.setFillColor(color)
    canvas.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if align == "center":
        canvas.drawCentredString(x, y, value)
    elif align == "right":
        canvas.drawRightString(x, y, value)
    else:
        canvas.drawString(x, y, value)


def _wrapped(
    canvas: Canvas,
    x: float,
    y: float,
    value: str,
    *,
    width: int,
    size: float = 9,
    leading: float = 11,
    color=MUTED,
    bold: bool = False,
    align: str = "left",
) -> None:
    for offset, line in enumerate(textwrap.wrap(value, width=width)):
        _text(
            canvas,
            x,
            y - offset * leading,
            line,
            size=size,
            color=color,
            bold=bold,
            align=align,
        )


def _finish(canvas: Canvas, path: Path) -> None:
    canvas.showPage()
    canvas.save()
    if path.stat().st_size < 1000 or path.read_bytes()[:4] != b"%PDF":
        raise ValueError(f"ReportLab did not create a valid PDF: {path}")


def render_risk_benefit(spec: Mapping[str, Any], path: Path) -> None:
    cells = {
        (int(row["risk"]), int(row["benefit"])): row
        for row in spec["risk_benefit_crosstab"]
        if row["row_type"] == "cell"
    }
    if set(cells) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("risk-benefit figure requires all four cells")
    colors = {
        (0, 0): HexColor("#d9e4df"),
        (0, 1): HexColor("#f2c879"),
        (1, 0): HexColor("#e9957c"),
        (1, 1): HexColor("#6ca6a3"),
    }
    canvas = _canvas(path, "Risk is not intervention benefit")
    _text(canvas, 48, 558, "Risk is not intervention benefit", size=24, bold=True)
    _text(
        canvas, 48, 538,
        "Frozen exact-state outcomes | 273 decisions | 20 independent sources",
        size=10.5, color=MUTED,
    )
    box_w, box_h = 248, 154
    x0, y0 = 192, 318
    x_gap, y_gap = 264, 170
    for risk in (0, 1):
        for benefit in (0, 1):
            row = cells[(risk, benefit)]
            x = x0 + benefit * x_gap
            y = y0 - risk * y_gap
            canvas.setFillColor(colors[(risk, benefit)])
            canvas.roundRect(x, y, box_w, box_h, 12, fill=1, stroke=0)
            _text(
                canvas, x + box_w / 2, y + 92,
                f"{int(row['raw_count'])} states",
                size=20, bold=True, align="center",
            )
            _text(
                canvas, x + box_w / 2, y + 62,
                f"{100 * float(row['source_macro_rate']):.1f}% source-macro mass",
                size=10, color=MUTED, align="center",
            )
            _text(
                canvas, x + box_w / 2, y + 38,
                f"{int(row['independent_sources_with_event'])} supporting sources",
                size=9.5, color=MUTED, align="center",
            )
    _text(canvas, 176, y0 + 75, "Base safe", size=11, bold=True, align="right")
    _text(canvas, 176, y0 + 58, "R=0", size=10, color=MUTED, align="right")
    _text(canvas, 176, y0 - y_gap + 75, "Base catastrophe", size=11, bold=True, align="right")
    _text(canvas, 176, y0 - y_gap + 58, "R=1", size=10, color=MUTED, align="right")
    _text(canvas, x0 + box_w / 2, 118, "No intervention benefit", size=10.5, bold=True, align="center")
    _text(canvas, x0 + box_w / 2, 102, "B=0", size=9.5, color=MUTED, align="center")
    _text(canvas, x0 + x_gap + box_w / 2, 118, "Positive intervention benefit", size=10.5, bold=True, align="center")
    _text(canvas, x0 + x_gap + box_w / 2, 102, "B=1", size=9.5, color=MUTED, align="center")
    _text(
        canvas, PAGE[0] / 2, 56,
        "19 disagreements across 12 sources: 10 risk-positive/no-benefit + 9 risk-negative/positive-benefit",
        size=10.5, color=HexColor("#9a3e2f"), bold=True, align="center",
    )
    _finish(canvas, path)


def _stage_label(stage: str) -> list[str]:
    replacements = {
        "Risk -> Best Fixed": ["Risk ->", "Best Fixed"],
        "LearnedGate + OracleChoice": ["LearnedGate +", "OracleChoice"],
        "OracleGate + learned linear choice": ["OracleGate +", "linear choice"],
        "OracleGate + learned tiny nonlinear choice": ["OracleGate +", "tiny choice"],
        "full learned ADR": ["Full tiny ADR", "diagnostic"],
        "OracleGate + OracleChoice": ["OracleGate +", "OracleChoice"],
    }
    return replacements.get(stage, textwrap.wrap(stage, width=16))[:2]


def render_waterfall(spec: Mapping[str, Any], path: Path) -> None:
    rows = sorted(spec["oracle_hybrid_waterfall"], key=lambda row: int(row["order"]))
    if len(rows) != 6:
        raise ValueError("oracle-gap figure requires the six frozen stages")
    values = [float(row["source_macro_utility"]) for row in rows]
    oracle = values[-1]
    colors = [GRAY, ORANGE, TEAL, LIGHT_TEAL, RED, INK]
    canvas = _canvas(path, "Gate-choice composition waterfall")
    _text(canvas, 44, 558, "The learned benefit gate dominates the frozen oracle gap", size=22, bold=True)
    _text(
        canvas, 44, 536,
        f"Tiny-choice loss = {float(rows[0]['oracle_to_tiny_choice_utility_loss']):.4f} | "
        f"learned-gate loss = {float(rows[0]['oracle_to_learned_gate_utility_loss']):.4f}",
        size=10.5, color=MUTED,
    )
    chart_x, chart_y, chart_w, chart_h = 58, 142, 690, 340
    max_y = 0.62
    for tick in (0.0, 0.2, 0.4, 0.6):
        y = chart_y + tick / max_y * chart_h
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.6)
        canvas.line(chart_x, y, chart_x + chart_w, y)
        _text(canvas, chart_x - 10, y - 3, f"{tick:.1f}", size=8.5, color=MUTED, align="right")
    oracle_y = chart_y + oracle / max_y * chart_h
    canvas.setStrokeColor(INK)
    canvas.setDash(5, 4)
    canvas.line(chart_x, oracle_y, chart_x + chart_w, oracle_y)
    canvas.setDash()
    _text(canvas, chart_x + chart_w - 2, oracle_y + 6, "Oracle", size=8.5, color=INK, align="right")
    slot = chart_w / len(rows)
    bar_w = 76
    for index, (row, value, color) in enumerate(zip(rows, values, colors)):
        x = chart_x + index * slot + (slot - bar_w) / 2
        height = max(0.0, value / max_y * chart_h)
        canvas.setFillColor(color)
        canvas.roundRect(x, chart_y, bar_w, height, 4, fill=1, stroke=0)
        _text(canvas, x + bar_w / 2, chart_y + height + 9, f"{value:.3f}", size=10, bold=True, align="center")
        for line_index, line in enumerate(_stage_label(str(row["stage"]))):
            _text(
                canvas, x + bar_w / 2, 116 - line_index * 12,
                line, size=8.2, color=INK, bold=line_index == 0, align="center",
            )
    _text(canvas, 18, chart_y + chart_h / 2, "Source-macro utility", size=9, color=MUTED)
    _text(
        canvas, 44, 46,
        "Oracle combinations are diagnostic only. The tiny nonlinear head is a capacity diagnostic, not the paper method.",
        size=9.5, color=HexColor("#9a3e2f"), bold=True,
    )
    _finish(canvas, path)


def render_source_support(spec: Mapping[str, Any], path: Path) -> None:
    rows = [row for row in spec["family_support"] if row["breakdown"] == "condition"]
    order = {"glass": 0, "offpath": 1, "noglass": 2}
    rows.sort(key=lambda row: order.get(str(row["value"]), 99))
    if len(rows) != 3:
        raise ValueError("source-support figure requires three condition rows")
    series = [
        ("disagreement_sources", "R/B disagreement", ORANGE),
        ("strict_base_sources", "Strict Base", GRAY),
        ("strict_detour_sources", "Strict Detour", TEAL),
        ("strict_retreat_sources", "Strict Retreat", RED),
    ]
    canvas = _canvas(path, "Source support is mechanically glass-scoped")
    _text(canvas, 44, 558, "Source support is real, but mechanically glass-scoped", size=22, bold=True)
    _text(
        canvas, 44, 536,
        "All three condition labels share one glass-recovery design; all 23 strict Retreat states and 51/60 strict D/R states are glass",
        size=9.8, color=MUTED,
    )
    chart_x, chart_y, chart_w, chart_h = 72, 150, 660, 330
    max_y = 20.0
    for tick in (0, 5, 10, 15, 20):
        y = chart_y + tick / max_y * chart_h
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.6)
        canvas.line(chart_x, y, chart_x + chart_w, y)
        _text(canvas, chart_x - 10, y - 3, str(tick), size=8.5, color=MUTED, align="right")
    group_w = chart_w / len(rows)
    bar_w = 34
    gap = 8
    for group_index, row in enumerate(rows):
        total_w = len(series) * bar_w + (len(series) - 1) * gap
        start = chart_x + group_index * group_w + (group_w - total_w) / 2
        for series_index, (key, _, color) in enumerate(series):
            value = int(row[key])
            x = start + series_index * (bar_w + gap)
            height = value / max_y * chart_h
            if value > 0:
                canvas.setFillColor(color)
                canvas.roundRect(x, chart_y, bar_w, height, 3, fill=1, stroke=0)
            _text(canvas, x + bar_w / 2, chart_y + height + 8, str(value), size=9, bold=True, align="center")
        _text(
            canvas, chart_x + group_index * group_w + group_w / 2, 126,
            str(row["value"]), size=11, bold=True, align="center",
        )
        _text(
            canvas, chart_x + group_index * group_w + group_w / 2, 109,
            f"{int(row['independent_sources'])} observed sources",
            size=8.5, color=MUTED, align="center",
        )
    legend_x, legend_y = 82, 74
    for index, (_, label, color) in enumerate(series):
        x = legend_x + index * 168
        canvas.setFillColor(color)
        canvas.rect(x, legend_y, 12, 12, fill=1, stroke=0)
        _text(canvas, x + 18, legend_y + 2, label, size=8.7, color=INK)
    _text(
        canvas, 44, 38,
        "Offpath and noglass are matched controls, not independent mechanical hazard families. Glass strict D/R: 51 states / 16 sources; all strict D/R: 60 / 17.",
        size=9.2, color=HexColor("#9a3e2f"), bold=True,
    )
    _finish(canvas, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "figure_risk_benefit_flow.pdf": render_risk_benefit,
        "figure_oracle_gap_waterfall.pdf": render_waterfall,
        "figure_source_support.pdf": render_source_support,
    }
    for name, renderer in outputs.items():
        renderer(spec, args.output_dir / name)


if __name__ == "__main__":
    main()
