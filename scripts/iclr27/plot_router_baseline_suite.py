#!/usr/bin/env python3
"""Render deterministic, dependency-free SVGs for the router baseline audit."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


WIDTH = 980
HEIGHT = 650
MARGIN_LEFT = 92
MARGIN_RIGHT = 245
MARGIN_TOP = 70
MARGIN_BOTTOM = 82
PALETTE = (
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
    "#be123c", "#4f46e5", "#65a30d", "#c2410c", "#0f766e", "#9333ea",
    "#475569", "#ca8a04", "#0369a1", "#b91c1c", "#15803d", "#6d28d9",
)


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _primary_method(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    for path in (("primary_method",), ("protocol", "primary_method"), ("primary", "method")):
        value: Any = manifest
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, Mapping):
            value = value.get("name")
        if value is not None and any(str(row.get("method")) == str(value) for row in rows):
            return str(value)
    marked = [str(row["method"]) for row in rows if _truth(row.get("is_primary_method"))]
    if marked:
        return marked[0]
    names = [str(row.get("method", "")) for row in rows]
    full = [name for name in names if "outcome" in name.lower() and "only" not in name.lower()]
    return full[0] if full else names[0]


def _nice_extent(values: Sequence[float], *, rates: bool = False) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return (0.0, 1.0)
    if rates:
        low = max(0.0, min(finite) - 0.04)
        high = min(1.0, max(finite) + 0.04)
        if high - low < 0.2:
            center = (low + high) / 2
            low, high = max(0.0, center - 0.1), min(1.0, center + 0.1)
    else:
        low, high = min(finite), max(finite)
        span = high - low
        padding = 0.08 * span if span > 1e-12 else max(0.1, abs(low) * 0.1)
        low, high = low - padding, high + padding
    if high <= low:
        high = low + 1.0
    return (low, high)


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    return [low + (high - low) * index / count for index in range(count + 1)]


def _fmt(value: float, rate: bool) -> str:
    return f"{100 * value:.0f}%" if rate else f"{value:.2f}"


def _svg_plot(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
    x_rate: bool,
    y_rate: bool,
    primary: str,
    size_key: str | None = None,
    frontier_key: str | None = None,
) -> None:
    valid = [
        row for row in rows
        if math.isfinite(_float(row, x_key)) and math.isfinite(_float(row, y_key))
    ]
    if not valid:
        raise ValueError(f"frontier.csv has no finite {x_key}/{y_key} rows")
    x_low, x_high = _nice_extent([_float(row, x_key) for row in valid], rates=x_rate)
    y_low, y_high = _nice_extent([_float(row, y_key) for row in valid], rates=y_rate)
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    def x_position(value: float) -> float:
        return MARGIN_LEFT + (value - x_low) / (x_high - x_low) * plot_width

    def y_position(value: float) -> float:
        return MARGIN_TOP + (y_high - value) / (y_high - y_low) * plot_height

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)}</desc>",
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172033}",
        ".grid{stroke:#dbe2ea;stroke-width:1}.axis{stroke:#475569;stroke-width:1.4}",
        ".tick{font-size:12px;fill:#64748b}.label{font-size:14px;font-weight:600}",
        ".legend{font-size:12px}.title{font-size:22px;font-weight:700}.subtitle{font-size:12px;fill:#64748b}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{MARGIN_LEFT}" y="31">{html.escape(title)}</text>',
        f'<text class="subtitle" x="{MARGIN_LEFT}" y="51">{html.escape(subtitle)}</text>',
    ]
    for value in _ticks(x_low, x_high):
        x = x_position(value)
        parts.extend([
            f'<line class="grid" x1="{x:.2f}" y1="{MARGIN_TOP}" x2="{x:.2f}" y2="{MARGIN_TOP + plot_height}"/>',
            f'<text class="tick" text-anchor="middle" x="{x:.2f}" y="{MARGIN_TOP + plot_height + 24}">{_fmt(value, x_rate)}</text>',
        ])
    for value in _ticks(y_low, y_high):
        y = y_position(value)
        parts.extend([
            f'<line class="grid" x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{MARGIN_LEFT + plot_width}" y2="{y:.2f}"/>',
            f'<text class="tick" text-anchor="end" x="{MARGIN_LEFT - 12}" y="{y + 4:.2f}">{_fmt(value, y_rate)}</text>',
        ])
    parts.extend([
        f'<line class="axis" x1="{MARGIN_LEFT}" y1="{MARGIN_TOP + plot_height}" x2="{MARGIN_LEFT + plot_width}" y2="{MARGIN_TOP + plot_height}"/>',
        f'<line class="axis" x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{MARGIN_TOP + plot_height}"/>',
        f'<text class="label" text-anchor="middle" x="{MARGIN_LEFT + plot_width / 2:.2f}" y="{HEIGHT - 25}">{html.escape(x_label)}</text>',
        f'<text class="label" text-anchor="middle" transform="translate(24 {MARGIN_TOP + plot_height / 2:.2f}) rotate(-90)">{html.escape(y_label)}</text>',
    ])

    # Draw non-primary points first so the frozen primary operating point stays visible.
    ordered = sorted(enumerate(valid), key=lambda item: str(item[1].get("method")))
    ordered.sort(key=lambda item: str(item[1].get("method")) == primary)
    color_by_method = {
        str(row.get("method")): PALETTE[index % len(PALETTE)]
        for index, row in enumerate(sorted(valid, key=lambda item: str(item.get("method"))))
    }
    for _, row in ordered:
        method = str(row.get("method"))
        x, y = x_position(_float(row, x_key)), y_position(_float(row, y_key))
        is_primary = method == primary
        on_frontier = frontier_key is not None and _truth(row.get(frontier_key))
        radius = 6.0
        if size_key is not None and math.isfinite(_float(row, size_key)):
            radius = 5.0 + 6.0 * max(0.0, min(1.0, _float(row, size_key)))
        stroke = "#0f172a" if is_primary else ("#334155" if on_frontier else "#ffffff")
        stroke_width = 3.0 if is_primary else (1.5 if on_frontier else 1.0)
        parts.extend([
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color_by_method[method]}" fill-opacity="0.84" stroke="{stroke}" stroke-width="{stroke_width}">',
            f'<title>{html.escape(method)}: {html.escape(x_label)} {_fmt(_float(row, x_key), x_rate)}, {html.escape(y_label)} {_fmt(_float(row, y_key), y_rate)}</title>',
            "</circle>",
        ])
        if is_primary:
            parts.append(
                f'<text class="legend" font-weight="700" x="{x + radius + 5:.2f}" y="{y - radius - 2:.2f}">{html.escape(method)}</text>'
            )

    legend_x = WIDTH - MARGIN_RIGHT + 31
    legend_y = MARGIN_TOP + 4
    parts.append(f'<text class="label" x="{legend_x}" y="{legend_y}">Methods</text>')
    for index, row in enumerate(sorted(valid, key=lambda item: str(item.get("method")))):
        method = str(row.get("method"))
        y = legend_y + 22 + index * 24
        if y > HEIGHT - 15:
            # All primary methods fit in the normal 18-method suite; this guard
            # keeps arbitrary exploratory additions from overflowing the SVG.
            break
        stroke = "#0f172a" if method == primary else "#ffffff"
        parts.extend([
            f'<circle cx="{legend_x + 6}" cy="{y - 4}" r="5" fill="{color_by_method[method]}" stroke="{stroke}" stroke-width="{2 if method == primary else 1}"/>',
            f'<text class="legend" x="{legend_x + 18}" y="{y}">{html.escape(method)}</text>',
        ])
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot(input_dir: Path) -> dict[str, Path]:
    input_dir = input_dir.resolve()
    frontier_path = input_dir / "frontier.csv"
    manifest_path = input_dir / "manifest.json"
    if not frontier_path.is_file():
        raise FileNotFoundError(frontier_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with frontier_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("frontier.csv is empty")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary = _primary_method(manifest, rows)
    figures = input_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = {
        "utility_vs_intervention": figures / "utility_vs_intervention.svg",
        "success_vs_catastrophe": figures / "success_vs_catastrophe.svg",
    }
    _svg_plot(
        rows,
        outputs["utility_vs_intervention"],
        x_key="intervention_rate",
        y_key="mean_utility",
        x_label="Intervention rate (lower is better)",
        y_label="Mean utility (higher is better)",
        title="Development utility–intervention frontier",
        subtitle="One frozen operating point per method; outlined points are nondominated.",
        x_rate=True,
        y_rate=False,
        primary=primary,
        frontier_key="is_utility_intervention_frontier",
    )
    _svg_plot(
        rows,
        outputs["success_vs_catastrophe"],
        x_key="catastrophe_rate",
        y_key="task_success_rate",
        x_label="Catastrophe rate (lower is better)",
        y_label="Task success rate (higher is better)",
        title="Development safety–success frontier",
        subtitle="Marker area encodes intervention rate; oracle geometry remains diagnostic only.",
        x_rate=True,
        y_rate=True,
        primary=primary,
        size_key="intervention_rate",
        frontier_key="is_success_catastrophe_frontier",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing manifest.json and frontier.csv; writes figures/ in place.",
    )
    args = parser.parse_args()
    outputs = plot(args.input_dir)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
