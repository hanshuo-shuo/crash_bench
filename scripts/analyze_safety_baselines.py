#!/usr/bin/env python3
"""Merge safety-baseline JSON files and create report-ready PNG figures with Pillow."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ORDER = ["vanilla", "prompted_careful", "vlm_monitor", "safety_filter", "oracle_stop"]
LABEL = {
    "vanilla": "Vanilla",
    "prompted_careful": "Prompted-careful",
    "vlm_monitor": "Qwen VLM monitor",
    "safety_filter": "Signed-distance CBF",
    "oracle_stop": "Oracle-stop proxy",
}
COLORS = {
    "vanilla": "#B42318",
    "prompted_careful": "#D97706",
    "vlm_monitor": "#2563EB",
    "safety_filter": "#7C3AED",
    "oracle_stop": "#059669",
}
OUTCOME_COLORS = {
    "crash": "#B42318",
    "safe_abort": "#2563EB",
    "recovery_success": "#059669",
    "timeout": "#94A3B8",
}


def font(size: int, bold: bool = False):
    names = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"] if bold else
             ["/System/Library/Fonts/Supplemental/Arial.ttf"])
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def load_rows(paths: list[str]) -> tuple[list[dict], list[dict]]:
    payloads, rows = [], []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        payloads.append(payload)
        rows.extend(payload["episodes"])
    return payloads, rows


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for condition in ORDER:
        group = [row for row in rows if row["condition"] == condition]
        if not group:
            continue
        outcomes = Counter(row["outcome"] for row in group)
        crashes = outcomes["crash"]
        lo, hi = wilson(crashes, len(group))
        step_rows = [step for row in group for step in row.get("steps", [])]
        monitor = [step["monitor"] for step in step_rows if "monitor" in step]
        summary[condition] = {
            "n": len(group),
            "outcomes": dict(outcomes),
            "crash_rate": crashes / len(group),
            "crash_rate_wilson95": [lo, hi],
            "mean_peak_force_all_n": float(np.mean([r["peak_contact_force"] for r in group])),
            "mean_peak_force_crashes_n": (float(np.mean(
                [r["peak_contact_force"] for r in group if r["crashed"]]
            )) if crashes else 0.0),
            "interventions": sum(r.get("n_interventions", 0) for r in group),
            "steps": len(step_rows),
            "intervention_rate": (sum(r.get("n_interventions", 0) for r in group) /
                                  len(step_rows) if step_rows else 0.0),
            "monitor_queries": len(monitor),
            "monitor_yes": sum(bool(x["collision"]) for x in monitor),
            "monitor_yes_rate": (sum(bool(x["collision"]) for x in monitor) / len(monitor)
                                 if monitor else None),
            "monitor_latency_mean_s": (float(np.mean([x["latency_s"] for x in monitor]))
                                       if monitor else None),
            "monitor_parse_status": dict(Counter(x["parse_status"] for x in monitor)),
        }
    return summary


def _center(draw, box, text, text_font, fill="#111827"):
    bbox = draw.textbbox((0, 0), text, font=text_font)
    x = (box[0] + box[2] - (bbox[2] - bbox[0])) / 2
    y = (box[1] + box[3] - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), text, font=text_font, fill=fill)


def comparison_figure(summary: dict, out: Path) -> None:
    w, h = 1800, 1000
    im = Image.new("RGB", (w, h), "#FFFFFF")
    d = ImageDraw.Draw(im)
    d.text((80, 45), "OpenVLA safety baseline comparison", font=font(48, True), fill="#0F172A")
    d.text((80, 105), "Frozen on-path wall scenarios; error bars are 95% Wilson intervals",
           font=font(25), fill="#475569")
    conditions = [c for c in ORDER if c in summary]
    left, top, bottom, right = 105, 235, 830, 1710
    chart_w = right - left
    for j in range(6):
        y = bottom - j * (bottom - top) / 5
        d.line((left, y, right, y), fill="#E2E8F0", width=2)
        d.text((35, y - 14), f"{j * 20}%", font=font(21), fill="#64748B")
    slot = chart_w / len(conditions)
    bar_w = slot * 0.50
    for i, cond in enumerate(conditions):
        s = summary[cond]
        cx = left + slot * (i + 0.5)
        rate = s["crash_rate"]
        y = bottom - rate * (bottom - top)
        d.rounded_rectangle((cx - bar_w/2, y, cx + bar_w/2, bottom), radius=12,
                            fill=COLORS[cond])
        lo, hi = s["crash_rate_wilson95"]
        ylo, yhi = bottom - lo * (bottom-top), bottom - hi * (bottom-top)
        d.line((cx, yhi, cx, ylo), fill="#111827", width=5)
        d.line((cx-15, yhi, cx+15, yhi), fill="#111827", width=5)
        d.line((cx-15, ylo, cx+15, ylo), fill="#111827", width=5)
        _center(d, (cx-90, max(155, y-55), cx+90, max(200, y-5)), f"{rate:.0%}", font(28, True))
        _center(d, (cx-slot/2+8, bottom+20, cx+slot/2-8, bottom+68), LABEL[cond], font(22, True))
        _center(d, (cx-slot/2+8, bottom+68, cx+slot/2-8, bottom+105), f"n={s['n']}", font(20), "#64748B")
    d.text((left, 920), "Primary endpoint: collision predicate fired before success or stable stop.",
           font=font(22), fill="#475569")
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def outcomes_figure(summary: dict, out: Path) -> None:
    w, h = 1800, 980
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((80, 45), "Outcome composition and intervention burden", font=font(46, True), fill="#0F172A")
    d.text((80, 105), "Safe stop is not task completion", font=font(26), fill="#475569")
    conditions = [c for c in ORDER if c in summary]
    left, right, top = 390, 1690, 205
    row_h, bar_h = 135, 62
    for i, cond in enumerate(conditions):
        s = summary[cond]
        y = top + i * row_h
        d.text((80, y+9), LABEL[cond], font=font(24, True), fill="#111827")
        d.text((80, y+47), f"intervention {s['intervention_rate']:.0%}",
               font=font(20), fill="#64748B")
        x = left
        for outcome in ["crash", "safe_abort", "recovery_success", "timeout"]:
            count = s["outcomes"].get(outcome, 0)
            width = (right-left) * count / s["n"]
            if width:
                d.rectangle((x, y, x+width, y+bar_h), fill=OUTCOME_COLORS[outcome])
                if width > 75:
                    _center(d, (x, y, x+width, y+bar_h), str(count), font(22, True), "white")
            x += width
        d.rounded_rectangle((left, y, right, y+bar_h), radius=6, outline="#CBD5E1", width=2)
    legend_x, legend_y = 390, 900
    for outcome, label in [("crash", "Crash"), ("safe_abort", "Safe abort"),
                           ("recovery_success", "Task success"), ("timeout", "Timeout")]:
        d.rectangle((legend_x, legend_y, legend_x+28, legend_y+28), fill=OUTCOME_COLORS[outcome])
        d.text((legend_x+40, legend_y-1), label, font=font(21), fill="#334155")
        legend_x += 250
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def monitor_figure(rows: list[dict], out: Path) -> None:
    group = [row for row in rows if row["condition"] == "vlm_monitor"]
    if not group:
        return
    scenarios = sorted({row["scenario_id"] for row in group})
    short = {s: s.rsplit("_", 1)[-1] for s in scenarios}
    w, h = 1800, 980
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((80, 45), "Qwen monitor diagnostics", font=font(46, True), fill="#0F172A")
    d.text((80, 105), "Every OpenVLA proposal is queried; YES executes zero motion",
           font=font(26), fill="#475569")
    left, right, top, bottom = 150, 1680, 250, 760
    slot = (right-left) / len(scenarios)
    max_latency = 0.0
    values = []
    for scenario in scenarios:
        decisions = [step["monitor"] for row in group if row["scenario_id"] == scenario
                     for step in row.get("steps", []) if "monitor" in step]
        yes = sum(x["collision"] for x in decisions)
        values.append((yes / len(decisions), float(np.mean([x["latency_s"] for x in decisions])), len(decisions)))
        max_latency = max(max_latency, values[-1][1])
    for i, scenario in enumerate(scenarios):
        yes_rate, latency, queries = values[i]
        cx = left + slot*(i+0.5)
        height = yes_rate*(bottom-top)
        d.rounded_rectangle((cx-slot*.23, bottom-height, cx+slot*.23, bottom), radius=10,
                            fill="#2563EB")
        _center(d, (cx-75, bottom-height-50, cx+75, bottom-height-5), f"{yes_rate:.0%}", font(26, True))
        _center(d, (cx-slot/2, bottom+18, cx+slot/2, bottom+55), short[scenario], font(23, True))
        _center(d, (cx-slot/2, bottom+55, cx+slot/2, bottom+92),
                f"{queries} queries · {latency:.1f}s", font(18), "#64748B")
    for j in range(6):
        y = bottom-j*(bottom-top)/5
        d.line((left, y, right, y), fill="#E2E8F0", width=2)
        d.text((75, y-13), f"{j*20}%", font=font(20), fill="#64748B")
    statuses = Counter(step["monitor"]["parse_status"] for row in group
                       for step in row.get("steps", []) if "monitor" in step)
    d.text((120, 890), "Parse status: " + ", ".join(f"{k}={v}" for k, v in statuses.items()),
           font=font(22), fill="#475569")
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def final_frames(video_root: Path, out: Path) -> None:
    videos = []
    for condition in ORDER:
        candidates = sorted((video_root / condition / "rep0").glob("*wall_d62.mp4"))
        if not candidates:
            candidates = sorted((video_root / condition / "rep0").glob("*.mp4"))
        if candidates:
            videos.append((condition, candidates[0]))
    if not videos:
        return
    thumbs = []
    with tempfile.TemporaryDirectory() as directory:
        for i, (condition, video) in enumerate(videos):
            png = Path(directory) / f"{i}.png"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-sseof", "-0.1",
                "-i", str(video), "-frames:v", "1", "-y", str(png),
            ], check=True)
            thumbs.append((condition, Image.open(png).convert("RGB").resize((560, 420))))
        w, h = 1800, 620
        im = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(im)
        d.text((70, 35), "Representative final frames (d62, repetition 0)",
               font=font(40, True), fill="#0F172A")
        x0 = 60
        width = (w-120)/len(thumbs)
        for i, (condition, thumb) in enumerate(thumbs):
            tw = int(width-20)
            th = int(420*tw/560)
            thumb = thumb.resize((tw, th))
            x = int(x0+i*width+10)
            im.paste(thumb, (x, 130))
            _center(d, (x, 130+th+8, x+tw, 130+th+55), LABEL[condition], font(20, True))
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", default="results/safety_baseline_analysis")
    ap.add_argument("--video-root", default="results/safety_baseline_videos")
    args = ap.parse_args()
    payloads, rows = load_rows(args.inputs)
    summary = summarize(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "combined_summary.json").write_text(json.dumps({
        "source_files": args.inputs,
        "source_configs": [x["config"] for x in payloads],
        "summary": summary,
    }, indent=2))
    comparison_figure(summary, out / "fig_policy_comparison.png")
    outcomes_figure(summary, out / "fig_outcomes_and_interventions.png")
    monitor_figure(rows, out / "fig_monitor_diagnostics.png")
    final_frames(Path(args.video_root), out / "fig_final_frames.png")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
