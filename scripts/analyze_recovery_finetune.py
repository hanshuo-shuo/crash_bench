#!/usr/bin/env python3
"""Validate recovery-finetune artifacts and create report-ready figures."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INK = "#0F172A"
MUTED = "#475569"
GRID = "#E2E8F0"
BASE = "#64748B"
FINE = "#2563EB"
RISK = "#B42318"
SAFE = "#059669"
OUTCOMES = {
    "crash": RISK,
    "safe_abort": FINE,
    "recovery_success": SAFE,
    "timeout": "#94A3B8",
}


def font(size: int, bold: bool = False):
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial.ttf"]
        if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def center(draw, box, text, text_font, fill=INK):
    bounds = draw.textbbox((0, 0), str(text), font=text_font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.text((x, y), str(text), font=text_font, fill=fill)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    denominator = 1 + z * z / n
    midpoint = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, midpoint - half), min(1.0, midpoint + half)]


def load_eval(path: Path, expected_condition: str) -> dict:
    payload = json.loads(path.read_text())
    for key in ("config", "summary", "episodes"):
        if key not in payload:
            raise ValueError(f"{path}: missing {key}")
    episodes = payload["episodes"]
    if not episodes or {row["condition"] for row in episodes} != {expected_condition}:
        raise ValueError(f"{path}: unexpected or empty condition set")
    summary = payload["summary"][expected_condition]
    outcomes = Counter(row["outcome"] for row in episodes)
    if summary["n"] != len(episodes) or summary["n_crash"] != outcomes["crash"]:
        raise ValueError(f"{path}: summary does not match episodes")
    return payload


def group(payload: dict, condition: str) -> dict:
    episodes = payload["episodes"]
    outcomes = Counter(row["outcome"] for row in episodes)
    crashes = outcomes["crash"]
    learned_candidates = [
        sum(bool(step.get("learned_stop_candidate")) for step in row["steps"])
        for row in episodes
    ]
    early_learned_abort = sum(
        row["outcome"] == "safe_abort"
        and row["steps_to_event"] < 20
        and learned_candidates[index] >= 10
        for index, row in enumerate(episodes)
    )
    return {
        "condition": condition,
        "n": len(episodes),
        "outcomes": dict(outcomes),
        "crash_rate": crashes / len(episodes),
        "crash_rate_wilson95": wilson(crashes, len(episodes)),
        "mean_peak_force_n": sum(row["peak_contact_force"] for row in episodes) / len(episodes),
        "episodes_with_learned_stop": sum(value > 0 for value in learned_candidates),
        "early_learned_abort": early_learned_abort,
    }


def comparison_figure(groups: dict, out: Path) -> None:
    width, height = 1800, 1040
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 45), "Recovery fine-tuning: safety gain and control cost",
              font=font(47, True), fill=INK)
    draw.text((80, 108), "Crash rate; held-out scenarios only; n=6 per bar",
              font=font(25), fill=MUTED)
    left, right, top, bottom = 130, 1690, 250, 820
    for tick in range(6):
        y = bottom - tick * (bottom - top) / 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((55, y - 14), f"{tick * 20}%", font=font(20), fill=MUTED)
    panels = [
        ("On-path wall held-out", groups["base_wall"], groups["fine_wall"]),
        ("Off-path control held-out", groups["base_control"], groups["fine_control"]),
    ]
    panel_width = (right - left) / len(panels)
    for panel_index, (label, base, fine) in enumerate(panels):
        panel_left = left + panel_index * panel_width
        for bar_index, (name, stats, color) in enumerate(
            [("Base OpenVLA", base, BASE), ("Recovery-finetuned", fine, FINE)]
        ):
            center_x = panel_left + panel_width * (0.32 + 0.36 * bar_index)
            rate = stats["crash_rate"]
            bar_top = bottom - rate * (bottom - top)
            draw.rounded_rectangle((center_x - 115, bar_top, center_x + 115, bottom),
                                   radius=14, fill=color)
            lo, hi = stats["crash_rate_wilson95"]
            y_lo = bottom - lo * (bottom - top)
            y_hi = bottom - hi * (bottom - top)
            draw.line((center_x, y_hi, center_x, y_lo), fill=INK, width=5)
            draw.line((center_x - 16, y_hi, center_x + 16, y_hi), fill=INK, width=5)
            draw.line((center_x - 16, y_lo, center_x + 16, y_lo), fill=INK, width=5)
            label_top = max(top + 8, bar_top - 55)
            draw.rounded_rectangle(
                (center_x - 73, label_top, center_x + 73, label_top + 44),
                radius=8, fill="white",
            )
            center(draw, (center_x - 72, label_top, center_x + 72, label_top + 44),
                   f"{rate:.1%}", font(27, True))
            center(draw, (center_x - 150, bottom + 15, center_x + 150, bottom + 55),
                   name, font(21, True))
        center(draw, (panel_left + 20, bottom + 78, panel_left + panel_width - 20, bottom + 125),
               label, font(24, True), fill=MUTED)
    draw.text((130, 970),
              "Wall: 6/6 -> 1/6 crashes. Control: 2/6 -> 3/6 crashes. Error bars: 95% Wilson intervals.",
              font=font(22), fill=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def training_figure(metrics: list[dict], out: Path) -> None:
    width, height = 1800, 930
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 42), "LoRA optimization trace", font=font(47, True), fill=INK)
    draw.text((80, 104), "100 optimizer steps; effective batch 16; rank 16",
              font=font(25), fill=MUTED)
    panels = [
        (100, 250, 840, 760, "Cross-entropy loss (log scale)"),
        (960, 250, 1700, 760, "Action-token accuracy"),
    ]
    for x0, y0, x1, y1, title in panels:
        draw.rectangle((x0, y0, x1, y1), outline=GRID, width=2)
        center(draw, (x0, 170, x1, 225), title, font(25, True))
        for tick in range(6):
            y = y1 - tick * (y1 - y0) / 5
            draw.line((x0, y, x1, y), fill=GRID, width=2)
    steps = [row["step"] for row in metrics]
    losses = [max(float(row["loss"]), 1e-6) for row in metrics]
    accuracies = [float(row["action_token_accuracy"]) for row in metrics]
    log_min, log_max = -6.0, 1.0
    loss_points, accuracy_points = [], []
    for step, loss, accuracy in zip(steps, losses, accuracies):
        x_left = panels[0][0] + (step - 1) / 99 * (panels[0][2] - panels[0][0])
        y_left = panels[0][3] - ((math.log10(loss) - log_min) / (log_max - log_min)) * (
            panels[0][3] - panels[0][1]
        )
        loss_points.append((x_left, y_left))
        x_right = panels[1][0] + (step - 1) / 99 * (panels[1][2] - panels[1][0])
        y_right = panels[1][3] - accuracy * (panels[1][3] - panels[1][1])
        accuracy_points.append((x_right, y_right))
    draw.line(loss_points, fill=RISK, width=5, joint="curve")
    draw.line(accuracy_points, fill=FINE, width=5, joint="curve")
    for tick, label in enumerate(["1e-6", "2.5e-5", "6.3e-4", "1.6e-2", "0.4", "10"]):
        y = panels[0][3] - tick * (panels[0][3] - panels[0][1]) / 5
        draw.text((35, y - 12), label, font=font(18), fill=MUTED)
    for tick in range(6):
        y = panels[1][3] - tick * (panels[1][3] - panels[1][1]) / 5
        draw.text((900, y - 12), f"{tick * 20}%", font=font(18), fill=MUTED)
    for x0, _, x1, y1, _ in panels:
        for step in (1, 25, 50, 75, 100):
            x = x0 + (step - 1) / 99 * (x1 - x0)
            center(draw, (x - 35, y1 + 10, x + 35, y1 + 45), step, font(18), fill=MUTED)
    draw.text((100, 850),
              f"Step 1: loss {losses[0]:.3f}, accuracy {accuracies[0]:.1%}    |    "
              f"Step 100: loss {losses[-1]:.2e}, accuracy {accuracies[-1]:.1%}",
              font=font(23, True), fill=MUTED)
    image.save(out)


def final_frame(video: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to extract rollout frames")
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-sseof", "-0.05",
        "-i", str(video), "-frames:v", "1", "-y", str(out),
    ], check=True)


def rollout_figure(video_root: Path, out: Path) -> None:
    cases = [
        ("Wall d85 | Base | CRASH", video_root / "videos_base_heldout/vanilla/rep0/"
         "env_collision__T5__libero_spatial_t0_wall_d85.mp4"),
        ("Wall d85 | Fine-tuned | SAFE ABORT", video_root /
         "videos_finetuned_heldout/recovery_finetuned/rep0/"
         "env_collision__T5__libero_spatial_t0_wall_d85.mp4"),
        ("Control 00 | Base | TIMEOUT", video_root /
         "videos_base_control_heldout/vanilla/rep0/"
         "ood_control__T5__libero_spatial_t0_v3_00_twin.mp4"),
        ("Control 00 | Fine-tuned | CRASH", video_root /
         "videos_finetuned_control_heldout/recovery_finetuned/rep0/"
         "ood_control__T5__libero_spatial_t0_v3_00_twin.mp4"),
    ]
    for _, path in cases:
        if not path.exists():
            raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as directory:
        frames = []
        for index, (label, video) in enumerate(cases):
            frame_path = Path(directory) / f"frame_{index}.png"
            final_frame(video, frame_path)
            frames.append((label, Image.open(frame_path).convert("RGB")))
        canvas = Image.new("RGB", (1800, 1240), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((70, 40), "Representative rollout final frames",
                  font=font(47, True), fill=INK)
        draw.text((70, 103), "Same held-out scenario within each row; repetition 0",
                  font=font(25), fill=MUTED)
        positions = [(70, 190), (925, 190), (70, 700), (925, 700)]
        for (label, frame), (x, y) in zip(frames, positions):
            scale = min(805 / frame.width, 435 / frame.height)
            frame = frame.resize(
                (round(frame.width * scale), round(frame.height * scale)),
                Image.Resampling.LANCZOS,
            )
            tile = Image.new("RGB", (805, 435), "#F8FAFC")
            tile.paste(frame, ((805 - frame.width) // 2, (435 - frame.height) // 2))
            canvas.paste(tile, (x, y))
            color = RISK if "CRASH" in label else (SAFE if "SAFE ABORT" in label else MUTED)
            center(draw, (x, y + 440, x + 805, y + 486), label, font(22, True), fill=color)
        canvas.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="results/oracle_recovery/eval_v1")
    parser.add_argument("--metrics", default="results/oracle_recovery/model_v1/train_metrics.jsonl")
    parser.add_argument("--dataset-metadata", default="results/oracle_recovery/dataset_v1/metadata.json")
    parser.add_argument("--training-summary", default="results/oracle_recovery/model_v1/training_summary.json")
    parser.add_argument("--out-dir", default="results/oracle_recovery/report_assets")
    args = parser.parse_args()

    root = Path(args.eval_dir)
    payloads = {
        "base_wall": load_eval(root / "base_heldout.json", "vanilla"),
        "fine_wall": load_eval(root / "finetuned_heldout.json", "recovery_finetuned"),
        "fine_train": load_eval(root / "finetuned_train.json", "recovery_finetuned"),
        "base_control": load_eval(root / "base_control_heldout.json", "vanilla"),
        "fine_control": load_eval(root / "finetuned_control_heldout.json", "recovery_finetuned"),
    }
    groups = {
        key: group(payload, "vanilla" if key.startswith("base") else "recovery_finetuned")
        for key, payload in payloads.items()
    }
    metrics = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    if len(metrics) != 100 or [row["step"] for row in metrics] != list(range(1, 101)):
        raise ValueError("training metrics must contain exactly steps 1..100")
    dataset = json.loads(Path(args.dataset_metadata).read_text())
    training = json.loads(Path(args.training_summary).read_text())
    if dataset["counts"]["train"] != {
        "samples": 150, "oracle_stop": 90, "reference_action": 60
    }:
        raise ValueError("unexpected training dataset counts")
    if training.get("heldout_used_for_training") is not False:
        raise ValueError("training summary does not prove held-out exclusion")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison_figure(groups, out / "fig_recovery_comparison.png")
    training_figure(metrics, out / "fig_training_trace.png")
    rollout_figure(root, out / "fig_rollout_final_frames.png")
    summary = {
        "schema_version": 1,
        "groups": groups,
        "training": {
            "start_loss": metrics[0]["loss"],
            "final_loss": metrics[-1]["loss"],
            "start_action_token_accuracy": metrics[0]["action_token_accuracy"],
            "final_action_token_accuracy": metrics[-1]["action_token_accuracy"],
            "code_commit": training["code_commit"],
            "base_checkpoint": training["base_checkpoint"],
            "hyperparameters": training["hyperparameters"],
        },
        "dataset_counts": dataset["counts"],
        "train_scenarios": dataset["train_scenarios"],
        "heldout_scenarios": dataset["heldout_scenarios"],
        "train_control_scenarios": dataset["train_control_scenarios"],
        "heldout_control_scenarios": dataset["heldout_control_scenarios"],
        "source_files": {key: str(root / {
            "base_wall": "base_heldout.json",
            "fine_wall": "finetuned_heldout.json",
            "fine_train": "finetuned_train.json",
            "base_control": "base_control_heldout.json",
            "fine_control": "finetuned_control_heldout.json",
        }[key]) for key in payloads},
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
