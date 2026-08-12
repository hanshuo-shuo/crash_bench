#!/usr/bin/env python3
"""Build paper-facing Pilot B/C figures from stored accepted-pair frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = (250, 250, 248)
INK = (31, 41, 55)
MUTED = (88, 99, 115)
BLUE = (56, 115, 184)
GREEN = (44, 139, 92)
AMBER = (205, 137, 33)
RED = (190, 61, 58)
LIGHT_BLUE = (220, 233, 247)
LIGHT_GREEN = (218, 239, 228)
LIGHT_AMBER = (248, 235, 204)
LIGHT_RED = (248, 222, 220)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save(canvas: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def draw_funnel(out: Path) -> None:
    canvas = Image.new("RGB", (1500, 760), BG)
    d = ImageDraw.Draw(canvas)
    d.text((70, 45), "Pilot B: H=20 certification funnel", fill=INK, font=font(42, bold=True))
    d.text(
        (70, 105),
        "Complete 15-scene ledger; counts are candidates, not independent repeated trials",
        fill=MUTED,
        font=font(24),
    )
    stages = [
        ("Authored candidates", 15, BLUE, LIGHT_BLUE),
        ("Base catastrophes", 12, RED, LIGHT_RED),
        ("Exact H=20 replay", 9, AMBER, LIGHT_AMBER),
        ("Certified pairs", 3, GREEN, LIGHT_GREEN),
    ]
    max_width = 1120
    x = 250
    y = 180
    for label, value, color, light in stages:
        width = int(max_width * value / 15)
        d.rounded_rectangle((x, y, x + width, y + 92), radius=16, fill=light, outline=color, width=3)
        d.text((x + 24, y + 23), label, fill=INK, font=font(28, bold=True))
        d.text((x + width - 75, y + 21), str(value), fill=color, font=font(36, bold=True), anchor="ra")
        y += 125
    d.text((70, 650), "Yield: 3/15 overall; 3/12 conditional on a Base accident", fill=INK, font=font(29, bold=True))
    save(canvas, out / "fig_pilot_b_funnel.png")


def draw_success_matrix(out: Path) -> None:
    canvas = Image.new("RGB", (1500, 660), BG)
    d = ImageDraw.Draw(canvas)
    d.text((70, 45), "Pilot C: scoped Oracle upper bound", fill=INK, font=font(42, bold=True))
    d.text(
        (70, 105),
        "Exact anchor, treatment, oracle timing + oracle recovery; development-only cohort",
        fill=MUTED,
        font=font(24),
    )
    seeds = [101, 202, 303]
    pairs = [("heldout_0000", 232), ("heldout_0004", 300)]
    x0, y0, cw, ch = 430, 205, 270, 145
    for j, seed in enumerate(seeds):
        d.text((x0 + j * cw + cw // 2, y0 - 48), f"seed {seed}", fill=INK, font=font(25, bold=True), anchor="mm")
    for i, (pair, steps) in enumerate(pairs):
        y = y0 + i * ch
        d.text((70, y + 38), pair, fill=INK, font=font(27, bold=True))
        d.text((70, y + 79), f"{steps} steps", fill=MUTED, font=font(23))
        for j in range(3):
            x = x0 + j * cw
            d.rounded_rectangle((x, y, x + 220, y + 105), radius=15, fill=LIGHT_GREEN, outline=GREEN, width=3)
            d.text((x + 110, y + 39), "SAFE TASK SUCCESS", fill=GREEN, font=font(20, bold=True), anchor="mm")
            d.text((x + 110, y + 73), "exact restore", fill=INK, font=font(18), anchor="mm")
    d.text((70, 545), "6/6 safe task success   |   0/6 catastrophes   |   6/6 exact restores", fill=INK, font=font(29, bold=True))
    save(canvas, out / "fig_pilot_c_success_matrix.png")


def draw_progress(out: Path) -> None:
    canvas = Image.new("RGB", (1680, 720), BG)
    d = ImageDraw.Draw(canvas)
    d.text((70, 45), "Glass-recovery evidence trajectory", fill=INK, font=font(42, bold=True))
    stages = [
        ("E14", "Existence smoke", "3 accepted / 109 attempts", GREEN),
        ("Pilot A", "State repair", "3/3 exact replay + recapture", GREEN),
        ("Broad B", "Population frontier", "No common H qualified", RED),
        ("Scoped B", "Certification screen", "3 accepted / 15 candidates", GREEN),
        ("Pilot C", "Oracle upper bound", "6/6 safe task success", GREEN),
        ("D/E/F", "Learned recovery", "Not evaluated", MUTED),
    ]
    y = 330
    left, right = 120, 1560
    d.line((left, y, right, y), fill=(170, 177, 187), width=6)
    gap = (right - left) / (len(stages) - 1)
    for i, (name, title, result, color) in enumerate(stages):
        x = int(left + i * gap)
        d.ellipse((x - 22, y - 22, x + 22, y + 22), fill=BG, outline=color, width=8)
        d.text((x, y - 112), name, fill=color, font=font(27, bold=True), anchor="mm")
        d.text((x, y + 70), title, fill=INK, font=font(22, bold=True), anchor="mm")
        d.multiline_text((x, y + 126), result, fill=MUTED, font=font(19), anchor="ma", align="center", spacing=5)
    d.text((70, 635), "Current endpoint: certified recoverability + Oracle task-completion upper bound; learned closed loop remains open.", fill=INK, font=font(27, bold=True))
    save(canvas, out / "fig_evidence_trajectory.png")


def load_images(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["images"])


def sample_indices(length: int, count: int = 6) -> list[int]:
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def make_filmstrip(pair_dir: Path, pair_label: str, out: Path) -> None:
    rows = [
        ("Base suffix: catastrophe", "nominal_catastrophe.npz", RED),
        ("Oracle: safe task completion", "oracle_recovery.npz", GREEN),
        ("Off-path control: task success", "off_path_control.npz", BLUE),
    ]
    frame_w, frame_h, count = 224, 224, 6
    left, top, gap = 315, 160, 15
    width = left + count * frame_w + (count - 1) * gap + 55
    height = top + len(rows) * (frame_h + 100) + 30
    canvas = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(canvas)
    d.text((50, 35), f"Certified accident trajectory: {pair_label}", fill=INK, font=font(40, bold=True))
    d.text((50, 92), "Frames are sampled directly from stored 224x224 trajectory arrays", fill=MUTED, font=font(23))
    y = top
    for title, filename, color in rows:
        images = load_images(pair_dir / filename)
        indices = sample_indices(len(images), count)
        d.text((50, y + 78), title, fill=color, font=font(25, bold=True))
        d.text((50, y + 116), f"{len(images)} stored frames", fill=MUTED, font=font(21))
        for j, idx in enumerate(indices):
            frame = Image.fromarray(images[idx])
            x = left + j * (frame_w + gap)
            canvas.paste(frame, (x, y))
            d.rectangle((x, y, x + frame_w - 1, y + frame_h - 1), outline=color, width=3)
            d.text((x + frame_w // 2, y + frame_h + 18), f"t={idx}", fill=INK, font=font(18), anchor="ma")
        y += frame_h + 100
    save(canvas, out / f"fig_{pair_label}_filmstrip.png")


def make_oracle_gif(pair_dir: Path, pair_label: str, out: Path) -> None:
    images = load_images(pair_dir / "oracle_recovery.npz")
    indices = sample_indices(len(images), min(60, len(images)))
    frames = []
    for idx in indices:
        frame = Image.fromarray(images[idx]).resize((448, 448), Image.Resampling.NEAREST)
        d = ImageDraw.Draw(frame)
        d.rectangle((0, 0, 448, 46), fill=(0, 0, 0))
        d.text((14, 11), f"{pair_label}  oracle recovery  frame {idx}/{len(images)-1}", fill=(255, 255, 255), font=font(19, bold=True))
        frames.append(frame)
    path = out / f"{pair_label}_oracle_recovery.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-0000", type=Path, required=True)
    parser.add_argument("--pair-0004", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    draw_funnel(args.out)
    draw_success_matrix(args.out)
    draw_progress(args.out)
    for pair_dir, label in ((args.pair_0000, "heldout_0000"), (args.pair_0004, "heldout_0004")):
        make_filmstrip(pair_dir, label, args.out)
        make_oracle_gif(pair_dir, label, args.out)


if __name__ == "__main__":
    main()
