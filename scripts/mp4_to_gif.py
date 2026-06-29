#!/usr/bin/env python
"""Convert an mp4 (or several) to a report-friendly looping gif.

Usage:
    python scripts/mp4_to_gif.py IN.mp4 [OUT.gif] [--stride 2] [--fps 12] [--scale 256]

Run with the openvla env which bundles imageio + ffmpeg:
    envs/openvla/bin/python scripts/mp4_to_gif.py ...
"""
import argparse
import os

import imageio
import numpy as np


def convert(src, dst, stride=2, fps=12, scale=None):
    reader = imageio.get_reader(src, "ffmpeg")
    frames = []
    for i, frame in enumerate(reader):
        if i % stride:
            continue
        if scale and frame.shape[0] != scale:
            # nearest-neighbour box resize, no extra deps
            h, w = frame.shape[:2]
            ys = (np.linspace(0, h - 1, scale)).astype(int)
            xs = (np.linspace(0, w - 1, scale)).astype(int)
            frame = frame[ys][:, xs]
        frames.append(frame)
    reader.close()
    imageio.mimsave(dst, frames, format="GIF", fps=fps, loop=0)
    print(f"{os.path.basename(src)} -> {dst}  ({len(frames)} frames, {os.path.getsize(dst)//1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--scale", type=int, default=None, help="output square size in px")
    a = ap.parse_args()
    dst = a.dst or os.path.splitext(a.src)[0] + ".gif"
    convert(a.src, dst, a.stride, a.fps, a.scale)
