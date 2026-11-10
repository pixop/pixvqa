#!/usr/bin/env python3
"""Create deterministic 10-bit limited-range PQ YUV420 video pairs for CVVDP A/B tests.

Synthetic patterns exercise luma, chroma, motion and spatial detail. Not an HDR
perceptual-quality benchmark: use strictly to compare two software environments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def make_pair(out: Path, frames: int, width: int, height: int, shift: int) -> dict:
    if width % 2 or height % 2:
        raise ValueError("dimensions must be even")
    out.mkdir(parents=True, exist_ok=True)
    ref = out / "pq_reference_420p10.yuv"
    dist = out / "pq_distorted_420p10.yuv"
    yy, xx = np.mgrid[:height, :width]
    uv_y, uv_x = np.mgrid[:height // 2, :width // 2]
    with ref.open("wb") as a, dist.open("wb") as b:
        for t in range(frames):
            phase = t / max(frames, 1)
            # Sweep legal limited-range PQ code values, plus high-frequency texture.
            grad = 150 + 660 * xx / width + 75 * np.sin(2 * np.pi * (yy / 31 + phase))
            checker = 45 * ((xx // 5 + yy // 5 + t // 2) % 2 - 0.5)
            y = np.clip(np.rint(grad + checker), 64, 940).astype('<u2')
            u = np.clip(np.rint(512 + 110 * np.sin(uv_x / 17 + phase * 2)), 64, 960).astype('<u2')
            v = np.clip(np.rint(512 + 120 * np.cos(uv_y / 13 - phase * 2)), 64, 960).astype('<u2')
            # Global luma shift + quantization bands, with modest chroma drift.
            yd = np.clip(np.rint((y.astype(np.int32) + shift) / 16) * 16, 64, 940).astype('<u2')
            ud = np.clip(u.astype(np.int32) + 24, 64, 960).astype('<u2')
            vd = np.clip(v.astype(np.int32) - 24, 64, 960).astype('<u2')
            for p in (y, u, v):
                a.write(p.tobytes())
            for p in (yd, ud, vd):
                b.write(p.tobytes())
    meta = {"width": width, "height": height, "fps": 60, "frames": frames,
            "format": "yuv420p10le", "transfer": "pq", "matrix": "bt2020nc",
            "range": "limited", "luma_shift_codes": shift,
            "purpose": "cross-environment numerical regression, not perceptual calibration"}
    for key, path in (("reference", ref), ("distorted", dist)):
        meta[key] = str(path)
        meta[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--shift", type=int, default=64)
    args = p.parse_args()
    print(json.dumps(make_pair(args.out, args.frames, args.width, args.height, args.shift), indent=2))
