#!/usr/bin/env python3
"""Compare CUDA PSNR and SSIM against the AVX2 CPU backend."""
from __future__ import annotations

import os
import random
import re
import struct
import subprocess
import sys
from pathlib import Path

CPU = Path(__file__).resolve().parents[1] / "build/cpu/pixvqa"
CUDA = Path(__file__).resolve().parents[1] / "build/full/pixvqa"
SS = re.compile(
    r"(?<![A-Z-])SSIM\s+y:(-?[0-9.]+)\s+u:(-?[0-9.]+)\s+v:(-?[0-9.]+)\s+avg:(-?[0-9.]+)"
)
PS = re.compile(r"PSNR\s+y:([0-9.]+)\s+u:([0-9.]+)\s+v:([0-9.]+)\s+avg:([0-9.]+)")
FR = re.compile(r"Total frames:(\d+)")


def plane(rng, w, h, lo=0, hi=1023):
    return [rng.randint(lo, hi) for _ in range(w * h)]


def write_yuv(path: Path, y, u, v):
    path.write_bytes(struct.pack("<" + "H" * (len(y) + len(u) + len(v)), *y, *u, *v))


def clip_noise(rng, src, amp):
    out = []
    for s in src:
        out.append(max(0, min(1023, s + rng.randint(-amp, amp))))
    return out


def make_pair(w, h, kind, seed=1):
    rng = random.Random(seed)
    cw, ch = w // 2, h // 2
    if kind == "identical":
        y, u, v = plane(rng, w, h), plane(rng, cw, ch), plane(rng, cw, ch)
        return (y, u, v), (list(y), list(u), list(v))
    if kind == "flat":
        y = [400] * (w * h)
        u = [512] * (cw * ch)
        v = [512] * (cw * ch)
        y2 = list(y)
        y2[10 * w + 10] = 401
        return (y, u, v), (y2, list(u), list(v))
    if kind == "noise":
        y, u, v = plane(rng, w, h), plane(rng, cw, ch), plane(rng, cw, ch)
        return (y, u, v), (clip_noise(rng, y, 30), clip_noise(rng, u, 20), clip_noise(rng, v, 20))
    raise SystemExit(kind)


def run(bin_path, ref, test, w, h, extra, prec=None):
    cmd = [
        str(bin_path),
        "-v", "1",
        "--metrics", "psnr,ssim",
        "--threads", "8",
        "-r", str(ref), "-t", str(test),
        "-s", f"{w}x{h}", "-f", "420p:10",
        *extra]
    env = os.environ.copy()
    if prec:
        env["PIXVQA_CUDA_PREC"] = prec
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(f"{bin_path} rc={p.returncode} {' '.join(extra)}")
    # Metric stdout must not mention the kernel.
    if "kernel=" in p.stdout or "strip=" in p.stdout or "cuda-prec" in p.stdout:
        raise SystemExit("metric stdout leaked a backend tag")
    out = {}
    for name, rx in (("psnr", PS), ("ssim", SS)):
        ms = list(rx.finditer(p.stdout))
        if len(ms) < 2:
            raise SystemExit(f"missing {name} in stdout")
        # summary is the last match
        out[name] = tuple(float(x) for x in ms[-1].groups())
    if not FR.search(p.stdout):
        raise SystemExit("missing Total frames")
    return out, p.stdout


def main():
    info = subprocess.run([str(CUDA)], capture_output=True, text=True, env={**os.environ, "PIXVQA_CUDA_INFO": "1"})
    print(info.stderr.strip())
    if info.returncode != 0:
        raise SystemExit(f"cuda-info rc={info.returncode}")
    root = Path("/tmp/pixvqa_cuda_num")
    root.mkdir(exist_ok=True)
    cases = [
        (64, 64, "identical", 1),
        (64, 64, "noise", 2),
        (64, 64, "flat", 3),
        (384, 384, "noise", 4),
        (384, 384, "identical", 5),
        (512, 256, "noise", 6)]
    precs = ["fp64", "fp32", "mixed"]
    worst = {p: 0.0 for p in precs}
    for w, h, kind, seed in cases:
        ref_a, test_a = make_pair(w, h, kind, seed)
        rp, tp = root / f"r_{w}_{kind}.yuv", root / f"t_{w}_{kind}.yuv"
        write_yuv(rp, *ref_a)
        write_yuv(tp, *test_a)
        cpu, _ = run(CPU, rp, tp, w, h, [ ])
        for prec in precs:
            gpu, _ = run(CUDA, rp, tp, w, h, ["--metric-device", "gpu"], prec)
            for metric in ("psnr", "ssim"):
                delta = [abs(a - b) for a, b in zip(cpu[metric], gpu[metric])]
                # PSNR is printed at 2 decimals; compare the printed values.
                # SSIM is printed at 6 decimals.
                md = max(delta)
                if metric != "psnr":
                    worst[prec] = max(worst[prec], md)
                flag = ""
                limit = 0.0 if metric == "psnr" else (1e-4 if prec == "fp64" else 5e-4)
                if md > limit + 1e-12:
                    flag = "  **OVER**"
                print(
                    f"{w}x{h} {kind:10} {prec:5} {metric:7} "
                    f"cpu={cpu[metric]} gpu={gpu[metric]} max|d|={md:.3e}{flag}"
                )
    print("worst non-psnr", worst)


if __name__ == "__main__":
    main()
