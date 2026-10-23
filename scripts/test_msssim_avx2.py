#!/usr/bin/env python3
"""AVX2 MS-SSIM against the scalar reference and the independent double oracle."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "cpu" / "pixvqa"
sys.path.insert(0, str(ROOT / "scripts"))
import test_msssim as oracle  # noqa: E402

FAILS = 0
ATOL = 1e-9


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def run(backend: str, ref: Path, tst: Path, w: int, h: int, chroma: str, bits: int,
        threads: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PIXVQA_MSSSIM_DEBUG"] = "1"
    env["PIXVQA_MSSSIM_BACKEND"] = backend
    fmt = f"{chroma}p:{bits}"
    cmd = [str(BIN), "--threads", str(threads), "--metrics", "msssim", "-v", "0",
           "-r", str(ref), "-s", f"{w}x{h}", "-f", fmt,
           "-t", str(tst), "-s", f"{w}x{h}", "-f", fmt]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)


def comps(stderr: str) -> list[tuple]:
    rows = []
    for line in stderr.splitlines():
        if not line.startswith("msssim_comp "):
            continue
        p = line.split()
        rows.append((p[1], int(p[2]), int(p[3]), int(p[4]), float(p[5]), float(p[6])))
    return rows


def check_pair(name: str, y0, y1, u0, u1, v0, v1, chroma: str, bits: int, threads: int) -> None:
    h, w = y0.shape
    peak = float((1 << bits) - 1)
    p0 = oracle.chroma_planes(y0, u0, v0, chroma)
    p1 = oracle.chroma_planes(y1, u1, v1, chroma)
    expected = []
    for a, b in zip(p0, p1):
        score, detail = oracle.oracle_plane(a, b, peak, oracle.kernel())
        expected.append((score, detail))
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.yuv"
        tst = Path(tmp) / "tst.yuv"
        oracle.write_yuv(ref, p0, bits)
        oracle.write_yuv(tst, p1, bits)
        scalar = run("scalar", ref, tst, w, h, chroma, bits, threads)
        avx = run("avx2", ref, tst, w, h, chroma, bits, threads)
    if scalar.returncode != 0 or avx.returncode != 0:
        fail(f"{name} rc scalar {scalar.returncode} avx {avx.returncode} {avx.stderr[-400:]}")
        return
    sc = comps(scalar.stderr)
    ac = comps(avx.stderr)
    if len(sc) != len(ac) or not sc:
        fail(f"{name} component count {len(sc)} {len(ac)}")
        return
    worst = 0.0
    planes = {"y": 0, "u": 1, "v": 2}
    for srow, arow in zip(sc, ac):
        if srow[:4] != arow[:4]:
            fail(f"{name} shape {srow[:4]} {arow[:4]}")
            return
        worst = max(worst, abs(srow[4] - arow[4]), abs(srow[5] - arow[5]))
        exp = expected[planes[srow[0]]][1][srow[1]]
        if abs(arow[4] - exp[2]) > ATOL or abs(arow[5] - exp[3]) > ATOL:
            fail(f"{name} oracle {srow[0]} scale {srow[1]} cs {arow[4]} vs {exp[2]}")
            return
    if worst > ATOL:
        fail(f"{name} scalar-vs-avx2 {worst:.3e}")
        return
    print(f"OK   {name} threads {threads} worst {worst:.3e}")


def main() -> int:
    if not BIN.is_file():
        print("missing binary", file=sys.stderr)
        return 1
    rng = np.random.default_rng(11)
    bits = 10
    peak = 1023
    cases = []
    h = w = 176
    y0 = np.full((h, w), 400, dtype=np.int32)
    y1 = np.full((h, w), 700, dtype=np.int32)
    u0 = np.full((h, w), 512, dtype=np.int32)
    cases.append(("const", y0, y1, u0, u0.copy(), u0.copy(), u0.copy(), "420", bits))
    y0n = rng.integers(0, peak + 1, size=(h, w))
    y1n = np.clip(y0n + rng.integers(-30, 31, size=(h, w)), 0, peak)
    u0n = rng.integers(0, peak + 1, size=(h, w))
    u1n = np.clip(u0n + rng.integers(-20, 21, size=u0n.shape), 0, peak)
    v0n = rng.integers(0, peak + 1, size=(h, w))
    v1n = np.clip(v0n + rng.integers(-20, 21, size=v0n.shape), 0, peak)
    cases.append(("noise420", y0n, y1n, u0n, u1n, v0n, v1n, "420", bits))
    ho, wo = 177, 181
    cases.append(("odd444", rng.integers(0, peak + 1, (ho, wo)),
                  rng.integers(0, peak + 1, (ho, wo)),
                  rng.integers(0, peak + 1, (ho, wo)),
                  rng.integers(0, peak + 1, (ho, wo)),
                  rng.integers(0, peak + 1, (ho, wo)),
                  rng.integers(0, peak + 1, (ho, wo)), "444", bits))
    y8 = rng.integers(0, 256, size=(64, 64))
    cases.append(("bit8", y8, np.clip(y8 + 3, 0, 255),
                  rng.integers(0, 256, (64, 64)), rng.integers(0, 256, (64, 64)),
                  rng.integers(0, 256, (64, 64)), rng.integers(0, 256, (64, 64)), "420", 8))
    g = np.tile(np.arange(176, dtype=np.int32), (176, 1))
    cases.append(("gradient", g, np.clip(g + 8, 0, 1023),
                  np.full((176, 176), 400), np.full((176, 176), 420),
                  np.full((176, 176), 400), np.full((176, 176), 400), "420", bits))
    yf = np.full((176, 176), 500, dtype=np.int32)
    yf2 = yf.copy()
    yf2[40, 40] = 508
    cases.append(("pixel", yf, yf2, u0, u0, u0, u0, "420", bits))
    inv = np.indices((176, 176)).sum(axis=0) % 2
    a = np.where(inv, 100, 900)
    b = np.where(inv, 900, 100)
    cases.append(("neg", a, b, u0, u0, u0, u0, "420", bits))
    cases.append(("ident", y0, y0, u0, u0, u0, u0, "420", bits))
    y422 = rng.integers(0, peak + 1, (96, 128))
    cases.append(("c422", y422, np.clip(y422 + 5, 0, peak),
                  rng.integers(0, peak + 1, (96, 128)), rng.integers(0, peak + 1, (96, 128)),
                  rng.integers(0, peak + 1, (96, 128)), rng.integers(0, peak + 1, (96, 128)),
                  "422", bits))
    for name, *rest in cases:
        check_pair(name, *rest, threads=1)
        if name in ("noise420", "odd444"):
            check_pair(name + "_t8", *rest, threads=8)
    ref4 = Path("/tmp/pixvqa_p4/job0_ref_48.yuv")
    tst4 = Path("/tmp/pixvqa_p4/job0_48.yuv")
    if ref4.is_file():
        cmd_base = ["--metrics", "msssim", "-n", "1", "-v", "0",
                    "-r", str(ref4), "-s", "3840x2160", "-f", "420p:10",
                    "-t", str(tst4), "-s", "3840x2160", "-f", "420p:10"]
        rows = {}
        for th in (1, 4, 8, 16, 20):
            env = os.environ.copy()
            env["PIXVQA_MSSSIM_DEBUG"] = "1"
            env["PIXVQA_MSSSIM_BACKEND"] = "avx2"
            done = subprocess.run([str(BIN), "--threads", str(th), *cmd_base],
                                  capture_output=True, text=True, env=env, timeout=180)
            if done.returncode != 0:
                fail(f"4k threads {th} rc {done.returncode}")
                continue
            rows[th] = comps(done.stderr)
        if 1 in rows and 8 in rows:
            worst = 0.0
            for a, b in zip(rows[1], rows[8]):
                worst = max(worst, abs(a[4] - b[4]), abs(a[5] - b[5]))
            if worst > ATOL:
                fail(f"4k thread reduction {worst:.3e}")
            else:
                print(f"OK   4k threads 1-vs-8 worst {worst:.3e}")
            for th, row in rows.items():
                if th == 1:
                    continue
                tw = 0.0
                for a, b in zip(rows[1], row):
                    tw = max(tw, abs(a[4] - b[4]), abs(a[5] - b[5]))
                if tw > ATOL:
                    fail(f"4k threads {th} {tw:.3e}")
                else:
                    print(f"OK   4k threads {th} worst {tw:.3e}")
        env = os.environ.copy()
        env["PIXVQA_MSSSIM_DEBUG"] = "1"
        env["PIXVQA_MSSSIM_BACKEND"] = "scalar"  # ignored; production stays on AVX2
        scalar = subprocess.run([str(BIN), "--threads", "4", *cmd_base],
                                capture_output=True, text=True, env=env, timeout=180)
        if scalar.returncode != 0 or 4 not in rows:
            fail("4k scalar")
        else:
            worst = 0.0
            for a, b in zip(comps(scalar.stderr), rows[4]):
                worst = max(worst, abs(a[4] - b[4]), abs(a[5] - b[5]))
            if worst > ATOL:
                fail(f"4k backend env changed the score {worst:.3e}")
            else:
                print(f"OK   4k PIXVQA_MSSSIM_BACKEND ignored worst {worst:.3e}")
    if FAILS:
        print(f"{FAILS} failed", file=sys.stderr)
        return 1
    print("All AVX2 MS-SSIM checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
