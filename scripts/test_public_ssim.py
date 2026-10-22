#!/usr/bin/env python3
"""Classical SSIM against an independent NumPy reference.

The reference repeats the scalar implementation: normalized 11-tap Gaussian
with sigma 1.5, clamp-to-edge, valid interior (width-10) x (height-10),
K1=0.01, K2=0.03, L=peak, mean of luminance*CS, not clamped to [0, 1].

This file is prepared for a later run. It is not part of the current
execution restriction.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "cpu" / "pixvqa"
# Printed SSIM uses six decimals. This is the rounding bound.
PRINT_ATOL = 5.1e-7
FAILS = 0

SSIM_RE = __import__("re").compile(
    r"\[ SSIM  y:(-?[0-9.]+) u:(-?[0-9.]+) v:(-?[0-9.]+) avg:(-?[0-9.]+) \]"
)
MS_RE = __import__("re").compile(
    r"\[ MS-SSIM  y:(-?[0-9.]+) u:(-?[0-9.]+) v:(-?[0-9.]+) avg:(-?[0-9.]+) \]"
)


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def gaussian_kernel() -> np.ndarray:
    x = np.arange(11, dtype=np.float64) - 5.0
    k = np.exp(-(x * x) / (2.0 * 1.5 * 1.5))
    return k / k.sum()


def blur_valid(src: np.ndarray) -> np.ndarray:
    """Separable clamp-to-edge blur, then the valid 11-tap interior."""
    k = gaussian_kernel()
    h, w = src.shape
    tmp = np.empty_like(src)
    radius = 5
    for y in range(h):
        row = src[y]
        for x in range(w):
            acc = 0.0
            for t in range(-radius, radius + 1):
                xx = min(max(x + t, 0), w - 1)
                acc += k[t + radius] * row[xx]
            tmp[y, x] = acc
    out = np.empty((h - 10, w - 10), dtype=np.float64)
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            acc = 0.0
            for t in range(-radius, radius + 1):
                yy = min(max(y + t, 0), h - 1)
                acc += k[t + radius] * tmp[yy, x]
            out[y - radius, x - radius] = acc
    return out


def ssim_plane(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    if a.shape != b.shape or a.shape[0] < 11 or a.shape[1] < 11:
        raise ValueError("plane too small or shape mismatch")
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    mu1 = blur_valid(x)
    mu2 = blur_valid(y)
    s1 = blur_valid(x * x)
    s2 = blur_valid(y * y)
    s12 = blur_valid(x * y)
    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2
    v1 = s1 - mu1 * mu1
    v2 = s2 - mu2 * mu2
    v12 = s12 - mu1 * mu2
    lum = (2.0 * mu1 * mu2 + c1) / (mu1 * mu1 + mu2 * mu2 + c1)
    cs = (2.0 * v12 + c2) / (v1 + v2 + c2)
    return float(np.mean(lum * cs))


def constant_closed_form(a: float, b: float, peak: float) -> float:
    c1 = (0.01 * peak) ** 2
    return (2.0 * a * b + c1) / (a * a + b * b + c1)


def weighted(y: float, u: float, v: float, chroma: str) -> float:
    if chroma == "420":
        return (4.0 * y + u + v) / 6.0
    if chroma == "422":
        return (2.0 * y + u + v) / 4.0
    return (y + u + v) / 3.0


def chroma_size(h: int, w: int, chroma: str) -> tuple[int, int]:
    if chroma == "420":
        return h // 2, w // 2
    if chroma == "422":
        return h, w // 2
    return h, w


def write_yuv(path: Path, planes: list[np.ndarray], bits: int) -> None:
    raw = bytearray()
    for plane in planes:
        flat = np.asarray(plane, dtype=np.uint16 if bits > 8 else np.uint8).reshape(-1)
        raw += flat.tobytes()
    path.write_bytes(raw)


def run_pix(ref: Path, test: Path, w: int, h: int, chroma: str, bits: int, flags: list[str]):
    fmt = {"420": "420p", "422": "422p", "444": "444p"}[chroma]
    cmd = [
        str(BIN), *flags, "-r", str(ref), "-s", f"{w}x{h}", "-f", f"{fmt}:{bits}",
        "-t", str(test), "-s", f"{w}x{h}", "-f", f"{fmt}:{bits}"]
    env = os.environ.copy()
    env["PIXVQA_SSIM_BACKEND"] = "scalar"
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)


def check_oracle_closed_forms() -> None:
    for bits, peak in ((8, 255.0), (10, 1023.0)):
        a = np.full((32, 32), 0, dtype=np.int64)
        b = np.full((32, 32), int(peak), dtype=np.int64)
        same = np.full((32, 32), 40, dtype=np.int64)
        got_same = ssim_plane(same, same, peak)
        got_ends = ssim_plane(a, b, peak)
        if abs(got_same - 1.0) > 1e-12:
            fail(f"oracle identical {bits}-bit {got_same}")
        closed = constant_closed_form(0.0, peak, peak)
        if abs(got_ends - closed) > 1e-12:
            fail(f"oracle constant {bits}-bit {got_ends} vs {closed}")
        else:
            print(f"OK   oracle constant {bits}-bit {closed:.8f}")
    rng = np.random.default_rng(1)
    y0 = rng.integers(0, 1024, size=(32, 32))
    y1 = np.clip(y0 + 30, 0, 1023)
    distorted = ssim_plane(y0, y1, 1023.0)
    if not (distorted < 1.0):
        fail(f"oracle distorted not below 1: {distorted}")
    else:
        print(f"OK   oracle distorted {distorted:.8f}")


def check_binary() -> None:
    if not BIN.exists():
        fail(f"missing {BIN}; build with bash build.sh before running this test")
        return
    cases = []
    for bits, peak in ((8, 255), (10, 1023)):
        for chroma in ("420", "422", "444"):
            h = w = 32
            ch, cw = chroma_size(h, w, chroma)
            y0 = np.full((h, w), 40, dtype=np.int64)
            u0 = np.full((ch, cw), peak // 2, dtype=np.int64)
            v0 = np.full((ch, cw), peak // 3, dtype=np.int64)
            y1 = y0.copy()
            u1 = u0.copy()
            v1 = v0.copy()
            cases.append(("identical", bits, chroma, y0, y1, u0, u1, v0, v1, peak))
            y1 = np.full((h, w), 0, dtype=np.int64)
            u1 = np.full((ch, cw), peak, dtype=np.int64)
            v1 = v0.copy()
            cases.append(("planes", bits, chroma, y0, y1, u0, u1, v0, v1, peak))
    rng = np.random.default_rng(7)
    y0 = rng.integers(0, 1024, size=(32, 32))
    y1 = np.clip(y0.astype(np.int64) + rng.integers(-25, 26, size=y0.shape), 0, 1023)
    u0 = rng.integers(0, 1024, size=(16, 16))
    u1 = np.clip(u0.astype(np.int64) + 15, 0, 1023)
    v0 = rng.integers(0, 1024, size=(16, 16))
    v1 = v0.copy()
    cases.append(("distorted", 10, "420", y0, y1, u0, u1, v0, v1, 1023))

    with tempfile.TemporaryDirectory(prefix="public_ssim_") as tmp:
        td = Path(tmp)
        for name, bits, chroma, y0, y1, u0, u1, v0, v1, peak in cases:
            ref, test = td / f"{name}_{bits}_{chroma}_r.yuv", td / f"{name}_{bits}_{chroma}_t.yuv"
            write_yuv(ref, [y0, u0, v0], bits)
            write_yuv(test, [y1, u1, v1], bits)
            ey = ssim_plane(y0, y1, float(peak))
            eu = ssim_plane(u0, u1, float(peak))
            ev = ssim_plane(v0, v1, float(peak))
            eavg = weighted(ey, eu, ev, chroma)
            proc = run_pix(ref, test, y0.shape[1], y0.shape[0], chroma, bits,
                           ["--metrics", "ssim"])
            if proc.returncode != 0:
                fail(f"{name} {bits} {chroma}: rc={proc.returncode} {proc.stderr[-300:]}")
                continue
            found = SSIM_RE.findall(proc.stdout)
            if not found:
                fail(f"{name} {bits} {chroma}: no SSIM line\n{proc.stdout[-200:]}")
                continue
            got = [float(x) for x in found[-1]]
            exp = [ey, eu, ev, eavg]
            if any(abs(a - b) > PRINT_ATOL for a, b in zip(got, exp)):
                fail(f"{name} {bits} {chroma}: {got} vs {exp}")
                continue
            if MS_RE.search(proc.stdout):
                fail(f"{name} {bits} {chroma}: --ssim also printed MS-SSIM")
                continue
            print(f"OK   {name} {bits}-bit {chroma} y={ey:.6f} u={eu:.6f} v={ev:.6f} avg={eavg:.6f}")

        # SSIM and MS-SSIM are independent. Identical MS-SSIM is exactly 1.
        y = np.full((64, 64), 400, dtype=np.int64)
        u = np.full((32, 32), 512, dtype=np.int64)
        ref, test = td / "ind_r.yuv", td / "ind_t.yuv"
        write_yuv(ref, [y, u, u], 10)
        write_yuv(test, [y, u, u], 10)
        both = run_pix(ref, test, 64, 64, "420", 10, ["--metrics", "ssim,msssim"])
        only = run_pix(ref, test, 64, 64, "420", 10, ["--metrics", "msssim"])
        if both.returncode or only.returncode:
            fail(f"independence rc {both.returncode} {only.returncode} {both.stderr[-200:]}")
        else:
            if MS_RE.findall(both.stdout) != MS_RE.findall(only.stdout):
                fail("enabling --ssim changed MS-SSIM lines")
            elif not SSIM_RE.search(both.stdout) or SSIM_RE.search(only.stdout):
                fail("--ssim and --msssim were not independent")
            else:
                got = [float(x) for x in MS_RE.findall(both.stdout)[-1]]
                if any(abs(v - 1.0) > PRINT_ATOL for v in got):
                    fail(f"identical MS-SSIM was not 1: {got}")
                else:
                    print("OK   ssim and msssim are independent; identical msssim is 1")

        # PSNR / XPSNR / VMAF compatibility on the public CLI. No VMAF scoring.
        psnr = run_pix(ref, test, 64, 64, "420", 10, ["--metrics", "psnr"])
        if psnr.returncode != 0 or "999.99" not in psnr.stdout:
            fail(f"identical PSNR: rc={psnr.returncode} {psnr.stdout[-200:]}")
        else:
            print("OK   identical PSNR prints 999.99")
        xpsnr = run_pix(ref, test, 64, 64, "420", 10, ["--metrics", "xpsnr", "--fps", "24"])
        if xpsnr.returncode != 0 or "XPSNR" not in xpsnr.stdout:
            fail(f"XPSNR rc={xpsnr.returncode} {xpsnr.stderr[-200:]}")
        else:
            print("OK   XPSNR still selected by --xpsnr --fps")
        vmaf = run_pix(ref, test, 64, 64, "420", 10, ["--metrics", "vmaf"])
        if vmaf.returncode == 0 or "requires --model" not in vmaf.stderr:
            fail(f"--vmaf without --model: rc={vmaf.returncode} {vmaf.stderr!r}")
        else:
            print("OK   --vmaf still requires --model")


def main() -> int:
    check_oracle_closed_forms()
    check_binary()
    if FAILS:
        print(f"public SSIM checks FAILED ({FAILS})", file=sys.stderr)
        return 1
    print("All public SSIM checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
