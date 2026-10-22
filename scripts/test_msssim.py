#!/usr/bin/env python3
"""Five-scale MS-SSIM against an independent NumPy oracle and pytorch-msssim 1.0.0.

The oracle does not call pixvqa. pytorch-msssim is the external library.
TensorFlow 2.16.1 uses the same product, ReLU, weights and even-size 2x2 pool;
odd sizes differ (symmetric pad versus zero pad) and are not claimed to match TF.
"""
from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "cpu" / "pixvqa"
CUDA_BIN = ROOT / "build" / "cuda" / "pixvqa"
# Measured on the synthetic corpus: fp64 components matched the CPU double path
# to 12 significant digits. fp32 and mixed reached 1.03e-4 on constant fields
# and stayed under 1.5e-5 on textured frames. 1.5e-4 is that observed ceiling.
CUDA_FP64_ATOL = 1e-11
CUDA_FP32_ATOL = 1.5e-4
WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)
WIN = 11
SCORE = re.compile(
    r"\[ MS-SSIM  y:([0-9.eE+-]+) u:([0-9.eE+-]+) v:([0-9.eE+-]+) avg:([0-9.eE+-]+) \]"
)
COMP = re.compile(
    r"msssim_comp ([yuv]) (\d+) (\d+) (\d+) ([0-9.eE+-]+) ([0-9.eE+-]+)"
)

# Float64 agreement with the NumPy oracle. pytorch-msssim float64 is the
# external check on five-scale planes.
NUMPY_ATOL = 1e-9
TORCH_ATOL = 1e-9
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def kernel() -> np.ndarray:
    x = np.arange(WIN, dtype=np.float64) - (WIN // 2)
    k = np.exp(-(x * x) / (2.0 * 1.5 * 1.5))
    return k / k.sum()


def kernel_pytorch() -> np.ndarray:
    """Exact float32 coefficients from pytorch-msssim 1.0.0, widened to float64."""
    from pytorch_msssim.ssim import _fspecial_gauss_1d
    return _fspecial_gauss_1d(WIN, 1.5).detach().to(dtype=torch.float64).view(-1).numpy()


def pool_out(n: int) -> int:
    pad = n & 1
    return (n + 2 * pad - 2) // 2 + 1


def nscales(h: int, w: int) -> int:
    m = 0
    for i in range(5):
        if h < WIN or w < WIN:
            break
        m += 1
        if i == 4:
            break
        h, w = pool_out(h), pool_out(w)
    return m


def avg_pool2(img: np.ndarray) -> np.ndarray:
    """Match F.avg_pool2d(kernel=2, stride=2, padding=size%2, count_include_pad=True)."""
    h, w = img.shape
    pad_h, pad_w = h & 1, w & 1
    if pad_h == 0 and pad_w == 0:
        return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    oh = (h + 2 * pad_h - 2) // 2 + 1
    ow = (w + 2 * pad_w - 2) // 2 + 1
    acc = np.zeros((oh, ow), dtype=np.float64)
    yy = (np.arange(oh, dtype=np.int64) * 2)[:, None]
    xx = (np.arange(ow, dtype=np.int64) * 2)[None, :]
    for dy in range(2):
        for dx in range(2):
            sy = yy + dy - pad_h
            sx = xx + dx - pad_w
            valid = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            acc += np.where(valid, img[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)], 0.0)
    return acc * 0.25


def sep_filter(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    tmp = np.zeros((img.shape[0], img.shape[1] - 10), dtype=np.float64)
    for t in range(WIN):
        tmp += k[t] * img[:, t:t + img.shape[1] - 10]
    out = np.zeros((img.shape[0] - 10, img.shape[1] - 10), dtype=np.float64)
    for t in range(WIN):
        out += k[t] * tmp[t:t + img.shape[0] - 10, :]
    return out


def oracle_plane(a: np.ndarray, b: np.ndarray, peak: float, k: np.ndarray | None = None
                 ) -> tuple[float, list[tuple]]:
    if k is None:
        k = kernel()
    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = nscales(*x.shape)
    if m < 1:
        raise RuntimeError("plane too small")
    detail = []
    terms = []
    for scale in range(m):
        mux = sep_filter(x, k)
        muy = sep_filter(y, k)
        xx = sep_filter(x * x, k)
        yy = sep_filter(y * y, k)
        xy = sep_filter(x * y, k)
        vx = xx - mux * mux
        vy = yy - muy * muy
        cov = xy - mux * muy
        cs_map = (2.0 * cov + c2) / (vx + vy + c2)
        lum = (2.0 * mux * muy + c1) / (mux * mux + muy * muy + c1)
        cs = float(cs_map.mean())
        ssim = float((lum * cs_map).mean())
        detail.append((x.shape[0], x.shape[1], cs, ssim))
        terms.append(max(0.0, ssim if scale + 1 == m else cs))
        if scale + 1 < m:
            x = avg_pool2(x)
            y = avg_pool2(y)
    score = 1.0
    for i, term in enumerate(terms):
        score *= float(term ** WEIGHTS[i])
    return score, detail


def torch_plane(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    if min(a.shape) <= (WIN - 1) * (2 ** 4):
        return float("nan")
    tx = torch.tensor(a, dtype=torch.float64).view(1, 1, *a.shape)
    ty = torch.tensor(b, dtype=torch.float64).view(1, 1, *b.shape)
    with torch.no_grad():
        val = ms_ssim(tx, ty, data_range=float(peak), size_average=True)
    return float(val)


def write_yuv(path: Path, planes: list[np.ndarray], bits: int) -> None:
    raw = bytearray()
    for p in planes:
        flat = np.asarray(p, dtype=np.int64).ravel()
        if bits == 8:
            raw += flat.astype(np.uint8).tobytes()
        else:
            raw += flat.astype(np.uint16).tobytes()
    path.write_bytes(raw)


def chroma_planes(y: np.ndarray, u: np.ndarray, v: np.ndarray, chroma: str):
    if chroma == "444":
        return [y, u, v]
    if chroma == "422":
        return [y, u[:, ::2], v[:, ::2]]
    return [y, u[::2, ::2], v[::2, ::2]]


def run_tool(ref: Path, tst: Path, w: int, h: int, chroma: str, bits: int, extra: list[str],
             threads: int = 1):
    fmt = f"{chroma}p:{bits}"
    env = os.environ.copy()
    env["PIXVQA_MSSSIM_DEBUG"] = "1"
    cmd = [str(BIN), "--threads", str(threads), *extra, "-r", str(ref), "-s", f"{w}x{h}", "-f", fmt,
           "-t", str(tst), "-s", f"{w}x{h}", "-f", fmt]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)


def weighted(y: float, u: float, v: float, chroma: str) -> float:
    if chroma == "420":
        return (4 * y + u + v) / 6
    if chroma == "422":
        return (2 * y + u + v) / 4
    return (y + u + v) / 3


def check_case(name: str, y0, y1, u0, u1, v0, v1, chroma: str, bits: int, also_pair: bool = False,
               subsample: bool = True) -> None:
    h, w = y0.shape
    peak = float((1 << bits) - 1)
    if subsample:
        planes0 = chroma_planes(y0, u0, v0, chroma)
        planes1 = chroma_planes(y1, u1, v1, chroma)
    else:
        planes0 = [y0, u0, v0]
        planes1 = [y1, u1, v1]
    expected = []
    details = []
    torch_vals = []
    torch_oracle = []
    for a, b in zip(planes0, planes1):
        score, detail = oracle_plane(a, b, peak)
        expected.append(score)
        details.append(detail)
        torch_vals.append(torch_plane(a, b, peak))
        torch_oracle.append(oracle_plane(a, b, peak, kernel_pytorch())[0])
    exp_avg = weighted(*expected, chroma)
    with tempfile.TemporaryDirectory(prefix="msssim_") as tmp:
        td = Path(tmp)
        ref, tst = td / "r.yuv", td / "t.yuv"
        write_yuv(ref, planes0, bits)
        write_yuv(tst, planes1, bits)
        proc = run_tool(ref, tst, w, h, chroma, bits, ["--metrics", "msssim"])
        with_ssim = None
        ssim_only = None
        if also_pair:
            with_ssim = run_tool(ref, tst, w, h, chroma, bits, ["--metrics", "ssim,msssim"])
            ssim_only = run_tool(ref, tst, w, h, chroma, bits, ["--metrics", "ssim"])
    if proc.returncode != 0:
        fail(f"{name}: rc={proc.returncode} {proc.stderr[-400:]}")
        return
    scores = SCORE.findall(proc.stdout)
    if len(scores) < 1:
        fail(f"{name}: no MS-SSIM line\n{proc.stdout[-300:]}")
        return
    printed = [float(x) for x in scores[0]]
    comps = COMP.findall(proc.stderr)
    by = {"y": [], "u": [], "v": []}
    for plane, scale, hh, ww, cs, ssim in comps:
        by[plane].append((int(scale), int(hh), int(ww), float(cs), float(ssim)))
    rebuilt = []
    for label, detail in zip("yuv", details):
        rows = by[label]
        if len(rows) != len(detail):
            fail(f"{name}: {label} scales {len(rows)} != {len(detail)}")
            return
        score = 1.0
        for row, exp in zip(rows, detail):
            if row[1] != exp[0] or row[2] != exp[1]:
                fail(f"{name}: {label} geometry {row} vs {exp}")
                return
            if abs(row[3] - exp[2]) > NUMPY_ATOL or abs(row[4] - exp[3]) > NUMPY_ATOL:
                fail(f"{name}: {label} scale components {row} vs {exp}")
                return
            term = max(0.0, row[4] if row[0] + 1 == len(rows) else row[3])
            score *= float(term ** WEIGHTS[row[0]])
        rebuilt.append(score)
    for i, label in enumerate("yuv"):
        if abs(rebuilt[i] - expected[i]) > NUMPY_ATOL:
            fail(f"{name}: {label} product err {abs(rebuilt[i] - expected[i]):.3e}")
            return
        if abs(printed[i] - rebuilt[i]) > 5.1e-7:
            fail(f"{name}: {label} print rounding {printed[i]} vs {rebuilt[i]}")
            return
        if np.isfinite(torch_vals[i]):
            # pytorch-msssim 1.0.0 uses a float32 kernel. The same formula with
            # that kernel must match the library; the double-kernel C++ result
            # is compared separately and is not forced onto the float32 kernel.
            if abs(torch_oracle[i] - torch_vals[i]) > TORCH_ATOL:
                fail(f"{name}: {label} pytorch algorithm err "
                     f"{abs(torch_oracle[i] - torch_vals[i]):.3e} "
                     f"oracle {torch_oracle[i]} torch {torch_vals[i]}")
                return
    rebuilt_avg = weighted(*rebuilt, chroma)
    if abs(rebuilt_avg - exp_avg) > NUMPY_ATOL or abs(printed[3] - rebuilt_avg) > 5.1e-7:
        fail(f"{name}: avg printed {printed[3]} rebuilt {rebuilt_avg} oracle {exp_avg}")
        return
    if also_pair:
        if with_ssim.returncode != 0:
            fail(f"{name}: --ssim --msssim rc={with_ssim.returncode}")
            return
        if ssim_only.returncode != 0 or SCORE.search(ssim_only.stdout) or "[ SSIM" not in ssim_only.stdout:
            fail(f"{name}: --ssim did not select only SSIM")
            return
        if "[ SSIM" in proc.stdout:
            fail(f"{name}: --msssim also printed SSIM")
            return
        alone = SCORE.findall(proc.stdout)
        together = SCORE.findall(with_ssim.stdout)
        if alone != together or not alone:
            fail(f"{name}: --ssim changed MS-SSIM lines\n{alone}\n{together}")
            return
    print(f"OK   {name} y={rebuilt[0]:.8f} avg={rebuilt_avg:.8f} scales={len(details[0])}")


def rng_plane(rng: np.random.Generator, h: int, w: int, bits: int) -> np.ndarray:
    return rng.integers(0, 1 << bits, size=(h, w), dtype=np.int64)


def check_threads(y0: np.ndarray, y1: np.ndarray, chroma: str, bits: int) -> None:
    h, w = y0.shape
    planes = chroma_planes(y0, y0, y0, chroma)
    planes1 = chroma_planes(y1, y0, y0, chroma)
    with tempfile.TemporaryDirectory(prefix="msssim_thr_") as tmp:
        td = Path(tmp)
        ref, tst = td / "r.yuv", td / "t.yuv"
        write_yuv(ref, planes, bits)
        write_yuv(tst, planes1, bits)
        a = run_tool(ref, tst, w, h, chroma, bits, ["--metrics", "msssim"], threads=1)
        b = run_tool(ref, tst, w, h, chroma, bits, ["--metrics", "msssim"], threads=4)
    if a.returncode or b.returncode or COMP.findall(a.stderr) != COMP.findall(b.stderr):
        fail("threads 1 vs 4 changed MS-SSIM components")
        return
    if SCORE.findall(a.stdout) != SCORE.findall(b.stdout):
        fail("threads 1 vs 4 changed MS-SSIM text")
        return
    print("OK   threads_1_vs_4")


def check_too_small() -> None:
    plane = np.zeros((10, 10), dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="msssim_small_") as tmp:
        td = Path(tmp)
        path = td / "a.yuv"
        write_yuv(path, [plane, plane, plane], 10)
        proc = run_tool(path, path, 10, 10, "444", 10, ["--metrics", "msssim"])
    if proc.returncode == 0 or "at least 11x11" not in proc.stderr:
        fail(f"10x10 should be rejected, rc={proc.returncode} {proc.stderr[-200:]}")
        return
    print("OK   reject_10x10")


def check_hevc_frame(ref: Path, tst: Path) -> None:
    """First 4K 10-bit 4:2:0 frame of a previously decoded HEVC pair."""
    y_size = 3840 * 2160
    c_size = 1920 * 1080
    frame = y_size + 2 * c_size
    raw_r = np.fromfile(ref, dtype=np.uint16, count=frame)
    raw_t = np.fromfile(tst, dtype=np.uint16, count=frame)
    if raw_r.size != frame or raw_t.size != frame:
        fail("hevc frame short")
        return
    def split(raw):
        y = raw[:y_size].reshape(2160, 3840)
        u = raw[y_size:y_size + c_size].reshape(1080, 1920)
        v = raw[y_size + c_size:].reshape(1080, 1920)
        return y, u, v
    y0, u0, v0 = split(raw_r)
    y1, u1, v1 = split(raw_t)
    # Compare luma only through the tool by using the real planes.
    check_case("hevc_4k_frame", y0, y1, u0, u1, v0, v1, "420", 10, subsample=False)


def check_cuda() -> None:
    if not CUDA_BIN.exists():
        print("SKIP cuda binary")
        return
    rng = np.random.default_rng(11)
    cases = []
    y = rng.integers(0, 1024, size=(176, 176))
    cases.append(("cuda_noise", y, np.clip(y + rng.integers(-60, 61, size=y.shape), 0, 1023), "444", 10))
    c0 = np.full((176, 176), 400)
    c1 = np.full((176, 176), 700)
    cases.append(("cuda_const", c0, c1, "444", 10))
    odd = rng.integers(0, 1024, size=(177, 181))
    cases.append(("cuda_odd", odd, np.clip(odd + 12, 0, 1023), "444", 10))
    uhd = rng.integers(0, 1024, size=(2160, 3840))
    cases.append(("cuda_4k", uhd, np.clip(uhd + rng.integers(-30, 31, size=uhd.shape), 0, 1023), "420", 10))
    for name, a, b, chroma, bits in cases:
        h, w = a.shape
        planes0 = chroma_planes(a, a, a, chroma)
        planes1 = chroma_planes(b, a, a, chroma)
        with tempfile.TemporaryDirectory(prefix="msssim_cu_") as tmp:
            td = Path(tmp)
            ref, tst = td / "r.yuv", td / "t.yuv"
            write_yuv(ref, planes0, bits)
            write_yuv(tst, planes1, bits)
            cpu = run_tool(ref, tst, w, h, chroma, bits, ["--metrics", "msssim"])
            if cpu.returncode:
                fail(f"{name}: cpu rc {cpu.returncode}")
                return
            cpu_c = {(p, int(s)): (float(cs), float(ss))
                     for p, s, hh, ww, cs, ss in COMP.findall(cpu.stderr)}
            for prec, atol in (("fp64", CUDA_FP64_ATOL), ("fp32", CUDA_FP32_ATOL), ("mixed", CUDA_FP32_ATOL)):
                gpu = run_cuda(ref, tst, w, h, chroma, bits, prec)
                if gpu.returncode:
                    fail(f"{name} {prec}: {gpu.stderr[-300:]}")
                    return
                gpu_c = {(p, int(s)): (float(cs), float(ss))
                         for p, s, hh, ww, cs, ss in COMP.findall(gpu.stderr)}
                if set(gpu_c) != set(cpu_c):
                    fail(f"{name} {prec}: scale set mismatch")
                    return
                err = max(max(abs(gpu_c[k][0] - cpu_c[k][0]), abs(gpu_c[k][1] - cpu_c[k][1])) for k in cpu_c)
                if err > atol:
                    fail(f"{name} {prec}: component err {err:.3e} > {atol:.1e}")
                    return
            print(f"OK   {name} cuda fp64/fp32/mixed")
    check_cuda_leaves_other_metrics()


def run_cuda(ref, tst, w, h, chroma, bits, prec, extra=None):
    fmt = f"{chroma}p:{bits}"
    env = os.environ.copy()
    env["PIXVQA_MSSSIM_DEBUG"] = "1"
    env["PIXVQA_CUDA_PREC"] = prec
    cmd = [str(CUDA_BIN), "--threads", "1", "--metric-device", "gpu",
           *(extra or ["--metrics", "msssim"]), "-r", str(ref), "-s", f"{w}x{h}", "-f", fmt,
           "-t", str(tst), "-s", f"{w}x{h}", "-f", fmt]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)


def check_cuda_leaves_other_metrics() -> None:
    rng = np.random.default_rng(3)
    w, h, n = 320, 240, 4
    def pack(seed_a, seed_b):
        ra, rb = np.random.default_rng(seed_a), np.random.default_rng(seed_b)
        raw_a, raw_b = bytearray(), bytearray()
        for _ in range(n):
            y = ra.integers(0, 1024, size=(h, w))
            y2 = np.clip(y // 2 + rb.integers(0, 1024, size=(h, w)) // 2, 0, 1023)
            u = ra.integers(0, 1024, size=(h // 2, w // 2))
            v = ra.integers(0, 1024, size=(h // 2, w // 2))
            u2 = rb.integers(0, 1024, size=(h // 2, w // 2))
            v2 = rb.integers(0, 1024, size=(h // 2, w // 2))
            for p in (y, u, v):
                raw_a += np.asarray(p, np.uint16).tobytes()
            for p in (y2, u2, v2):
                raw_b += np.asarray(p, np.uint16).tobytes()
        return bytes(raw_a), bytes(raw_b)
    a, b = pack(3, 4)
    with tempfile.TemporaryDirectory(prefix="msssim_side_") as tmp:
        td = Path(tmp)
        ref, tst = td / "r.yuv", td / "t.yuv"
        ref.write_bytes(a)
        tst.write_bytes(b)
        base = ["--fps", "60", "--metrics", "psnr,ssim,xpsnr", 
                "-r", str(ref), "-s", f"{w}x{h}", "-f", "420p:10", "-t", str(tst), "-s", f"{w}x{h}",
                "-f", "420p:10"]
        env = os.environ.copy()
        alone = subprocess.run([str(CUDA_BIN), "--metric-device", "gpu", *base],
                               capture_output=True, text=True, env=env, timeout=300)
        both = subprocess.run(
            [str(CUDA_BIN), "--metric-device", "gpu", *base, "--metrics", "msssim"],
            capture_output=True, text=True, env=env, timeout=300)
    if alone.returncode or both.returncode:
        fail(f"combined cuda rc {alone.returncode} {both.returncode} {both.stderr[-200:]}")
        return
    def metric_lines(text):
        out = []
        for ln in text.splitlines():
            if ln.startswith(" ") or ln.startswith("Execution"):
                continue
            ln = re.sub(r"\[ MS-SSIM[^\]]*\] ?", "", ln).rstrip()
            if ln and any(k in ln for k in ("PSNR", "SSIM", "XPSNR")):
                out.append(ln)
        return out
    if metric_lines(alone.stdout) != metric_lines(both.stdout):
        fail("enabling MS-SSIM changed CUDA PSNR/SSIM/XPSNR text")
        return
    print("OK   cuda_msssim_does_not_change_other_metrics")


def main() -> int:
    if not BIN.exists():
        print("missing", BIN, file=sys.stderr)
        return 2
    rng = np.random.default_rng(7)
    h = w = 176
    y0 = rng_plane(rng, h, w, 10)
    check_case("identical_5scale", y0, y0, y0, y0, y0, y0, "444", 10, also_pair=True)
    y1 = y0.copy()
    y1[20, 20] = (int(y1[20, 20]) + 400) % 1024
    u0 = rng_plane(rng, h, w, 10)
    v0 = rng_plane(rng, h, w, 10)
    check_case("pixel_444", y0, y1, u0, u0, v0, v0, "444", 10)
    noise = np.clip(y0 + rng.integers(-80, 81, size=y0.shape), 0, 1023)
    check_case("noise_420", y0, noise, u0, u0, v0, v0, "420", 10)
    flat_a = np.full((h, w), 400, dtype=np.int64)
    flat_b = np.full((h, w), 700, dtype=np.int64)
    check_case("constants", flat_a, flat_b, flat_a, flat_a, flat_a, flat_a, "444", 10)
    checker = np.indices((h, w)).sum(axis=0) % 2 * 1023
    inv = 1023 - checker
    check_case("negative_cov", checker, inv, checker, checker, checker, checker, "444", 10)
    small = rng_plane(rng, 64, 64, 10)
    check_case("reduced_64", small, small, small, np.clip(small + 30, 0, 1023), small, small, "444", 10)
    y8 = rng.integers(0, 256, size=(176, 176), dtype=np.int64)
    y8b = np.clip(y8 + rng.integers(-20, 21, size=y8.shape), 0, 255)
    check_case("bit8_420", y8, y8b, y8, y8, y8, y8, "420", 8)
    odd = rng_plane(rng, 177, 181, 10)
    check_case("odd_zero_pad", odd, np.clip(odd + 15, 0, 1023), odd, odd, odd, odd, "444", 10)
    hd_y = rng_plane(rng, 1080, 1920, 10)
    hd_n = np.clip(hd_y + rng.integers(-40, 41, size=hd_y.shape), 0, 1023)
    check_case("hd_420", hd_y, hd_n, hd_y, hd_y, hd_y, hd_y, "420", 10)
    check_case("chroma_only_422", y0, y0, u0, np.clip(u0 + 80, 0, 1023), v0, v0, "422", 10)
    grad = np.tile(np.linspace(0, 1023, w, dtype=np.float64), (h, 1)).astype(np.int64)
    shifted = np.roll(grad, 2, axis=1)
    check_case("gradient_shift", grad, shifted, grad, grad, grad, grad, "444", 10)
    near = flat_a.copy()
    near[8, 8] = 401
    check_case("near_flat", flat_a, near, flat_a, flat_a, flat_a, flat_a, "444", 10)
    tiny = rng_plane(rng, 11, 11, 10)
    check_case("one_scale_11", tiny, np.clip(tiny + 5, 0, 1023), tiny, tiny, tiny, tiny, "444", 10)
    check_threads(y0, y1, "444", 10)
    check_too_small()
    hevc = Path("/tmp/pixvqa_p4/job0_ref_48.yuv")
    hevc_t = Path("/tmp/pixvqa_p4/job0_48.yuv")
    if hevc.exists() and hevc_t.exists():
        check_hevc_frame(hevc, hevc_t)
    check_cuda()
    if FAILS:
        print(f"{FAILS} MS-SSIM checks failed", file=sys.stderr)
        return 1
    print("All MS-SSIM checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
