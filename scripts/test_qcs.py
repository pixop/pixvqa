#!/usr/bin/env python3
"""QCS against the exact empirical-survival integral.

The published Python reference integrates a linearly interpolated survival with
a 1001-point trapezoidal rule. That is a different number. This test locks the
exact closed form and checks that the trapezoidal result is not silently used.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "cpu" / "pixvqa"
CUDA = ROOT / "build" / "cuda" / "pixvqa"
QCS = re.compile(r"\[ ([A-Z0-9-]+) value:(unavailable|[0-9.]+) range:\[([^,\]]+),([^,\]]+)\] \]")
SAMPLE = re.compile(r"qcs_sample (\S+) ([0-9.eE+-]+|inf|nan|-inf)")
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def exact(scores, a, b) -> float | None:
    """Closed form. +inf clamps to 1, -inf clamps to 0, NaN is omitted.

    This is independent of QcsAccum. It does not drop infinities.
    """
    q = np.asarray(scores, dtype=np.float64).reshape(-1)
    q = q[~np.isnan(q)]
    if q.size == 0:
        return None
    z = np.clip((q - a) / (b - a), 0.0, 1.0)
    return 100.0 * float(z.mean())


def trapz_qcs(scores, a, b, grid=1001) -> float:
    """Published qcs.py method. NumPy 2.4 names the rule trapezoid, not trapz."""
    q = np.asarray(scores, dtype=np.float64)
    q = q[np.isfinite(q)]
    x, counts = np.unique(q, return_counts=True)
    ge = np.cumsum(counts[::-1])[::-1] / q.size
    xi = np.linspace(a, b, grid)
    si = np.interp(xi, x, ge, left=1.0, right=0.0)
    integrate = getattr(np, "trapezoid", np.trapz)
    area = float(integrate(si, xi))
    return 100.0 * area / (b - a)


def psnr_db(mse_y, mse_u, mse_v, chroma, peak) -> float:
    if chroma == "420":
        mse = (4 * mse_y + mse_u + mse_v) / 6
    elif chroma == "422":
        mse = (2 * mse_y + mse_u + mse_v) / 4
    else:
        mse = (mse_y + mse_u + mse_v) / 3
    if mse <= 0:
        return math.inf
    return 10.0 * math.log10((peak * peak) / mse)


def write_const_frames(path: Path, frames, size, bits, base=400):
    h = w = size
    raw = bytearray()
    for dy, du, dv in frames:
        for delta in (dy, du, dv):
            plane = np.full((h, w), base + delta, dtype=np.int64)
            if bits == 8:
                raw += plane.astype(np.uint8).tobytes()
            else:
                raw += plane.astype(np.uint16).tobytes()
    path.write_bytes(raw)


def run(extra, ref, tst, size, fmt, bin_path=BIN, env=None):
    cmd = [str(bin_path), *extra, "-r", str(ref), "-s", f"{size}x{size}", "-f", fmt,
           "-t", str(tst), "-s", f"{size}x{size}", "-f", fmt]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)


def qcs_map(text: str) -> dict:
    out = {}
    for name, value, lo, hi in QCS.findall(text):
        out[name] = (None if value == "unavailable" else float(value), lo, hi)
    return out


def check_close(name, got, exp, tol=5.1e-7) -> None:
    if got is None or abs(got - exp) > tol:
        fail(f"{name}: got {got} expected {exp}")
    else:
        print(f"OK   {name} {got:.6f}")


def main() -> int:
    if not BIN.exists():
        print("missing", BIN, file=sys.stderr)
        return 2
    peak = 1023.0
    size = 16
    # Y-only offsets. U and V match, so their MSE is 0. 4:4:4 avg MSE is mse_y/3.
    offsets = [0, 4, 20, 80, 4]
    ref_frames = [(0, 0, 0)] * len(offsets)
    tst_frames = [(d, 0, 0) for d in offsets]
    scores = [psnr_db(d * d, 0, 0, "444", peak) for d in offsets]
    # The first frame is an exact match, so its PSNR is +inf. Counting that
    # frame as 100% is the closed form. The pre-fix result dropped it and
    # printed 61.185157. The corrected value on this fixture is 68.948125.
    with tempfile.TemporaryDirectory(prefix="qcs_") as tmp:
        td = Path(tmp)
        ref, tst = td / "r.yuv", td / "t.yuv"
        write_const_frames(ref, ref_frames, size, 10)
        write_const_frames(tst, tst_frames, size, 10)
        proc = run(["--metrics", "psnr", "--qcs"], ref, tst, size, "444p:10")
        if proc.returncode:
            fail(proc.stderr[-300:])
            return 1
        got = qcs_map(proc.stdout)
        # Closed form for this fixture, including the leading +inf frame.
        # Pre-fix finite-only result: 61.185157. Do not accept that value.
        locked = 68.9481253081602
        if abs(exact(scores, 30, 50) - locked) > 1e-9:
            fail(f"independent oracle drifted from {locked}: {exact(scores, 30, 50)}")
        if got["PSNR-QCS"][0] is not None and abs(got["PSNR-QCS"][0] - 61.185157) < 1e-4:
            fail("psnr_default still excludes the +inf frame (pre-fix 61.185157)")
        check_close("psnr_default", got["PSNR-QCS"][0], locked)
        if got["PSNR-QCS"][1:] != ("30", "50"):
            fail(f"default range {got['PSNR-QCS']}")
        # Reorder the two distorted frames by rewriting the test in swapped order.
        swapped = [tst_frames[2], tst_frames[1], tst_frames[3], tst_frames[4], tst_frames[0]]
        write_const_frames(tst, swapped, size, 10)
        proc2 = run(["--metrics", "psnr", "--qcs"], ref, tst, size, "444p:10")
        got2 = qcs_map(proc2.stdout)
        if got2["PSNR-QCS"][0] != got["PSNR-QCS"][0]:
            fail(f"reorder changed QCS {got['PSNR-QCS'][0]} vs {got2['PSNR-QCS'][0]}")
        else:
            print("OK   reorder")
        write_const_frames(tst, tst_frames, size, 10)
        custom = run(["--metrics", "psnr", "--qcs", "--qcs-range", "psnr=40:55"], ref, tst, size, "444p:10")
        cg = qcs_map(custom.stdout)
        check_close("psnr_custom", cg["PSNR-QCS"][0], exact(scores, 40, 55))
        if exact(scores, 40, 55) == exact(scores, 30, 50):
            fail("custom and default ranges are mathematically identical")
        elif cg["PSNR-QCS"][0] == got["PSNR-QCS"][0]:
            fail("custom range did not change QCS")
        else:
            print("OK   range_changes_qcs")
        if cg["PSNR-QCS"][1:] != ("40", "55"):
            fail(f"custom range text {cg['PSNR-QCS']}")
        # Identical pictures: internal PSNR is +inf, printed 999.99, QCS 100%.
        write_const_frames(tst, ref_frames, size, 10)
        ident = run(["--metrics", "psnr", "--qcs"], ref, tst, size, "444p:10")
        ig = qcs_map(ident.stdout)
        if "999.99" not in ident.stdout:
            fail(f"identical PSNR text lost 999.99: {ident.stdout[-200:]}")
        else:
            check_close("identical_psnr_qcs", ig["PSNR-QCS"][0], 100.0)
        xident = run(["--fps", "30", "--metrics", "xpsnr", "--qcs"], ref, ref, size, "444p:10")
        xig = qcs_map(xident.stdout)
        if xident.returncode != 0 or "inf" not in xident.stdout:
            fail(f"identical XPSNR: rc={xident.returncode} {xident.stdout[-200:]} {xident.stderr[-200:]}")
        else:
            check_close("identical_xpsnr_qcs", xig["XPSNR-QCS"][0], 100.0)
        # Metric text without QCS is unchanged when the flag is absent.
        write_const_frames(tst, tst_frames, size, 10)
        plain = run(["--metrics", "psnr"], ref, tst, size, "444p:10")
        with_q = run(["--metrics", "psnr", "--qcs"], ref, tst, size, "444p:10")
        def strip(text):
            return "\n".join(ln for ln in text.splitlines()
                             if "QCS" not in ln and not ln.startswith("Total frames:"))
        if strip(plain.stdout) != strip(with_q.stdout):
            fail("enabling QCS changed PSNR text")
        else:
            print("OK   psnr_text_unchanged")
        # SSIM and MS-SSIM on identical frames sit at the top of [0.8, 1].
        same = run(["--metrics", "ssim,msssim", "--qcs"], ref, ref, size, "444p:10")
        sg = qcs_map(same.stdout)
        check_close("ssim_identical", sg["SSIM-QCS"][0], 100.0)
        check_close("msssim_identical", sg["MS-SSIM-QCS"][0], 100.0)
        if sg["SSIM-QCS"][1:] != ("0.8", "1.0") or sg["MS-SSIM-QCS"][1:] != ("0.8", "1.0"):
            fail(f"fractional ranges {sg}")
        # Trace path: XPSNR internal avg, not the printed min.
        env = os.environ.copy()
        env["PIXVQA_QCS_TRACE"] = "1"
        xp = run(["--fps", "30", "--metrics", "xpsnr", "--qcs"], ref, tst, size, "444p:10", env=env)
        samples = [float(v) for kind, v in SAMPLE.findall(xp.stderr) if kind == "xpsnr"]
        xg = qcs_map(xp.stdout)
        check_close("xpsnr_trace", xg["XPSNR-QCS"][0], exact(samples, 30, 50))
        if "min:" not in xp.stdout:
            fail("XPSNR min disappeared")
        # The trapezoidal reference must not be what we printed on a repeated trace.
        repeated = np.array([40, 40, 40, 45, 45, 50], dtype=np.float64)
        gap_30 = abs(exact(repeated, 30, 50) - trapz_qcs(repeated, 30, 50))
        gap_40 = abs(exact(repeated, 40, 50) - trapz_qcs(repeated, 40, 50))
        if abs(gap_30 - 10.4166666667) > 1e-6 or abs(gap_40 - 20.8333333333) > 1e-6:
            fail(f"trapz gaps {gap_30} {gap_40}")
        else:
            print(f"OK   trapz_gap [30,50]={gap_30:.4f} [40,50]={gap_40:.4f}")
        # Reject bad ranges.
        for args, needle in (
            (["--metrics", "psnr", "--qcs", "--qcs-range", "nope=1:2"], "unknown"),
            (["--metrics", "psnr", "--qcs", "--qcs-range", "psnr=5:1"], "lower < upper"),
            (["--metrics", "psnr", "--qcs", "--qcs-range", "psnr=nan:2"], "lower bound"),
            (["--metrics", "psnr", "--qcs", "--qcs-range", "vmaf=70:100"], "not supported"),
            (["--metrics", "pu21-psnr", "--qcs", "--hdr-transfer", "pq"], "not QCS metrics"),
            (["--metrics", "psnr", "--qcs-range", "psnr=30:40"], "requires --qcs"),
            (["--metrics", "psnr", "--qcs", "--qcs-range", "psnr=1-2"], "invalid"),
        ):
            bad = run(args, ref, tst, size, "444p:10")
            if bad.returncode != 2 or needle not in bad.stderr:
                fail(f"range reject {args} rc={bad.returncode} {bad.stderr.strip()}")
            else:
                print(f"OK   reject {needle}")
        if CUDA.exists():
            cpu = run(["--metrics", "psnr", "--qcs"], ref, tst, size, "444p:10")
            gpu = run(["--metric-device", "gpu", "--metrics", "psnr", "--qcs"],
                      ref, tst, size, "444p:10", bin_path=CUDA)
            if qcs_map(cpu.stdout)["PSNR-QCS"] != qcs_map(gpu.stdout)["PSNR-QCS"]:
                fail(f"cuda qcs {qcs_map(gpu.stdout)} cpu {qcs_map(cpu.stdout)}")
            else:
                print("OK   cuda_psnr_qcs")
            cpu_id = run(["--metrics", "psnr", "--qcs"], ref, ref, size, "444p:10")
            gpu_id = run(["--metric-device", "gpu", "--metrics", "psnr", "--qcs"],
                         ref, ref, size, "444p:10", bin_path=CUDA)
            if "999.99" not in cpu_id.stdout or "999.99" not in gpu_id.stdout:
                fail("identical PSNR lost 999.99 on cpu or cuda")
            elif qcs_map(cpu_id.stdout)["PSNR-QCS"][0] != 100.0 or qcs_map(gpu_id.stdout)["PSNR-QCS"][0] != 100.0:
                fail(f"identical cuda qcs cpu {qcs_map(cpu_id.stdout)} gpu {qcs_map(gpu_id.stdout)}")
            else:
                print("OK   cuda_identical_psnr_qcs")
    if FAILS:
        print(f"{FAILS} QCS checks failed", file=sys.stderr)
        return 1
    print("All QCS checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
