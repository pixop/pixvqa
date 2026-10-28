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
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
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


def run(extra, ref, tst, size, fmt, bin_path=BIN, env=None, timeout=120):
    geom = size if isinstance(size, str) else f"{size}x{size}"
    cmd = [str(bin_path), *extra, "-r", str(ref), "-s", geom, "-f", fmt,
           "-t", str(tst), "-s", geom, "-f", fmt]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)


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


def trace_of(stderr: str, kind: str) -> list[float]:
    return [float(v) for name, v in SAMPLE.findall(stderr) if name == kind]


def write_420p10(path: Path, w: int, h: int, y_offsets: list[int]) -> None:
    """Limited-range 10-bit 4:2:0. y_offsets shifts each frame's luma."""
    cw, ch = w // 2, h // 2
    yy, xx = np.mgrid[0:h, 0:w]
    base = 64 + (xx * 3 + yy) % 700
    chroma = np.full((ch, cw), 512, np.uint16).tobytes()
    chunks = []
    for off in y_offsets:
        y = np.clip(base + off, 64, 940).astype(np.uint16)
        chunks.append(y.tobytes())
        chunks.append(chroma)
        chunks.append(chroma)
    path.write_bytes(b"".join(chunks))


HDR = [
    "--hdr-transfer", "pq",
    "--hdr-primaries", "bt2020",
    "--hdr-matrix", "bt2020nc",
    "--hdr-range", "limited",
]
VMAF_LINE = re.compile(r"\[ VMAF:([0-9.eE+-]+) \]")
PU_LINE = re.compile(r"\[ (PU21-PSNR|PU21-SSIM|PU21-MS-SSIM):([0-9.eE+-]+) \]")


def qcs_from_trace(proc, qcs_name: str, kind: str, lo: float, hi: float, label: str) -> list[float]:
    if proc.returncode != 0:
        fail(f"{label}: rc={proc.returncode} {proc.stderr[-400:]}")
        return []
    samples = trace_of(proc.stderr, kind)
    got = qcs_map(proc.stdout)
    if qcs_name not in got:
        fail(f"{label}: missing {qcs_name}\n{proc.stdout[-400:]}")
        return []
    if not samples:
        fail(f"{label}: no qcs_sample {kind}")
        return []
    check_close(label, got[qcs_name][0], exact(samples, lo, hi))
    return samples


def scores_match_trace(printed: list[float], samples: list[float], label: str) -> None:
    if len(printed) != len(samples):
        fail(f"{label}: printed {len(printed)} trace {len(samples)}")
        return
    worst = 0.0
    compared = 0
    for shown, used in zip(printed, samples):
        if math.isinf(used):
            # Identical PSNR and PU21-PSNR print 999.99. QCS still receives +inf.
            if abs(shown - 999.99) > 1e-9:
                fail(f"{label}: +inf sample printed as {shown}")
                return
            continue
        worst = max(worst, abs(shown - used))
        compared += 1
    if compared == 0 or worst > 5.1e-7:
        fail(f"{label}: printed scores differ from QCS samples by {worst}")
    else:
        print(f"OK   {label} uses printed scores")


def check_vmaf_and_pu21() -> None:
    """QCS reads the frame scores PixVQA prints. It does not recompute them."""
    sdr = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"
    hdr = ROOT / "model" / "hdr_vmaf_v1.1.8" / "hdr_vmaf_v1.1.8_3d0h_2160.json"
    if not sdr.is_file() or not hdr.is_file():
        fail(f"missing VMAF model {sdr} or {hdr}")
        return
    env = os.environ.copy()
    env["PIXVQA_QCS_TRACE"] = "1"
    with tempfile.TemporaryDirectory(prefix="qcs_vmaf_") as tmp:
        td = Path(tmp)
        ref, dist = td / "r.yuv", td / "d.yuv"
        write_420p10(ref, 320, 240, [0, 0, 0, 0])
        write_420p10(dist, 320, 240, [0, 12, 40, 4])
        sdr_default = run(
            ["--metrics", "vmaf", "--qcs", "--model", str(sdr), "--threads", "4"],
            ref, dist, "320x240", "420p:10", env=env, timeout=300)
        if sdr_default.returncode != 0 and "no libvmaf" in sdr_default.stderr:
            print("SKIP vmaf qcs: this build has no libvmaf")
            vmaf_built = False
        else:
            vmaf_built = True
            samples = qcs_from_trace(sdr_default, "VMAF-QCS", "vmaf", 70, 100, "sdr_vmaf_default")
            if qcs_map(sdr_default.stdout).get("VMAF-QCS", (None, "", ""))[1:] != ("70", "100"):
                fail(f"sdr default range {qcs_map(sdr_default.stdout).get('VMAF-QCS')}")
            printed = [float(v) for v in VMAF_LINE.findall(sdr_default.stdout)]
            if samples:
                scores_match_trace(printed, samples, "sdr_vmaf")
            sdr_custom = run(
                ["--metrics", "vmaf", "--qcs", "--qcs-range", "vmaf=60:90", "--model", str(sdr),
                 "--threads", "4"],
                ref, dist, "320x240", "420p:10", env=env, timeout=300)
            custom_samples = qcs_from_trace(sdr_custom, "VMAF-QCS", "vmaf", 60, 90, "sdr_vmaf_custom")
            if qcs_map(sdr_custom.stdout).get("VMAF-QCS", (None, "", ""))[1:] != ("60", "90"):
                fail(f"sdr custom range {qcs_map(sdr_custom.stdout).get('VMAF-QCS')}")
            if samples and custom_samples and samples != custom_samples:
                fail("custom VMAF range changed the frame scores")
            plain = run(["--metrics", "vmaf", "--model", str(sdr), "--threads", "4"],
                        ref, dist, "320x240", "420p:10", timeout=300)
            def strip(text: str) -> str:
                return "\n".join(ln for ln in text.splitlines()
                                 if "QCS" not in ln and not ln.startswith("Total frames:"))
            if plain.returncode == 0 and sdr_default.returncode == 0 and strip(plain.stdout) != strip(sdr_default.stdout):
                fail("enabling QCS changed VMAF text")
            elif plain.returncode == 0:
                print("OK   vmaf_text_unchanged")
            # HDR model clips at 110. Identical frames should reach that ceiling.
            # A range whose upper bound is 120 must keep a score above 100, so
            # clipping the metric to 100 before QCS would produce a different value.
            write_420p10(dist, 320, 240, [0, 0, 0, 0])
            hdr_args = ["--metrics", "vmaf", "--qcs", "--model", str(hdr), "--threads", "4", *HDR]
            hdr_default = run(hdr_args, ref, dist, "320x240", "420p:10", env=env, timeout=300)
            hdr_samples = qcs_from_trace(hdr_default, "VMAF-QCS", "vmaf", 70, 100, "hdr_vmaf_default")
            hdr_printed = [float(v) for v in VMAF_LINE.findall(hdr_default.stdout)]
            if hdr_samples:
                scores_match_trace(hdr_printed, hdr_samples, "hdr_vmaf")
            if not hdr_printed or max(hdr_printed) <= 100.0:
                fail(f"hdr vmaf did not report a score above 100: {hdr_printed}")
            else:
                print(f"OK   hdr_vmaf_above_100 max={max(hdr_printed):.6f}")
            if qcs_map(hdr_default.stdout).get("VMAF-QCS", (None, "", ""))[1:] != ("70", "100"):
                fail("hdr default range is not [70,100]")
            hdr_wide = run(
                ["--metrics", "vmaf", "--qcs", "--qcs-range", "vmaf=70:120", "--model", str(hdr),
                 "--threads", "4", *HDR],
                ref, dist, "320x240", "420p:10", env=env, timeout=300)
            wide_samples = qcs_from_trace(hdr_wide, "VMAF-QCS", "vmaf", 70, 120, "hdr_vmaf_above_kept")
            if wide_samples and max(wide_samples) > 100.0:
                clipped = [min(s, 100.0) for s in wide_samples]
                if exact(clipped, 70, 120) == exact(wide_samples, 70, 120):
                    fail("score above 100 did not change QCS on [70,120]")
                else:
                    print("OK   hdr_score_not_clipped_before_qcs")
            if CUDA.exists() and hdr_default.returncode == 0:
                gpu = run(hdr_args + ["--metric-device", "gpu"], ref, dist, "320x240", "420p:10",
                          bin_path=CUDA, env=env, timeout=300)
                if gpu.returncode != 0 or "using CPU libvmaf" not in gpu.stderr:
                    fail(f"gpu vmaf qcs rc={gpu.returncode} {gpu.stderr[-300:]}")
                elif qcs_map(gpu.stdout).get("VMAF-QCS") != qcs_map(hdr_default.stdout).get("VMAF-QCS"):
                    fail(f"gpu vmaf qcs {qcs_map(gpu.stdout)} cpu {qcs_map(hdr_default.stdout)}")
                else:
                    print("OK   cuda_vmaf_qcs_matches_cpu")
        # PU21 on a small frame. QCS uses the printed per-frame scores.
        write_420p10(ref, 32, 32, [0, 0, 0, 0])
        write_420p10(dist, 32, 32, [0, 8, 30, 2])
        pu = run(
            ["--metrics", "pu21-psnr,pu21-ssim,pu21-msssim", "--qcs", *HDR],
            ref, dist, "32x32", "420p:10", env=env, timeout=180)
        pu_psnr = qcs_from_trace(pu, "PU21-PSNR-QCS", "pu21-psnr", 30, 50, "pu21_psnr_default")
        pu_ssim = qcs_from_trace(pu, "PU21-SSIM-QCS", "pu21-ssim", 0.8, 1.0, "pu21_ssim_default")
        pu_ms = qcs_from_trace(pu, "PU21-MS-SSIM-QCS", "pu21-msssim", 0.8, 1.0, "pu21_msssim_default")
        got_pu = qcs_map(pu.stdout)
        if got_pu.get("PU21-PSNR-QCS", (None, "", ""))[1:] != ("30", "50"):
            fail(f"pu21-psnr range {got_pu.get('PU21-PSNR-QCS')}")
        if got_pu.get("PU21-SSIM-QCS", (None, "", ""))[1:] != ("0.8", "1.0"):
            fail(f"pu21-ssim range {got_pu.get('PU21-SSIM-QCS')}")
        if got_pu.get("PU21-MS-SSIM-QCS", (None, "", ""))[1:] != ("0.8", "1.0"):
            fail(f"pu21-msssim range {got_pu.get('PU21-MS-SSIM-QCS')}")
        printed_pu = {}
        for name, value in PU_LINE.findall(pu.stdout):
            printed_pu.setdefault(name, []).append(float(value))
        # Summary lines repeat the names. Frame lines come first, four of them.
        if pu_psnr:
            scores_match_trace(printed_pu.get("PU21-PSNR", [])[:len(pu_psnr)], pu_psnr, "pu21_psnr")
        if pu_ssim:
            scores_match_trace(printed_pu.get("PU21-SSIM", [])[:len(pu_ssim)], pu_ssim, "pu21_ssim")
        if pu_ms:
            scores_match_trace(printed_pu.get("PU21-MS-SSIM", [])[:len(pu_ms)], pu_ms, "pu21_msssim")
        pu_custom = run(
            ["--metrics", "pu21-psnr,pu21-ssim,pu21-msssim", "--qcs",
             "--qcs-range", "pu21-psnr=40:60", "--qcs-range", "pu21-ssim=0.9:1.0",
             "--qcs-range", "pu21-msssim=0.7:0.95", *HDR],
            ref, dist, "32x32", "420p:10", env=env, timeout=180)
        qcs_from_trace(pu_custom, "PU21-PSNR-QCS", "pu21-psnr", 40, 60, "pu21_psnr_custom")
        qcs_from_trace(pu_custom, "PU21-SSIM-QCS", "pu21-ssim", 0.9, 1.0, "pu21_ssim_custom")
        qcs_from_trace(pu_custom, "PU21-MS-SSIM-QCS", "pu21-msssim", 0.7, 0.95, "pu21_msssim_custom")
        custom_pu = qcs_map(pu_custom.stdout)
        if custom_pu.get("PU21-PSNR-QCS", (None, "", ""))[1:] != ("40", "60"):
            fail(f"pu21 custom psnr range {custom_pu.get('PU21-PSNR-QCS')}")
        if custom_pu.get("PU21-SSIM-QCS", (None, "", ""))[1:] != ("0.9", "1.0"):
            fail(f"pu21 custom ssim range {custom_pu.get('PU21-SSIM-QCS')}")
        if custom_pu.get("PU21-MS-SSIM-QCS", (None, "", ""))[1:] != ("0.7", "0.95"):
            fail(f"pu21 custom msssim range {custom_pu.get('PU21-MS-SSIM-QCS')}")
        write_420p10(dist, 32, 32, [0, 0, 0, 0])
        ident = run(["--metrics", "pu21-psnr,pu21-ssim,pu21-msssim", "--qcs", *HDR],
                    ref, dist, "32x32", "420p:10", env=env, timeout=180)
        ident_psnr = trace_of(ident.stderr, "pu21-psnr")
        if ident.returncode != 0 or not ident_psnr or not all(math.isinf(s) and s > 0 for s in ident_psnr):
            fail(f"identical pu21-psnr trace {ident_psnr} rc={ident.returncode} {ident.stderr[-200:]}")
        elif "999.99" not in ident.stdout:
            fail("identical PU21-PSNR lost 999.99")
        else:
            check_close("identical_pu21_psnr_qcs", qcs_map(ident.stdout)["PU21-PSNR-QCS"][0], 100.0)
            check_close("identical_pu21_ssim_qcs", qcs_map(ident.stdout)["PU21-SSIM-QCS"][0], 100.0)
            check_close("identical_pu21_msssim_qcs", qcs_map(ident.stdout)["PU21-MS-SSIM-QCS"][0], 100.0)
        # Simultaneous metrics. Each QCS uses that metric's own default range.
        write_420p10(ref, 320, 240, [0, 0, 0, 0])
        write_420p10(dist, 320, 240, [0, 6, 24, 2])
        if vmaf_built:
            mixed = run(
                ["--metrics", "psnr,ssim,vmaf,pu21-psnr", "--qcs", "--model", str(hdr),
                 "--threads", "4", *HDR],
                ref, dist, "320x240", "420p:10", env=env, timeout=300)
            qcs_from_trace(mixed, "VMAF-QCS", "vmaf", 70, 100, "mixed_vmaf")
            expected = {"PSNR-QCS", "SSIM-QCS", "VMAF-QCS", "PU21-PSNR-QCS"}
        else:
            mixed = run(
                ["--metrics", "psnr,ssim,pu21-psnr", "--qcs", *HDR],
                ref, dist, "320x240", "420p:10", env=env, timeout=180)
            expected = {"PSNR-QCS", "SSIM-QCS", "PU21-PSNR-QCS"}
        qcs_from_trace(mixed, "PSNR-QCS", "psnr", 30, 50, "mixed_psnr")
        qcs_from_trace(mixed, "SSIM-QCS", "ssim", 0.8, 1.0, "mixed_ssim")
        qcs_from_trace(mixed, "PU21-PSNR-QCS", "pu21-psnr", 30, 50, "mixed_pu21_psnr")
        mixed_map = qcs_map(mixed.stdout)
        if set(mixed_map) != expected:
            fail(f"mixed qcs names {sorted(mixed_map)}")
        else:
            print("OK   mixed_metric_names")


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
            (["--metrics", "psnr", "--qcs", "--qcs-range", "pu21-vmaf=1:2"], "unknown"),
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
            for metric, label in (("ssim", "SSIM-QCS"), ("msssim", "MS-SSIM-QCS")):
                cpu_m = run(["--metrics", metric, "--qcs"], ref, tst, size, "444p:10")
                gpu_m = run(["--metric-device", "gpu", "--metrics", metric, "--qcs"],
                            ref, tst, size, "444p:10", bin_path=CUDA)
                if cpu_m.returncode or gpu_m.returncode:
                    fail(f"cuda {metric} rc cpu={cpu_m.returncode} gpu={gpu_m.returncode} {gpu_m.stderr[-200:]}")
                elif qcs_map(cpu_m.stdout).get(label) != qcs_map(gpu_m.stdout).get(label):
                    fail(f"cuda {metric} qcs {qcs_map(gpu_m.stdout)} cpu {qcs_map(cpu_m.stdout)}")
                else:
                    print(f"OK   cuda_{metric}_qcs")
            cpu_x = run(["--fps", "30", "--metrics", "xpsnr", "--qcs"], ref, tst, size, "444p:10")
            gpu_x = run(["--metric-device", "gpu", "--fps", "30", "--metrics", "xpsnr", "--qcs"],
                        ref, tst, size, "444p:10", bin_path=CUDA)
            if cpu_x.returncode or gpu_x.returncode:
                fail(f"cuda xpsnr rc cpu={cpu_x.returncode} gpu={gpu_x.returncode} {gpu_x.stderr[-200:]}")
            elif qcs_map(cpu_x.stdout).get("XPSNR-QCS") != qcs_map(gpu_x.stdout).get("XPSNR-QCS"):
                fail(f"cuda xpsnr qcs {qcs_map(gpu_x.stdout)} cpu {qcs_map(cpu_x.stdout)}")
            else:
                print("OK   cuda_xpsnr_qcs")
    check_vmaf_and_pu21()
    if FAILS:
        print(f"{FAILS} QCS checks failed", file=sys.stderr)
        return 1
    print("All QCS checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
