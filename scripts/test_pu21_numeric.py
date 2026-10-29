#!/usr/bin/env python3
"""Numerical checks for PU21-PSNR, PU21-SSIM, and PU21-MS-SSIM."""
from __future__ import annotations

import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
W = H = 32
P = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def pu21(y: float) -> float:
    y = min(max(y, 0.005), 10000.0)
    yp = y ** P[3]
    inner = (P[0] + P[1] * yp) / (1.0 + P[2] * yp)
    return max(P[6] * (inner ** P[4] - P[5]), 0.0)


def pq_eotf(e: float) -> float:
    if e <= 0.0:
        return 0.0
    e = min(e, 1.0)
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    npow = e ** (1.0 / m2)
    num = max(npow - c1, 0.0)
    den = c2 - c3 * npow
    return 10000.0 * (num / den) ** (1.0 / m1)


def limited_code(e: float) -> int:
    return int(round(e * 219.0 * 4.0 + 16.0 * 4.0))


def write_grey(path: Path, code: int) -> None:
    y = struct.pack("<" + "H" * (W * H), *([code] * (W * H)))
    c = struct.pack("<" + "H" * ((W // 2) * (H // 2)), *([512] * ((W // 2) * (H // 2))))
    path.write_bytes((y + c + c) * 1)


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(BIN), *args], capture_output=True, text=True)


def score(text: str, name: str) -> float | None:
    match = re.search(rf"\[ {re.escape(name)}:([0-9.]+) \]", text)
    if not match:
        return None
    return float(match.group(1))


def expect_close(label: str, got: float | None, exp: float, tol: float) -> None:
    if got is None or abs(got - exp) > tol:
        fail(f"{label}: got={got} exp={exp}")
    else:
        print(f"OK   {label}: {got:.8f} err={got - exp:.3e}")


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN, file=sys.stderr)
        return 2
    code_a = limited_code(0.5)
    # A second legal limited code, still grey, so RGB stays equal.
    code_b = limited_code(0.58)
    nits_a = pq_eotf((code_a - 64.0) / (219.0 * 4.0))
    nits_b = pq_eotf((code_b - 64.0) / (219.0 * 4.0))
    pu_a = pu21(nits_a)
    pu_b = pu21(nits_b)
    mse = (pu_a - pu_b) ** 2
    expect_psnr = 10.0 * math.log10((256.0 * 256.0) / mse)
    c1 = (0.01 * 256.0) ** 2
    expect_ssim = (2.0 * pu_a * pu_b + c1) / (pu_a * pu_a + pu_b * pu_b + c1)
    # 32x32 uses two MS-SSIM scales. Uniform contrast/structure terms are 1.
    expect_msssim = expect_ssim ** 0.2856

    with tempfile.TemporaryDirectory(prefix="pixvqa_pu21_") as td:
        t = Path(td)
        ref = t / "ref.yuv"
        dist = t / "dist.yuv"
        write_grey(ref, code_a)
        write_grey(dist, code_b)
        base = [
            "-r", str(ref), "-s", f"{W}x{H}", "-f", "420p:10",
            "-t", str(dist), "-s", f"{W}x{H}", "-f", "420p:10",
            "--hdr-transfer", "pq", "-n", "1", "-v", "0",
        ]
        done = run(base + ["--metrics", "psnr,pu21-psnr,pu21-ssim,pu21-msssim"])
        if done.returncode != 0:
            fail(f"score rc={done.returncode} {done.stderr[-500:]}")
        else:
            text = done.stdout
            if "variant=banding_glare" not in text or "primaries assumed bt2020" not in text:
                fail(f"missing pu21 provenance:\n{text}")
            elif "range assumed limited" not in text or "(assumption)" not in text:
                fail(f"missing assumption source:\n{text}")
            else:
                expect_close("pu21-psnr", score(text, "PU21-PSNR"), expect_psnr, 1e-4)
                expect_close("pu21-ssim", score(text, "PU21-SSIM"), expect_ssim, 1e-5)
                expect_close("pu21-msssim", score(text, "PU21-MS-SSIM"), expect_msssim, 1e-5)
            ordinary = run(base + ["--metrics", "psnr"])
            both = run(base + ["--metrics", "psnr,pu21-psnr"])
            psnr_re = re.compile(r"\[ PSNR  y:([0-9.]+) u:([0-9.]+) v:([0-9.]+) avg:([0-9.]+) \]")
            if ordinary.returncode != 0 or both.returncode != 0:
                fail("psnr regression command failed")
            elif not psnr_re.search(ordinary.stdout) or psnr_re.search(ordinary.stdout).group(0) != psnr_re.search(both.stdout).group(0):
                fail("adding pu21-psnr changed ordinary PSNR")
            else:
                print("OK   ordinary PSNR unchanged")

            def level(label: str, e0: float, e1: float) -> None:
                c0, c1c = limited_code(e0), limited_code(e1)
                n0 = pq_eotf((c0 - 64.0) / (219.0 * 4.0))
                n1 = pq_eotf((c1c - 64.0) / (219.0 * 4.0))
                delta = pu21(n0) - pu21(n1)
                expect = 10.0 * math.log10((256.0 * 256.0) / (delta * delta))
                write_grey(ref, c0)
                write_grey(dist, c1c)
                level_run = run(base + ["--metrics", "pu21-psnr"])
                if level_run.returncode != 0:
                    fail(f"{label} rc={level_run.returncode} {level_run.stderr[-300:]}")
                else:
                    expect_close(label, score(level_run.stdout, "PU21-PSNR"), expect, 1e-4)

            level("pu21-psnr dark", 0.15, 0.18)
            level("pu21-psnr highlight", 0.85, 0.90)

        same = run([
            "-r", str(ref), "-s", f"{W}x{H}", "-f", "420p:10",
            "-t", str(ref), "-s", f"{W}x{H}", "-f", "420p:10",
            "--metrics", "pu21-psnr,pu21-ssim,pu21-msssim",
            "--hdr-transfer", "pq", "--hdr-primaries", "bt2020",
            "--hdr-matrix", "bt2020nc", "--hdr-range", "limited",
            "--hdr-chroma-loc", "left", "--hdr-chroma-up", "bilinear",
            "-n", "1", "-v", "0",
        ])
        if same.returncode != 0 or "999.99" not in same.stdout:
            fail(f"identical PU21-PSNR: rc={same.returncode} {same.stdout[-300:]} {same.stderr[-300:]}")
        elif score(same.stdout, "PU21-SSIM") != 1.0 or score(same.stdout, "PU21-MS-SSIM") != 1.0:
            fail(f"identical structural scores: {same.stdout}")
        elif "(override)" not in same.stdout or "assumed" in same.stdout:
            fail(f"explicit config still assumed a field:\n{same.stdout}")
        else:
            print("OK   identical frames and explicit sources")

        missing = run(base + ["--metrics", "pu21-psnr", "--hdr-transfer", "auto"])
        # auto leaves transfer unset. The base list already passed --hdr-transfer pq,
        # so build this command without it.
        missing = run([
            "-r", str(ref), "-s", f"{W}x{H}", "-f", "420p:10",
            "-t", str(dist), "-s", f"{W}x{H}", "-f", "420p:10",
            "--metrics", "pu21-psnr", "-n", "1", "-v", "0",
        ])
        if missing.returncode == 0 or "unspecified" not in (missing.stderr + missing.stdout):
            fail(f"missing transfer was accepted: {missing.stderr[-300:]}")
        elif "Summary results" in missing.stdout:
            fail("missing transfer printed a score")
        else:
            print("OK   missing transfer rejected")

        for name in ("pu21-xpsnr", "pu21-vmaf"):
            blocked = run(base + ["--metrics", name])
            text = blocked.stderr + blocked.stdout
            if blocked.returncode == 0 or "Summary results" in blocked.stdout:
                fail(f"{name} produced a score")
            elif f"unsupported metric {name}" not in text:
                fail(f"{name} was not rejected as unsupported: {text[-400:]}")
            elif "PU21-PSNR" in blocked.stdout or "PU21-XPSNR" in blocked.stdout or "PU21-VMAF" in blocked.stdout:
                fail(f"{name} printed a quality score")
            else:
                print(f"OK   {name} unsupported")

        gpu = run(base + ["--metrics", "pu21-psnr", "--metric-device", "gpu"])
        if gpu.returncode == 0 or "GPU" not in gpu.stderr:
            fail(f"gpu pu21 was accepted: {gpu.stderr[-300:]}")
        else:
            print("OK   gpu pu21 refused")

        help_hdr = run(["--help-hdr"])
        if help_hdr.returncode != 0 or "10.1109/PCS50896.2021.9477471" not in help_hdr.stdout:
            fail("help-hdr missing the paper DOI")
        elif "gfxdisp/pu21" not in help_hdr.stdout or "--ref-hdr-transfer" not in help_hdr.stdout:
            fail("help-hdr missing citation or per-input example")
        else:
            print("OK   help-hdr")
        help_main = run(["--help"])
        supported = ("psnr", "xpsnr", "ssim", "msssim", "vmaf", "pu21-psnr", "pu21-ssim", "pu21-msssim")
        if any(name not in help_main.stdout for name in supported):
            fail("ordinary help does not list the eight supported metrics")
        elif "pu21-xpsnr" in help_main.stdout or "pu21-vmaf" in help_main.stdout:
            fail("ordinary help still lists an excluded experimental metric")
        elif "pu21-xpsnr" in help_hdr.stdout or "pu21-vmaf" in help_hdr.stdout:
            fail("help-hdr still lists an excluded experimental metric")
        else:
            print("OK   help lists the eight supported metrics")

        check_overrides(run)
        check_hlg_black_ootf(run)

    if FAILS:
        print(FAILS, "failed", file=sys.stderr)
        return 1
    print("All PU21 numerical checks passed")
    return 0


def coef(matrix: str) -> tuple[float, float, float]:
    if matrix == "bt709":
        return 0.2126, 0.7152, 0.0722
    return 0.2627, 0.6780, 0.0593


def nominal(code: float, chroma: bool, full: bool, bd: int = 10) -> float:
    if full:
        maxv = float((1 << bd) - 1)
        return code / maxv - (0.5 if chroma else 0.0)
    scale = float(1 << (bd - 8))
    if chroma:
        return (code - 128.0 * scale) / (224.0 * scale)
    return (code - 16.0 * scale) / (219.0 * scale)


def rgb_ncl(kr: float, kg: float, kb: float, y: float, cb: float, cr: float) -> tuple[float, float, float]:
    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return r, g, b


def hlg_scene(e: float) -> float:
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 1.0 - a * math.log(12.0 - b)
    e = min(max(e, 0.0), 1.0)
    if e <= 0.5:
        return (e * e) / 3.0
    return (math.exp((e - c) / a) + b) / 12.0


def display_pq(rgb: tuple[float, float, float], clip: float | None) -> tuple[float, float, float]:
    out = tuple(pq_eotf(c) for c in rgb)
    if clip is None:
        return out  # type: ignore[return-value]
    return tuple(min(c, clip) for c in out)  # type: ignore[return-value]


def display_hlg(rgb: tuple[float, float, float], kr: float, kg: float, kb: float, peak: float,
                black: float, gamma: float) -> tuple[float, float, float]:
    """BT.2100 reference OOTF: Rd = (Lw-Lb) * Ys^(gamma-1) * Rs + Lb."""
    scene = tuple(hlg_scene(c) for c in rgb)
    ys = kr * scene[0] + kg * scene[1] + kb * scene[2]
    if ys <= 0.0:
        return black, black, black
    if black == 0.0:
        scale = peak * (ys ** gamma) / ys
        return scene[0] * scale, scene[1] * scale, scene[2] * scale
    factor = (peak - black) * (ys ** (gamma - 1.0))
    return scene[0] * factor + black, scene[1] * factor + black, scene[2] * factor + black


def display_hlg_proportional(rgb: tuple[float, float, float], kr: float, kg: float, kb: float, peak: float,
                             black: float, gamma: float) -> tuple[float, float, float]:
    """Pre-correction OOTF, which folds Lb into the scale. Used only to show it differs."""
    scene = tuple(hlg_scene(c) for c in rgb)
    ys = kr * scene[0] + kg * scene[1] + kb * scene[2]
    if ys <= 0.0:
        return black, black, black
    yd = black + (peak - black) * (ys ** gamma)
    scale = yd / ys
    return scene[0] * scale, scene[1] * scale, scene[2] * scale


def psnr_of(ref_rgb: tuple[float, float, float], dist_rgb: tuple[float, float, float]) -> float:
    mse = sum((a - b) ** 2 for a, b in zip((pu21(c) for c in ref_rgb), (pu21(c) for c in dist_rgb))) / 3.0
    return 10.0 * math.log10((256.0 * 256.0) / mse)


def write_444(path: Path, y: int, cb: int, cr: int) -> None:
    n = W * H
    plane = lambda code: struct.pack("<" + "H" * n, *([code] * n))
    path.write_bytes(plane(y) + plane(cb) + plane(cr))


def sample_clamp(data: list[int], w: int, h: int, x: int, y: int) -> float:
    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)
    return float(data[y * w + x])


def reconstruct(data: list[int], w: int, h: int, x: float, y: float, nearest: bool) -> float:
    x = min(max(x, 0.0), float(w - 1))
    y = min(max(y, 0.0), float(h - 1))
    if nearest:
        return sample_clamp(data, w, h, int(math.floor(x + 0.5)), int(math.floor(y + 0.5)))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx = x - x0
    ty = y - y0
    a = sample_clamp(data, w, h, x0, y0)
    b = sample_clamp(data, w, h, x1, y0)
    c = sample_clamp(data, w, h, x0, y1)
    d = sample_clamp(data, w, h, x1, y1)
    return (1.0 - tx) * (1.0 - ty) * a + tx * (1.0 - ty) * b + (1.0 - tx) * ty * c + tx * ty * d


def chroma_xy(loc: str, x: int, y: int) -> tuple[float, float]:
    if loc == "center":
        return (x - 0.5) / 2.0, (y - 0.5) / 2.0
    if loc == "topleft":
        return x / 2.0, y / 2.0
    return x / 2.0, (y - 0.5) / 2.0


def psnr_420(y0: int, cb0: list[int], cr0: list[int], y1: int, cb1: list[int], cr1: list[int], loc: str,
             nearest: bool, full: bool, matrix: str) -> float:
    kr, kg, kb = coef(matrix)
    cw, ch = W // 2, H // 2
    acc = 0.0
    for y in range(H):
        for x in range(W):
            cx, cy = chroma_xy(loc, x, y)
            def one(yc: int, cb: list[int], cr: list[int]) -> tuple[float, float, float]:
                code_cb = reconstruct(cb, cw, ch, cx, cy, nearest)
                code_cr = reconstruct(cr, cw, ch, cx, cy, nearest)
                return rgb_ncl(kr, kg, kb, nominal(yc, False, full), nominal(code_cb, True, full),
                               nominal(code_cr, True, full))
            left = display_pq(one(y0, cb0, cr0), None)
            right = display_pq(one(y1, cb1, cr1), None)
            acc += sum((pu21(a) - pu21(b)) ** 2 for a, b in zip(left, right))
    mse = acc / float(W * H * 3)
    return 10.0 * math.log10((256.0 * 256.0) / mse)


def write_420_planes(path: Path, y: int, cb: list[int], cr: list[int]) -> None:
    yb = struct.pack("<" + "H" * (W * H), *([y] * (W * H)))
    cb_b = struct.pack("<" + "H" * len(cb), *cb)
    cr_b = struct.pack("<" + "H" * len(cr), *cr)
    path.write_bytes(yb + cb_b + cr_b)


def run_psnr(run, args: list[str]) -> tuple[subprocess.CompletedProcess[str], float | None]:
    done = run(args)
    return done, score(done.stdout, "PU21-PSNR")


def check_hlg_black_ootf(run) -> None:
    """Neutral grey and saturated RGB at black 0 and 5 cd/m2.

    Grey and black 0 agree with the old proportional scale. Saturated color at
    5 cd/m2 must follow BT.2100 and must not follow that scale.
    """
    kr, kg, kb = coef("bt2020")
    gamma = 1.2
    peak = 1000.0
    grey = ((480, 512, 512), (560, 512, 512))
    saturated = ((480, 512, 800), (560, 400, 700))
    with tempfile.TemporaryDirectory(prefix="pixvqa_pu21_hlg_black_") as td:
        root = Path(td)
        for kind, pair_codes in (("grey", grey), ("saturated", saturated)):
            ref = root / f"{kind}_ref.yuv"
            dist = root / f"{kind}_dist.yuv"
            write_444(ref, *pair_codes[0])
            write_444(dist, *pair_codes[1])
            for black in (0.0, 5.0):
                def light(codes: tuple[int, int, int], formula) -> tuple[float, float, float]:
                    nonlinear = rgb_ncl(kr, kg, kb, nominal(codes[0], False, False),
                                        nominal(codes[1], True, False), nominal(codes[2], True, False))
                    return formula(nonlinear, kr, kg, kb, peak, black, gamma)

                bt2100 = psnr_of(light(pair_codes[0], display_hlg), light(pair_codes[1], display_hlg))
                old = psnr_of(light(pair_codes[0], display_hlg_proportional),
                              light(pair_codes[1], display_hlg_proportional))
                label = f"hlg {kind} black {black:g}"
                if kind == "grey" or black == 0.0:
                    if abs(bt2100 - old) > 1e-9:
                        fail(f"{label}: zero-black or grey formulas diverged {bt2100} vs {old}")
                elif abs(bt2100 - old) < 1e-3:
                    fail(f"{label}: BT.2100 did not separate from the proportional OOTF")
                args = ["-r", str(ref), "-s", f"{W}x{H}", "-f", "444p:10",
                        "-t", str(dist), "-s", f"{W}x{H}", "-f", "444p:10",
                        "--metrics", "pu21-psnr", "--hdr-transfer", "hlg",
                        "--hdr-primaries", "bt2020", "--hdr-matrix", "bt2020nc",
                        "--hdr-range", "limited", "--hdr-peak-nits", "1000",
                        "--hdr-black-nits", f"{black:g}", "-n", "1", "-v", "0"]
                done, got = run_psnr(run, args)
                if done.returncode != 0:
                    fail(f"{label}: rc={done.returncode} {done.stderr[-300:]}")
                    continue
                expect_close(label, got, bt2100, 1e-4)
                if got is not None and kind == "saturated" and black == 5.0 and abs(got - old) <= 1e-3:
                    fail(f"{label}: score still matches the proportional OOTF")


def check_overrides(run) -> None:
    """Every implemented HDR knob against an independent PU21-PSNR calculation."""
    with tempfile.TemporaryDirectory(prefix="pixvqa_pu21_over_") as td:
        root = Path(td)
        ref = root / "ref.yuv"
        dist = root / "dist.yuv"
        write_444(ref, 520, 420, 640)
        write_444(dist, 560, 470, 600)
        base = ["-r", str(ref), "-s", f"{W}x{H}", "-f", "444p:10",
                "-t", str(dist), "-s", f"{W}x{H}", "-f", "444p:10",
                "--metrics", "pu21-psnr", "-n", "1", "-v", "0"]

        def pair(full: bool, matrix: str, kind: str, clip: float | None = None, peak: float = 1000.0,
                 black: float = 0.0, gamma: float | None = None) -> float:
            ckr, ckg, ckb = coef(matrix)
            def pix(y: int, cb: int, cr: int) -> tuple[float, float, float]:
                nonlinear = rgb_ncl(ckr, ckg, ckb, nominal(y, False, full), nominal(cb, True, full),
                                    nominal(cr, True, full))
                if kind == "pq":
                    return display_pq(nonlinear, clip)
                g = 1.2 + 0.42 * math.log10(peak / 1000.0) if gamma is None else gamma
                return display_hlg(nonlinear, ckr, ckg, ckb, peak, black, g)
            return psnr_of(pix(520, 420, 640), pix(560, 470, 600))

        cases = [
            ("range_limited", base + ["--hdr-transfer", "pq", "--hdr-primaries", "bt2020",
                                      "--hdr-matrix", "bt2020nc", "--hdr-range", "limited"],
             pair(False, "bt2020", "pq")),
            ("range_full", base + ["--hdr-transfer", "pq", "--hdr-primaries", "bt2020",
                                   "--hdr-matrix", "bt2020nc", "--hdr-range", "full"],
             pair(True, "bt2020", "pq")),
            ("matrix_bt709", base + ["--hdr-transfer", "pq", "--hdr-primaries", "bt709",
                                     "--hdr-matrix", "bt709", "--hdr-range", "limited"],
             pair(False, "bt709", "pq")),
            ("pq_clip_100", base + ["--hdr-transfer", "pq", "--hdr-primaries", "bt2020",
                                    "--hdr-matrix", "bt2020nc", "--hdr-range", "limited",
                                    "--hdr-peak-nits", "100"],
             pair(False, "bt2020", "pq", clip=100.0)),
            ("hlg_peak_1000", base + ["--hdr-transfer", "hlg", "--hdr-primaries", "bt2020",
                                      "--hdr-matrix", "bt2020nc", "--hdr-range", "limited",
                                      "--hdr-peak-nits", "1000"],
             pair(False, "bt2020", "hlg", peak=1000.0)),
            ("hlg_peak_400", base + ["--hdr-transfer", "hlg", "--hdr-primaries", "bt2020",
                                     "--hdr-matrix", "bt2020nc", "--hdr-range", "limited",
                                     "--hdr-peak-nits", "400"],
             pair(False, "bt2020", "hlg", peak=400.0)),
            ("hlg_black", base + ["--hdr-transfer", "hlg", "--hdr-primaries", "bt2020",
                                  "--hdr-matrix", "bt2020nc", "--hdr-range", "limited",
                                  "--hdr-peak-nits", "1000", "--hdr-black-nits", "5"],
             pair(False, "bt2020", "hlg", peak=1000.0, black=5.0)),
            ("hlg_gamma", base + ["--hdr-transfer", "hlg", "--hdr-primaries", "bt2020",
                                  "--hdr-matrix", "bt2020nc", "--hdr-range", "limited",
                                  "--hdr-peak-nits", "1000", "--hdr-hlg-gamma", "1.0"],
             pair(False, "bt2020", "hlg", peak=1000.0, gamma=1.0)),
        ]
        got_scores: dict[str, float] = {}
        for label, args, exp in cases:
            done, got = run_psnr(run, args)
            if done.returncode != 0:
                fail(f"{label}: rc={done.returncode} {done.stderr[-300:]}")
                continue
            expect_close(label, got, exp, 1e-4)
            if got is not None:
                got_scores[label] = got
        if "range_limited" in got_scores and "range_full" in got_scores:
            delta = abs(got_scores["range_limited"] - got_scores["range_full"])
            if delta < 1e-3:
                fail("range override did not change PU21-PSNR")
            else:
                print(f"OK   range changes the score by {delta:.6f} dB")
        for left, right, what in (
            ("range_limited", "matrix_bt709", "matrix"),
            ("range_limited", "pq_clip_100", "PQ clip"),
            ("hlg_peak_1000", "hlg_peak_400", "HLG peak"),
            ("hlg_peak_1000", "hlg_black", "HLG black"),
            ("hlg_peak_1000", "hlg_gamma", "HLG gamma"),
        ):
            if left in got_scores and right in got_scores and abs(got_scores[left] - got_scores[right]) < 1e-3:
                fail(f"{what} override did not change PU21-PSNR")
        gamma_out = run(base + ["--hdr-transfer", "hlg", "--hdr-primaries", "bt2020",
                                "--hdr-matrix", "bt2020nc", "--hdr-range", "limited",
                                "--hdr-peak-nits", "1000", "--hdr-hlg-gamma", "1.0"])
        if "gamma=1.000000000 (override)" not in gamma_out.stdout:
            fail(f"explicit gamma was not recorded as override: {gamma_out.stdout[-400:]}")
        else:
            print("OK   explicit gamma recorded")

        cw, ch = W // 2, H // 2
        def grid(left: int, right: int) -> list[int]:
            out = []
            for row in range(ch):
                for col in range(cw):
                    out.append(left if col < cw // 2 else right)
            return out
        cref = root / "cref.yuv"
        cdist = root / "cdist.yuv"
        write_420_planes(cref, 512, grid(380, 700), grid(620, 400))
        write_420_planes(cdist, 540, grid(430, 660), grid(580, 450))
        cbase = ["-r", str(cref), "-s", f"{W}x{H}", "-f", "420p:10",
                 "-t", str(cdist), "-s", f"{W}x{H}", "-f", "420p:10",
                 "--metrics", "pu21-psnr", "--hdr-transfer", "pq",
                 "--hdr-primaries", "bt2020", "--hdr-matrix", "bt2020nc",
                 "--hdr-range", "limited", "-n", "1", "-v", "0"]
        chroma_cases = [
            ("chroma_left_bilinear", ["--hdr-chroma-loc", "left", "--hdr-chroma-up", "bilinear"], "left", False),
            ("chroma_center_bilinear", ["--hdr-chroma-loc", "center", "--hdr-chroma-up", "bilinear"], "center", False),
            ("chroma_topleft_nearest", ["--hdr-chroma-loc", "topleft", "--hdr-chroma-up", "nearest"], "topleft", True),
        ]
        chroma_scores = {}
        for label, extra, loc, nearest in chroma_cases:
            exp = psnr_420(512, grid(380, 700), grid(620, 400), 540, grid(430, 660), grid(580, 450),
                           loc, nearest, False, "bt2020")
            done, got = run_psnr(run, cbase + extra)
            if done.returncode != 0:
                fail(f"{label}: rc={done.returncode} {done.stderr[-300:]}")
                continue
            expect_close(label, got, exp, 1e-4)
            if got is not None:
                chroma_scores[label] = got
        if len(chroma_scores) == 3 and len(set(round(v, 4) for v in chroma_scores.values())) < 2:
            fail(f"chroma location and reconstruction did not change the score: {chroma_scores}")
        elif len(chroma_scores) == 3:
            print("OK   chroma location and reconstruction change PU21-PSNR")

        pq_hlg = run(cbase + ["--ref-hdr-transfer", "pq", "--dist-hdr-transfer", "hlg",
                              "--dist-hdr-peak-nits", "1000", "--hdr-chroma-loc", "left",
                              "--hdr-chroma-up", "bilinear"])
        if pq_hlg.returncode != 0 or score(pq_hlg.stdout, "PU21-PSNR") is None:
            fail(f"pq versus hlg: rc={pq_hlg.returncode} {pq_hlg.stderr[-300:]}")
        elif "reference: transfer=pq" not in pq_hlg.stdout or "distorted: transfer=hlg" not in pq_hlg.stdout:
            fail("pq versus hlg did not keep both transfers")
        else:
            print("OK   pq versus hlg scored independently")
        mixed = run(cbase + ["--metrics", "psnr,pu21-psnr", "--ref-hdr-transfer", "pq",
                             "--dist-hdr-transfer", "hlg", "--dist-hdr-peak-nits", "1000"])
        if mixed.returncode != 0 or "conventional:" not in mixed.stdout or "[ PSNR" not in mixed.stdout:
            fail(f"mixed conventional notice: rc={mixed.returncode} {mixed.stdout[-300:]} {mixed.stderr[-200:]}")
        elif score(mixed.stdout, "PU21-PSNR") is None:
            fail("mixed run lost the PU21 score")
        else:
            print("OK   pq versus hlg keeps sample-domain PSNR and a light-domain PU21 score")


if __name__ == "__main__":
    sys.exit(main())
