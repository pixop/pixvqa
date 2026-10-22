#!/usr/bin/env python3
"""Compare pixvqa XPSNR with the installed FFmpeg xpsnr filter.

FFmpeg is the oracle for Y/U/V and the sequence minimum. The pixvqa avg
score is checked from the per-frame weighted-SSE debug line and the
sample-count formula, not from FFmpeg.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "cpu" / "pixvqa"
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or ""

FRAME_RE = re.compile(
    r"\[ XPSNR y:(\S+) u:(\S+) v:(\S+) avg:(\S+) min:(\S+) \]"
)
WSSE_RE = re.compile(r"xpsnr_wsse (\d+) (\d+) (\d+) (\d+)")
FF_FRAME = re.compile(
    r"n:\s*(\d+)\s+XPSNR y:\s*(\S+)\s+XPSNR u:\s*(\S+)\s+XPSNR v:\s*(\S+)"
)
FF_AVG = re.compile(
    r"XPSNR average,\s*(\d+)\s+frames\s+y:\s*(\S+)\s+u:\s*(\S+)\s+v:\s*(\S+)\s+\(minimum:\s*(\S+)\)"
)

FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def parse_num(text: str) -> float:
    if text.lower() in {"inf", "infinity", "+inf"}:
        return math.inf
    if text.lower() in {"-inf", "-infinity"}:
        return -math.inf
    if text.lower() == "nan":
        return math.nan
    return float(text)


def same_printed(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0)
    return f"{a:.4f}" == f"{b:.4f}"


def chroma_size(w: int, h: int, chroma: str) -> tuple[int, int]:
    if chroma == "420":
        return w // 2, h // 2
    if chroma == "422":
        return w // 2, h
    return w, h


def write_yuv(path: Path, w: int, h: int, chroma: str, bits: int, frames: list[tuple]) -> None:
    cw, ch = chroma_size(w, h, chroma)
    mask = (1 << bits) - 1
    chunks = []
    for y_plane, u_plane, v_plane in frames:
        samples = []
        for plane, pw, ph in ((y_plane, w, h), (u_plane, cw, ch), (v_plane, cw, ch)):
            if len(plane) != pw * ph:
                raise RuntimeError(f"plane length {len(plane)} != {pw * ph}")
            samples.extend(int(v) & mask for v in plane)
        if bits <= 8:
            chunks.append(bytes(samples))
        else:
            chunks.append(struct.pack("<" + "H" * len(samples), *samples))
    path.write_bytes(b"".join(chunks))


def solid(w: int, h: int, value: int) -> list[int]:
    return [value] * (w * h)


def fn_plane(w: int, h: int, fn) -> list[int]:
    out = [0] * (w * h)
    k = 0
    for y in range(h):
        for x in range(w):
            out[k] = int(fn(x, y))
            k += 1
    return out


def run_ffmpeg(ref: Path, test: Path, w: int, h: int, pix: str, fps: str, stats: Path) -> str:
    cmd = [
        FFMPEG, "-hide_banner", "-nostats",
        "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", fps, "-i", str(ref),
        "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", fps, "-i", str(test),
        "-lavfi", f"[0:v][1:v]xpsnr=stats_file={stats}",
        "-f", "null", "-"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={done.returncode}\n{done.stderr[-800:]}")
    return stats.read_text(errors="replace")


def run_pixvqa(ref: Path, test: Path, w: int, h: int, fmt: str, fps: str) -> subprocess.CompletedProcess:
    cmd = [
        str(BIN), "--metrics", "xpsnr", "--fps", fps,
        "-r", str(ref), "-s", f"{w}x{h}", "-f", fmt,
        "-t", str(test), "-s", f"{w}x{h}", "-f", fmt]
    env = os.environ.copy()
    env["PIXVQA_XPSNR_DEBUG"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)


def parse_ffmpeg(text: str) -> tuple[list[tuple[float, float, float]], tuple]:
    frames = []
    for m in FF_FRAME.finditer(text):
        frames.append((parse_num(m.group(2)), parse_num(m.group(3)), parse_num(m.group(4))))
    avg = FF_AVG.search(text)
    if not avg:
        raise RuntimeError(f"no ffmpeg average in:\n{text[-500:]}")
    summary = (
        int(avg.group(1)),
        parse_num(avg.group(2)),
        parse_num(avg.group(3)),
        parse_num(avg.group(4)),
        parse_num(avg.group(5)),
    )
    return frames, summary


def parse_ours(proc: subprocess.CompletedProcess) -> tuple[list, tuple, list]:
    if proc.returncode != 0:
        raise RuntimeError(f"pixvqa rc={proc.returncode}\n{proc.stderr[-800:]}")
    frame_lines = []
    summary = None
    for line in proc.stdout.splitlines():
        m = FRAME_RE.search(line)
        if not m:
            continue
        vals = tuple(parse_num(m.group(i)) for i in range(1, 6))
        if re.match(r"^\d+\s*:", line):
            frame_lines.append(vals)
        else:
            summary = vals
    wsse = []
    for m in WSSE_RE.finditer(proc.stderr):
        wsse.append(tuple(int(m.group(i)) for i in range(2, 5)))
    if summary is None:
        raise RuntimeError(f"no XPSNR summary in:\n{proc.stdout}")
    return frame_lines, summary, wsse


def expected_avg(wsse: list[tuple[int, int, int]], w: int, h: int, chroma: str, bits: int) -> tuple[list[float], float]:
    cw, ch = chroma_size(w, h, chroma)
    ny, nu, nv = w * h, cw * ch, cw * ch
    n_total = ny + nu + nv
    max_sq = float((1 << bits) - 1) ** 2
    frames = []
    acc = 0.0
    for wy, wu, wv in wsse:
        distortion = (wy + wu + wv) / n_total
        if distortion <= 0:
            frames.append(math.inf)
            acc += 0.0
        else:
            frames.append(10.0 * math.log10(max_sq / distortion))
            acc += math.sqrt(distortion)
    if not wsse:
        return frames, math.inf
    r = acc / len(wsse)
    if r <= 0:
        return frames, math.inf
    return frames, 20.0 * math.log10(((1 << bits) - 1) / r)


def check_case(name: str, w: int, h: int, chroma: str, bits: int, fps: str, ref_frames, test_frames) -> None:
    pix = {8: f"yuv{chroma}p", 10: f"yuv{chroma}p10le"}[bits]
    fmt = f"{chroma}p:{bits}"
    with tempfile.TemporaryDirectory(prefix="xpsnr_") as tmp:
        td = Path(tmp)
        ref = td / "ref.yuv"
        tst = td / "test.yuv"
        stats = td / "stats.txt"
        write_yuv(ref, w, h, chroma, bits, ref_frames)
        write_yuv(tst, w, h, chroma, bits, test_frames)
        try:
            ff_text = run_ffmpeg(ref, tst, w, h, pix, fps, stats)
            ff_frames, ff_sum = parse_ffmpeg(ff_text)
            ours = run_pixvqa(ref, tst, w, h, fmt, fps)
            our_frames, our_sum, wsse = parse_ours(ours)
        except Exception as ex:
            fail(f"{name}: {ex}")
            return
    if len(ff_frames) != len(our_frames) or ff_sum[0] != len(our_frames):
        fail(f"{name}: frame count ffmpeg {len(ff_frames)}/{ff_sum[0]} ours {len(our_frames)}")
        return
    if len(wsse) != len(our_frames):
        fail(f"{name}: wsse lines {len(wsse)} != frames {len(our_frames)}")
        return
    worst = 0.0
    worst_at = ""
    for i, ((fy, fu, fv), (oy, ou, ov, _oa, omin)) in enumerate(zip(ff_frames, our_frames)):
        for comp, a, b in (("y", fy, oy), ("u", fu, ou), ("v", fv, ov)):
            if not same_printed(a, b):
                err = abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
                if err >= worst:
                    worst = err
                    worst_at = f"frame {i} {comp} ffmpeg={a} ours={b}"
        # per-frame min is min of the three plane scores
        expect_min = min(oy, ou, ov)
        if not same_printed(omin, expect_min):
            fail(f"{name}: frame {i} min {omin} != min(y,u,v) {expect_min}")
    for comp, a, b in (("seq y", ff_sum[1], our_sum[0]), ("seq u", ff_sum[2], our_sum[1]),
                       ("seq v", ff_sum[3], our_sum[2]), ("seq min", ff_sum[4], our_sum[4])):
        if not same_printed(a, b):
            err = abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
            if err >= worst:
                worst = err
                worst_at = f"{comp} ffmpeg={a} ours={b}"
    exp_frames, exp_seq = expected_avg(wsse, w, h, chroma, bits)
    for i, (got, exp) in enumerate(zip(our_frames, exp_frames)):
        if not same_printed(got[3], exp):
            fail(f"{name}: frame {i} avg {got[3]} != formula {exp}")
    if not same_printed(our_sum[3], exp_seq):
        fail(f"{name}: sequence avg {our_sum[3]} != SMR {exp_seq}")
    # avg is not the arithmetic mean of the dB plane scores when those differ.
    for i, (oy, ou, ov, oa, _m) in enumerate(our_frames):
        if all(math.isfinite(v) for v in (oy, ou, ov, oa)):
            db_mean = (oy + ou + ov) / 3.0
            if abs(oy - ou) > 0.05 or abs(oy - ov) > 0.05:
                if same_printed(oa, db_mean):
                    fail(f"{name}: frame {i} avg equals the dB mean {db_mean}")
    if worst_at:
        fail(f"{name}: {worst_at}")
    else:
        print(f"OK   {name} frames={len(our_frames)} seq_y={our_sum[0]} avg={our_sum[3]}")


def pattern_pair(w: int, h: int, chroma: str, kind: str, frames: int):
    """Return (reference_frames, reconstructed_frames).

    Activity follows the reference. The reconstructed picture differs except
    for kind == 'identical'.
    """
    cw, ch = chroma_size(w, h, chroma)

    def luma(x: int, yy: int, t: int, distorted: bool) -> int:
        if kind == "identical":
            return 400
        if kind == "pixel":
            return 408 if distorted and x == 3 and yy == 5 else 400
        if kind == "const":
            return 220 if distorted else 400
        if kind == "near":
            return 400 + (1 if distorted and (x + yy) % 17 == 0 else 0)
        if kind == "grad":
            base = (8 * x + 3 * yy + 4 * t) % 900 + 20
            return min(1023, base + (6 if distorted else 0))
        if kind == "check":
            on = ((x // 4 + yy // 4 + t) % 2) == 0
            return (80 if on else 900) if not distorted else (96 if on else 860)
        if kind == "noise":
            n = (x * 131 + yy * 17 + t * 997 + (99 if distorted else 0) + 12345) & 1023
            return 64 + (n % 800)
        if kind == "tex":
            return 200 + ((x * 3) ^ (yy * 5) ^ (t * 9) ^ (17 if distorted else 0)) % 600
        if kind == "flatc":
            return 120 + (x * 5 + yy + (3 if distorted else 0)) % 700
        if kind == "scene":
            base = 100 if t < max(frames // 2, 1) else 800
            return min(1023, base + (15 if distorted else 0))
        if kind == "still":
            return 220 + (x % 16) * 20 + (4 if distorted else 0)
        if kind == "move":
            return 100 + ((x + t * 3) * 2 + yy + (2 if distorted else 0)) % 800
        return 256

    def chroma(x: int, yy: int, t: int, distorted: bool, base: int) -> int:
        if kind in {"identical", "pixel", "near", "flatc", "still"}:
            return base
        if kind == "const":
            return base - 32 if distorted else base
        if kind == "scene":
            return min(1023, base + (40 if t >= max(frames // 2, 1) else 0) + (5 if distorted else 0))
        if kind == "move":
            return base + ((x + t + (1 if distorted else 0)) % 7) - 3
        # Sample the reconstructed/reference luma so chroma is not flat.
        sx = min(w - 1, x * w // cw)
        sy = min(h - 1, yy * h // ch)
        return max(0, min(1023, base + (luma(sx, sy, t, distorted) % 40) - 20))

    def frame(t: int, distorted: bool):
        return (
            fn_plane(w, h, lambda x, yy: luma(x, yy, t, distorted)),
            fn_plane(cw, ch, lambda x, yy: chroma(x, yy, t, distorted, 512)),
            fn_plane(cw, ch, lambda x, yy: chroma(x, yy, t, distorted, 496)),
        )

    return [frame(t, False) for t in range(frames)], [frame(t, True) for t in range(frames)]


def main() -> int:
    if not BIN.exists():
        print("missing", BIN, file=sys.stderr)
        return 2
    # Missing fps must fail closed.
    probe = subprocess.run(
        [str(BIN), "--metrics", "xpsnr", "-r", "a", "-s", "64x64", "-f", "420p:10", "-t", "b", "-s", "64x64", "-f", "420p:10"],
        capture_output=True, text=True,
    )
    if probe.returncode == 0 or "requires --fps" not in probe.stderr:
        fail(f"missing --fps rc={probe.returncode} stderr={probe.stderr.strip()}")
    else:
        print("OK   missing --fps rejected")

    bad = subprocess.run(
        [str(BIN), "--metrics", "xpsnr", "--fps", "0", "-r", "a", "-s", "64x64", "-f", "420p:10", "-t", "b"],
        capture_output=True, text=True,
    )
    if bad.returncode == 0 or "invalid --fps" not in bad.stderr:
        fail(f"invalid --fps rc={bad.returncode} stderr={bad.stderr.strip()}")
    else:
        print("OK   invalid --fps rejected")

    if not FFMPEG:
        print("Set FFMPEG or put ffmpeg on PATH", file=sys.stderr)
        return 2

    kinds = ["identical", "pixel", "const", "near", "grad", "check", "noise", "tex", "flatc", "scene", "still", "move"]
    # Priority and structural cases. Widths are multiples of 32 so FFmpeg's
    # 10-bit linesize matches a tightly packed row.
    configs = [
        ("64x64", 64, 64, "420", 10, "60", 4),
        ("32x32_b1", 32, 32, "420", 10, "60", 2),
        ("320x240_sd", 320, 240, "420", 10, "25", 3),
        ("640x480", 640, 480, "420", 10, "30", 2),
        ("96x80_partial", 96, 80, "420", 10, "24", 2),
        ("64x64_422", 64, 64, "422", 10, "50", 2),
        ("64x64_444", 64, 64, "444", 10, "60", 2),
        ("64x64_8bit", 64, 64, "420", 8, "60", 2),
        ("128x96_8bit_444", 128, 96, "444", 8, "24", 2)]
    for cname, w, h, chroma, bits, fps, n in configs:
        for kind in kinds:
            frames = 1 if kind == "identical" and cname != "64x64" else n
            if kind in {"scene", "move", "still"} and frames < 3:
                frames = 3
            ref, tst = pattern_pair(w, h, chroma, kind, frames)
            check_case(f"{cname}_{kind}_{fps}fps_n{frames}", w, h, chroma, bits, fps, ref, tst)

    # Frame-rate boundary on one moving pair. 31 -> 1st order, 32 -> 2nd.
    ref, tst = pattern_pair(96, 64, "420", "move", 4)
    for fps in ("24", "25", "30", "31", "32", "50", "60000/1001", "59.94", "60"):
        check_case(f"rate_{fps}", 96, 64, "420", 10, fps, ref, tst)

    # One-frame and two-frame explicit cases.
    ref, tst = pattern_pair(64, 64, "420", "noise", 1)
    check_case("one_frame", 64, 64, "420", 10, "60", ref, tst)
    ref, tst = pattern_pair(64, 64, "420", "noise", 2)
    check_case("two_frames", 64, 64, "420", 10, "60", ref, tst)

    # HD uses the non-downsampled high-pass. A short clip is enough.
    ref, tst = pattern_pair(1920, 1080, "420", "move", 2)
    check_case("hd1080_move", 1920, 1080, "420", 10, "60", ref, tst)
    ref, tst = pattern_pair(1920, 1080, "420", "pixel", 2)
    check_case("hd1080_pixel", 1920, 1080, "420", 10, "50", ref, tst)

    # UHD priority workload, few frames, several contents.
    for kind, fps, n in (("move", "60", 2), ("noise", "60", 2), ("pixel", "60", 1), ("scene", "60000/1001", 3)):
        ref, tst = pattern_pair(3840, 2160, "420", kind, n)
        check_case(f"uhd4k_{kind}", 3840, 2160, "420", 10, fps, ref, tst)

    if FAILS:
        print(f"{FAILS} XPSNR checks failed", file=sys.stderr)
        return 1
    print("All XPSNR FFmpeg checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
