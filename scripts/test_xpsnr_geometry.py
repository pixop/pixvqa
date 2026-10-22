#!/usr/bin/env python3
"""High-resolution XPSNR geometry: reject odd edges, match FFmpeg on safe neighbors.

Unsafe pictures must exit 2 with no score summary. Safe pictures, including
even sizes just above 2048*1152 and odd sizes at or below that threshold,
must match FFmpeg Y/U/V/min and the sample-count avg formula.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_xpsnr_ffmpeg as xf

BIN = Path(os.environ.get("PIXVQA", os.environ.get("PIXVQA", str(xf.BIN))))
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def one_pixel(w: int, h: int, chroma: str):
    ref, tst = xf.pattern_pair(w, h, chroma, "pixel", 1)
    return ref, tst


def reject(name: str, w: int, h: int, fmt: str) -> None:
    with tempfile.TemporaryDirectory(prefix="xpsnr_rej_") as tmp:
        td = Path(tmp)
        chroma = fmt[:3]
        bits = int(fmt.split(":")[1])
        ref_frames, test_frames = one_pixel(w, h, chroma)
        ref = td / "ref.yuv"
        tst = td / "test.yuv"
        xf.write_yuv(ref, w, h, chroma, bits, ref_frames)
        xf.write_yuv(tst, w, h, chroma, bits, test_frames)
        cmd = [
            str(BIN), "--metrics", "xpsnr", "--fps", "60",
            "-r", str(ref), "-s", f"{w}x{h}", "-f", fmt,
            "-t", str(tst), "-s", f"{w}x{h}", "-f", fmt]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    blob = proc.stdout + proc.stderr
    ok = proc.returncode != 0 and "Summary results" not in proc.stdout
    ok = ok and "XPSNR y:" not in proc.stdout
    ok = ok and "AddressSanitizer" not in blob
    if not ok:
        fail(f"{name}: rc={proc.returncode}\n{proc.stdout[-300:]}\n{proc.stderr[-400:]}")
    else:
        print(f"OK   reject {name} rc={proc.returncode}")


def psnr_without_xpsnr(w: int, h: int) -> None:
    with tempfile.TemporaryDirectory(prefix="xpsnr_psnr_") as tmp:
        td = Path(tmp)
        ref_frames, test_frames = one_pixel(w, h, "444")
        ref = td / "ref.yuv"
        tst = td / "test.yuv"
        xf.write_yuv(ref, w, h, "444", 10, ref_frames)
        xf.write_yuv(tst, w, h, "444", 10, test_frames)
        cmd = [
            str(BIN), "--metrics", "psnr",
            "-r", str(ref), "-s", f"{w}x{h}", "-f", "444p:10",
            "-t", str(tst), "-s", f"{w}x{h}", "-f", "444p:10"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ok = proc.returncode == 0 and "Summary results" in proc.stdout and "[ PSNR" in proc.stdout
    ok = ok and "XPSNR" not in proc.stdout
    if not ok:
        fail(f"psnr-only {w}x{h}: rc={proc.returncode}\n{proc.stdout[-300:]}")
    else:
        print(f"OK   psnr-only {w}x{h}")


def threads_match(w: int, h: int, chroma: str, bits: int) -> None:
    fmt = f"{chroma}p:{bits}"
    with tempfile.TemporaryDirectory(prefix="xpsnr_thr_") as tmp:
        td = Path(tmp)
        ref_frames, test_frames = xf.pattern_pair(w, h, chroma, "grad", 2)
        ref = td / "ref.yuv"
        tst = td / "test.yuv"
        xf.write_yuv(ref, w, h, chroma, bits, ref_frames)
        xf.write_yuv(tst, w, h, chroma, bits, test_frames)
        outs = []
        for threads in (1, 8):
            cmd = [
                str(BIN), "--metrics", "xpsnr", "--fps", "60", "--threads", str(threads),
                "-r", str(ref), "-s", f"{w}x{h}", "-f", fmt,
                "-t", str(tst), "-s", f"{w}x{h}", "-f", fmt]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if proc.returncode != 0 or "AddressSanitizer" in proc.stderr:
                fail(f"threads {threads} {w}x{h}: rc={proc.returncode}\n{proc.stderr[-400:]}")
                return
            lines = [ln for ln in proc.stdout.splitlines() if "XPSNR y:" in ln]
            outs.append("\n".join(lines))
    if outs[0] != outs[1] or not outs[0]:
        fail(f"thread mismatch {w}x{h}\n{outs[0]}\n{outs[1]}")
    else:
        print(f"OK   threads {w}x{h}")


def main() -> int:
    saved = xf.BIN
    xf.BIN = BIN
    try:
        reject("1600x1601-444", 1600, 1601, "444p:10")
        reject("1601x1600-444", 1601, 1600, "444p:10")
        reject("1601x1601-444", 1601, 1601, "444p:10")
        reject("2048x1153-444", 2048, 1153, "444p:10")
        reject("2049x1152-444", 2049, 1152, "444p:10")
        reject("1600x1601-422", 1600, 1601, "422p:10")
        psnr_without_xpsnr(1600, 1601)

        safe = [
            (1600, 1600, "444", 10, "even-highres"),
            (1600, 1602, "444", 10, "even-highres-partial"),
            (2048, 1152, "444", 10, "at-threshold-per-pixel"),
            (2048, 1154, "420", 10, "just-over-even-420"),
            (1921, 1080, "444", 10, "odd-below-threshold"),
            (1920, 1081, "422", 10, "odd-height-422-hd"),
            (641, 480, "444", 10, "odd-lowres-smoother")]
        for w, h, chroma, bits, name in safe:
            ref, tst = one_pixel(w, h, chroma)
            xf.check_case(name, w, h, chroma, bits, "60", ref, tst)
        threads_match(1600, 1600, "444", 10)
        threads_match(1921, 1080, "444", 10)
    finally:
        xf.BIN = saved
    total = FAILS + xf.FAILS
    print(f"geometry failures: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
