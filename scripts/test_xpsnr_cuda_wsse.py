#!/usr/bin/env python3
"""Exact per-frame WSSE: CUDA XPSNR versus CPU XPSNR.

Printed Y/U/V/avg/min must match as well. A 4-decimal match with a different
WSSE integer is a failure.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_xpsnr_ffmpeg as xf

CPU = Path(__file__).resolve().parents[1] / "build/cpu/pixvqa"
CUDA = Path(__file__).resolve().parents[1] / "build/cuda/pixvqa"
WSSE = re.compile(r"xpsnr_wsse (\d+) (\d+) (\d+) (\d+)")
FRAME = re.compile(r"\[ XPSNR y:(\S+) u:(\S+) v:(\S+) avg:(\S+) min:(\S+) \]")
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def run(bin_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PIXVQA_XPSNR_DEBUG"] = "1"
    return subprocess.run([str(bin_path), *args], capture_output=True, text=True, timeout=600, env=env)


def wsse_of(proc: subprocess.CompletedProcess) -> list[tuple[int, int, int, int]]:
    return [tuple(int(g) for g in m.groups()) for m in WSSE.finditer(proc.stderr)]


def scores_of(text: str) -> list[tuple[str, ...]]:
    return [m.groups() for m in FRAME.finditer(text)]


def compare_pair(name: str, w: int, h: int, chroma: str, bits: int, fps: str, n: int, kind: str,
                 backend_cuda: bool, also_metrics: bool = False) -> None:
    fmt = f"{chroma}p:{bits}"
    ref_frames, test_frames = xf.pattern_pair(w, h, chroma, kind, n)
    with tempfile.TemporaryDirectory(prefix="xpsnr_cu_") as tmp:
        td = Path(tmp)
        ref = td / "ref.yuv"
        tst = td / "test.yuv"
        xf.write_yuv(ref, w, h, chroma, bits, ref_frames)
        xf.write_yuv(tst, w, h, chroma, bits, test_frames)
        base = ["--metrics", "xpsnr", "--fps", fps, "-r", str(ref), "-s", f"{w}x{h}", "-f", fmt,
                "-t", str(tst), "-s", f"{w}x{h}", "-f", fmt]
        if also_metrics:
            base = ["--metrics", "psnr,ssim", *base]
        cpu = run(CPU, ["--threads", "1", *base])
        extra = ["--metric-device", "gpu" if backend_cuda else "cpu", *base]
        gpu = run(CUDA, extra)
    if cpu.returncode != 0:
        fail(f"{name}: cpu rc={cpu.returncode} {cpu.stderr[-300:]}")
        return
    if gpu.returncode != 0:
        fail(f"{name}: cuda rc={gpu.returncode} {gpu.stderr[-400:]}")
        return
    cw, gw = wsse_of(cpu), wsse_of(gpu)
    if cw != gw:
        fail(f"{name}: WSSE mismatch\n cpu={cw[:4]}\n gpu={gw[:4]}")
        return
    if scores_of(cpu.stdout) != scores_of(gpu.stdout):
        fail(f"{name}: score text mismatch\n{scores_of(cpu.stdout)}\n{scores_of(gpu.stdout)}")
        return
    if "Summary results" not in gpu.stdout:
        fail(f"{name}: missing summary")
        return
    print(f"OK   {name} frames={len(cw)} wsse0={cw[0][1:] if cw else None}")


def expect_reject(name: str, args: list[str], bin_path: Path) -> None:
    proc = run(bin_path, args)
    blob = proc.stdout + proc.stderr
    ok = proc.returncode != 0 and "Summary results" not in proc.stdout and "XPSNR y:" not in proc.stdout
    ok = ok and "AddressSanitizer" not in blob
    if not ok:
        fail(f"{name}: rc={proc.returncode}\n{proc.stdout[-200:]}\n{proc.stderr[-300:]}")
    else:
        print(f"OK   reject {name} rc={proc.returncode}")


def main() -> int:
    if not CUDA.exists():
        print("missing", CUDA, file=sys.stderr)
        return 2
    # CPU binary must refuse the CUDA backend.
    expect_reject(
        "cpu-build",
        ["--metrics", "xpsnr", "--fps", "60",  "-r", "a", "-s", "64x64", "-f", "420p:10",
         "-t", "b"],
        CPU,
    )
    expect_reject(
        "backend-without-kernel",
        ["--metrics", "xpsnr", "--fps", "60",  "-r", "a", "-s", "64x64", "-f", "420p:10",
         "-t", "b"],
        CUDA,
    )

    cases = [
        ("64_pixel", 64, 64, "420", 10, "60", 4, "pixel"),
        ("64_noise", 64, 64, "420", 10, "60", 3, "noise"),
        ("32_b1", 32, 32, "420", 10, "60", 2, "grad"),
        ("640_smooth", 640, 480, "420", 10, "30", 2, "tex"),
        ("641_nosmooth", 641, 480, "444", 10, "30", 2, "tex"),
        ("hd_move", 1920, 1080, "420", 10, "50", 2, "move"),
        ("hd_pixel", 1920, 1080, "420", 10, "31", 3, "pixel"),
        ("rate32", 96, 64, "420", 10, "32", 4, "move"),
        ("rate31", 96, 64, "420", 10, "31", 4, "move"),
        ("uhd_pixel", 3840, 2160, "420", 10, "60", 2, "pixel"),
        ("uhd_move", 3840, 2160, "420", 10, "60", 2, "move"),
        ("uhd_noise", 3840, 2160, "420", 10, "60", 2, "noise"),
        ("uhd_scene", 3840, 2160, "420", 10, "60", 3, "scene"),
        ("uhd_ident", 3840, 2160, "420", 10, "60", 2, "identical"),
        ("c422", 64, 64, "422", 10, "60", 2, "grad"),
        ("c444", 128, 96, "444", 8, "24", 2, "check"),
        ("odd444", 1921, 1080, "444", 10, "60", 1, "pixel"),
        ("edge_hr", 2048, 1154, "420", 10, "60", 1, "grad"),
        ("bit8_420", 320, 240, "420", 8, "25", 3, "noise"),
        ("rate5994", 96, 64, "420", 10, "59.94", 4, "move"),
        ("chroma_only", 64, 64, "420", 10, "60", 2, "const"),
        ("near_sd", 320, 240, "422", 10, "24", 2, "near"),
        ("still4", 128, 96, "444", 10, "60", 4, "still")]
    only = [s for s in os.environ.get("XPSNR_CUDA_ONLY", "").split(",") if s]
    for name, w, h, chroma, bits, fps, n, kind in cases:
        if only and not any(s in name for s in only):
            continue
        compare_pair(name, w, h, chroma, bits, fps, n, kind, True)
        if FAILS:
            print(f"{FAILS} CUDA XPSNR checks failed", file=sys.stderr)
            return 1

    if only:
        if FAILS:
            print(f"{FAILS} CUDA XPSNR checks failed", file=sys.stderr)
            return 1
        print("All requested CUDA XPSNR WSSE checks passed")
        return 0

    # Default backend on the CUDA binary stays on the CPU and matches.
    compare_pair("default_backend_uhd", 3840, 2160, "420", 10, "60", 1, "pixel", False)
    # One upload: PSNR + MS-SSIM + CUDA XPSNR versus CPU XPSNR WSSE.
    compare_pair("unified_uhd", 3840, 2160, "420", 10, "60", 1, "noise", True, also_metrics=True)

    if FAILS:
        print(f"{FAILS} CUDA XPSNR checks failed", file=sys.stderr)
        return 1
    print("All CUDA XPSNR WSSE checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
