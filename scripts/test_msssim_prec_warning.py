#!/usr/bin/env python3
"""CUDA MS-SSIM precision warning: once, stderr only, absent for fp64 and when MS-SSIM is off."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "build" / "cpu" / "pixvqa"
CUDA = ROOT / "build" / "cuda" / "pixvqa"
FAILS = 0
NEEDLES = (
    "validated MS-SSIM precision",
    "flat or nearly flat",
)


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def write_pair(directory: Path, frames: int) -> tuple[Path, Path]:
    h = w = 32
    ch = cw = 16
    y = np.full((h, w), 400, dtype=np.uint16)
    chroma = np.full((ch, cw), 512, dtype=np.uint16)
    blob = y.tobytes() + chroma.tobytes() + chroma.tobytes()
    ref = directory / "ref.yuv"
    tst = directory / "tst.yuv"
    ref.write_bytes(blob * frames)
    y2 = y.copy()
    y2[8, 8] = 460
    tst.write_bytes((y2.tobytes() + chroma.tobytes() + chroma.tobytes()) * frames)
    return ref, tst


def run(binary: Path, args: list[str], prec: str | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if prec:
        env["PIXVQA_CUDA_PREC"] = prec
    return subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=60, env=env)


def score_ok(stdout: str) -> bool:
    return "[ MS-SSIM  y:" in stdout and "warning:" not in stdout


def main() -> int:
    if not CUDA.is_file() or not CPU.is_file():
        print("missing binaries", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        ref, tst = write_pair(Path(tmp), 2)
        common = [
            "-r", str(ref), "-s", "32x32", "-f", "420p:10",
            "-t", str(tst), "-s", "32x32", "-f", "420p:10",
            "-n", "2", "-v", "1"]
        cases = (
            ("public-fp64", ["--metrics", "msssim", "--metric-device", "gpu"], None, False, None),
            ("fp32", ["--metrics", "msssim", "--metric-device", "gpu"], "fp32", False, None),
            ("mixed", ["--metrics", "msssim", "--metric-device", "gpu"], "mixed", False, None),
            ("fp64", ["--metrics", "msssim", "--metric-device", "gpu"], "fp64", False, None),
            ("psnr-only", ["--metrics", "psnr", "--metric-device", "gpu"], None, False, None),
            ("psnr-fp32", ["--metrics", "psnr", "--metric-device", "gpu"], "fp32", False, None),
        )
        for name, extra, prec, expect, phrase in cases:
            done = run(CUDA, extra + common, prec)
            if done.returncode != 0:
                fail(f"{name} rc {done.returncode} {done.stderr}")
                continue
            count = done.stderr.count("pixvqa: warning: GPU MS-SSIM")
            if expect:
                if count != 1:
                    fail(f"{name} warning count {count}\n{done.stderr}")
                    continue
                if phrase not in done.stderr:
                    fail(f"{name} missing {phrase!r}\n{done.stderr}")
                    continue
                for needle in NEEDLES:
                    if needle not in done.stderr:
                        fail(f"{name} missing {needle!r}")
                if not score_ok(done.stdout):
                    fail(f"{name} stdout changed\n{done.stdout}")
                    continue
            else:
                if count != 0 or "warning:" in done.stderr:
                    fail(f"{name} unexpected warning\n{done.stderr}")
                    continue
            print(f"OK   {name}")
        cpu = run(CPU, ["--metrics", "msssim", *common])
        if cpu.returncode != 0 or "warning:" in cpu.stderr or not score_ok(cpu.stdout):
            fail(f"cpu MS-SSIM warning or scores\n{cpu.stderr}\n{cpu.stdout}")
        else:
            print("OK   cpu-msssim")
        # Score lines stay in the existing six-decimal form when the warning is on.
        warned = run(CUDA, ["--metrics", "msssim,psnr", "--metric-device", "gpu", *common], "fp32")
        quiet = run(CUDA, ["--metrics", "msssim,psnr", "--metric-device", "gpu", *common], "fp64")
        if warned.returncode != 0 or quiet.returncode != 0:
            fail("score-format rc")
        else:
            score_re = re.compile(
                r"\[ MS-SSIM  y:\d\.\d{6} u:\d\.\d{6} v:\d\.\d{6} avg:\d\.\d{6} \]"
            )
            psnr_w = [re.sub(r"\[ MS-SSIM[^\]]*\] ?", "", ln)
                      for ln in warned.stdout.splitlines() if "[ PSNR " in ln]
            psnr_q = [re.sub(r"\[ MS-SSIM[^\]]*\] ?", "", ln)
                      for ln in quiet.stdout.splitlines() if "[ PSNR " in ln]
            scores = score_re.findall(warned.stdout)
            if psnr_w != psnr_q:
                fail(f"PSNR text changed beside the warning\n{psnr_w}\n{psnr_q}")
            elif len(scores) != 3 or "warning" in warned.stdout:
                fail(f"MS-SSIM format {scores}")
            else:
                print("OK   score-format")
    if FAILS:
        print(f"{FAILS} failed", file=sys.stderr)
        return 1
    print("All MS-SSIM precision-warning checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
