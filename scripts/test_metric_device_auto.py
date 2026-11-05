#!/usr/bin/env python3
"""--device auto: CUDA per metric when supported, CPU otherwise, VMAF on CPU.

The CPU-build case always runs. The CUDA cases need build/full/pixvqa and a
visible CUDA device of compute capability 8.9 or later. Without them the CUDA
cases are reported as skipped, not passed.
"""
from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
FULL = Path(os.environ.get("PIXVQA_FULL", ROOT / "build" / "full" / "pixvqa"))
HDR_MODEL = ROOT / "model" / "hdr_vmaf_v1.1.8" / "hdr_vmaf_v1.1.8_3d0h.json"
HDR = ["--hdr-transfer", "pq", "--hdr-primaries", "bt2020", "--hdr-matrix", "bt2020nc", "--hdr-range", "limited"]
WARN = "VMAF CUDA computation is unavailable"
CUDA_METRICS = ("PSNR", "XPSNR", "SSIM", "MS-SSIM", "PU21-PSNR", "PU21-SSIM", "PU21-MS-SSIM")
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def run(binary: Path, args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary), *args], capture_output=True, text=True, env=env, timeout=600)


def write_yuv(path: Path, w: int, h: int, chroma: str, frames: int, offset: int) -> None:
    cw = w if chroma == "444" else w // 2
    ch = h if chroma != "420" else h // 2
    with path.open("wb") as f:
        for i in range(frames):
            y = [min(940, 64 + ((x * 7 + row * 3 + i * 11 + offset) % 860)) for row in range(h) for x in range(w)]
            u = [480 + ((x + row + offset) % 64) for row in range(ch) for x in range(cw)]
            v = [500 + ((x * 2 + row + i) % 48) for row in range(ch) for x in range(cw)]
            for plane in (y, u, v):
                f.write(struct.pack("<" + "H" * len(plane), *plane))


def devices(stdout: str) -> dict[str, str]:
    """Execution-summary metric lines: name -> rest of the line."""
    out = {}
    for line in stdout.splitlines():
        match = re.match(r"^  ([A-Z0-9-]+): (cpu|gpu)(.*)$", line)
        if match and match.group(1) in (*CUDA_METRICS, "VMAF"):
            out[match.group(1)] = match.group(2) + match.group(3)
    return out


def scores(stdout: str) -> list[str]:
    """Frame and summary score lines, without the execution summary."""
    return [line for line in stdout.splitlines() if "[ " in line]


def sides(ref: Path, dist: Path, size: str, fmt: str) -> list[str]:
    return ["-r", str(ref), "-s", size, "-f", fmt, "-t", str(dist), "-s", size, "-f", fmt]


def expect_devices(label: str, proc: subprocess.CompletedProcess[str], want: dict[str, str]) -> None:
    if proc.returncode != 0:
        fail(f"{label}: rc={proc.returncode}\n{proc.stderr[-400:]}")
        return
    got = devices(proc.stdout)
    for name, prefix in want.items():
        if not got.get(name, "").startswith(prefix):
            fail(f"{label}: {name} expected '{prefix}...', got '{got.get(name)}'\n{proc.stdout[:900]}")
            return
    if "requested device: auto" not in proc.stdout:
        fail(f"{label}: execution summary does not report the auto request")
        return
    print(f"OK   {label}: " + ", ".join(f"{k}={got[k]}" for k in want))


def check_cpu_build(td: Path) -> None:
    ref, dist = td / "c_r.yuv", td / "c_d.yuv"
    write_yuv(ref, 64, 48, "420", 2, 0)
    write_yuv(dist, 64, 48, "420", 2, 9)
    proc = run(CPU, ["--metrics", "psnr,ssim,pu21-psnr", "--hdr-transfer", "pq", "-v", "0",
                     *sides(ref, dist, "64x48", "420p:10")])
    reason = "cpu (auto: this build has no GPU metrics)"
    expect_devices("cpu build auto", proc, {"PSNR": "cpu", "SSIM": "cpu", "PU21-PSNR": reason})


def check_cuda(td: Path) -> None:
    ref, dist = td / "r.yuv", td / "d.yuv"
    write_yuv(ref, 320, 240, "420", 3, 0)
    write_yuv(dist, 320, 240, "420", 3, 13)
    every = ["--metrics", "psnr,xpsnr,ssim,msssim,pu21-psnr,pu21-ssim,pu21-msssim", "--fps", "60", *HDR, "-v", "0",
             *sides(ref, dist, "320x240", "420p:10")]
    auto = run(FULL, every)
    gpu = run(FULL, ["--device", "gpu", *every])
    cpu = run(FULL, ["--device", "cpu", *every])
    expect_devices("auto prefers CUDA for every supported metric", auto,
                   {name: "gpu" for name in CUDA_METRICS})
    if auto.returncode == 0 and gpu.returncode == 0:
        if scores(auto.stdout) != scores(gpu.stdout) or not scores(auto.stdout):
            fail("auto scores differ from --device gpu scores")
        else:
            print("OK   auto scores equal --device gpu scores")
    if WARN in auto.stderr:
        fail("auto printed the VMAF warning without VMAF")

    hidden = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    nodev = run(FULL, every, env=hidden)
    expect_devices("auto with no visible CUDA device falls back to CPU", nodev,
                   {name: "cpu" for name in CUDA_METRICS})
    if nodev.returncode == 0:
        if "(auto: no usable CUDA device:" not in nodev.stdout:
            fail(f"no-device fallback reason missing\n{nodev.stdout[:900]}")
        elif scores(nodev.stdout) != scores(cpu.stdout):
            fail("no-device auto scores differ from --device cpu scores")
        else:
            print("OK   no-device auto reports the reason and matches --device cpu")
    explicit = run(FULL, ["--device", "gpu", *every], env=hidden)
    if explicit.returncode == 0:
        fail("--device gpu with no visible device did not fail")
    else:
        print("OK   --device gpu with no visible device still fails")

    if HDR_MODEL.is_file():
        with_vmaf = run(FULL, ["--metrics", "psnr,pu21-psnr,vmaf", "--model", str(HDR_MODEL), *HDR, "-v", "0",
                               *sides(ref, dist, "320x240", "420p:10")])
        expect_devices("auto keeps VMAF on CPU beside CUDA metrics", with_vmaf,
                       {"PSNR": "gpu", "PU21-PSNR": "gpu", "VMAF": "cpu libvmaf"})
        if with_vmaf.returncode == 0 and WARN in with_vmaf.stderr:
            fail("auto printed the VMAF CUDA warning")
    else:
        print("SKIP auto with VMAF: HDR model absent")

    odd_r, odd_d = td / "o_r.yuv", td / "o_d.yuv"
    write_yuv(odd_r, 45, 32, "422", 2, 0)
    write_yuv(odd_d, 45, 32, "422", 2, 5)
    odd = run(FULL, ["--metrics", "psnr,pu21-psnr", "--hdr-transfer", "pq", "-v", "0",
                     *sides(odd_r, odd_d, "45x32", "422p:10")])
    expect_devices("odd-width 4:2:2 keeps PU21 on CPU", odd,
                   {"PSNR": "gpu", "PU21-PSNR": "cpu (auto: GPU PU21 needs even chroma geometry)"})
    if odd.returncode == 0:
        odd_cpu = run(FULL, ["--device", "cpu", "--metrics", "pu21-psnr", "--hdr-transfer", "pq", "-v", "0",
                             *sides(odd_r, odd_d, "45x32", "422p:10")])
        pu_auto = [line for line in scores(odd.stdout) if "PU21" in line]
        pu_cpu = [line for line in scores(odd_cpu.stdout) if "PU21" in line]
        if not pu_auto or pu_auto != pu_cpu:
            fail("odd-width auto PU21 differs from --device cpu")
        else:
            print("OK   odd-width auto PU21 equals --device cpu")


def main() -> int:
    if not CPU.is_file():
        print(f"missing {CPU}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="pixvqa_device_auto_") as td:
        root = Path(td)
        check_cpu_build(root)
        if not FULL.is_file():
            print("SKIP CUDA auto cases: build/full/pixvqa is absent")
        else:
            probe = run(FULL, ["--version"])
            info = subprocess.run([str(FULL)], capture_output=True, text=True,
                                  env=dict(os.environ, PIXVQA_CUDA_INFO="1"))
            if "build: cuda" not in probe.stdout or info.returncode != 0:
                print("SKIP CUDA auto cases: no CUDA build or no usable device")
            else:
                check_cuda(root)
    if FAILS:
        print(f"{FAILS} failure(s)", file=sys.stderr)
        return 1
    print("PASS metric-device auto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
