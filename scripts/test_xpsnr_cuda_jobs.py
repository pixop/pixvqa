#!/usr/bin/env python3
"""48-frame job0/job1: CUDA XPSNR WSSE equals CPU, and PSNR/SSIM text is unchanged."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

CPU = Path(__file__).resolve().parents[1] / "build/cpu/pixvqa"
CUDA = Path(__file__).resolve().parents[1] / "build/cuda/pixvqa"
CACHE = Path("/tmp/pixvqa_p4")
WSSE = re.compile(r"xpsnr_wsse (\d+) (\d+) (\d+) (\d+)")
XLINE = re.compile(r"\[ XPSNR y:(\S+) u:(\S+) v:(\S+) avg:(\S+) min:(\S+) \]")
PLINE = re.compile(r"\[ PSNR  y:\S+ u:\S+ v:\S+ avg:\S+ \] \[ MSE  y:\S+ u:\S+ v:\S+ avg:\S+ \]")
MLINE = re.compile(r"\[ SSIM  y:\S+ u:\S+ v:\S+ avg:\S+ \]")


def run(bin_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PIXVQA_XPSNR_DEBUG"] = "1"
    return subprocess.run([str(bin_path), *args], capture_output=True, text=True, timeout=900, env=env)


def main() -> int:
    fails = 0
    for job in (0, 1):
        ref = CACHE / f"job{job}_ref_48.yuv"
        tst = CACHE / f"job{job}_48.yuv"
        if not ref.exists() or not tst.exists():
            print(f"SKIP job{job}: decoded YUV missing")
            return 2
        base = ["--metrics", "xpsnr", "--fps", "60", "-r", str(ref), "-s", "3840x2160", "-f", "420p:10",
                "-t", str(tst), "-s", "3840x2160", "-f", "420p:10"]
        cpu = run(CPU, ["--threads", "8", *base])
        gpu = run(CUDA, ["--metric-device", "gpu",  *base])
        gpu2 = run(CUDA, ["--metric-device", "gpu",  *base])
        if cpu.returncode or gpu.returncode or gpu2.returncode:
            print(f"FAIL job{job} rc cpu={cpu.returncode} gpu={gpu.returncode}", file=sys.stderr)
            return 1
        cw = WSSE.findall(cpu.stderr)
        gw = WSSE.findall(gpu.stderr)
        if cw != gw or cw != WSSE.findall(gpu2.stderr) or len(cw) != 48:
            print(f"FAIL job{job} wsse count cpu={len(cw)} gpu={len(gw)}", file=sys.stderr)
            fails += 1
        elif XLINE.findall(cpu.stdout) != XLINE.findall(gpu.stdout):
            print(f"FAIL job{job} XPSNR text", file=sys.stderr)
            fails += 1
        else:
            seq = XLINE.findall(cpu.stdout)[-1]
            print(f"OK   job{job} frames=48 y={seq[0]} u={seq[1]} v={seq[2]} avg={seq[3]} min={seq[4]}")

        geom = base[4:]
        both_cpu = run(CUDA, ["--metric-device", "gpu", "--metrics", "psnr,ssim", *geom])
        both_gpu = run(CUDA, ["--metric-device", "gpu", "--metrics", "psnr,ssim,xpsnr", "--fps", "60", *geom])
        if both_cpu.returncode or both_gpu.returncode:
            print(f"FAIL job{job} unified rc", file=sys.stderr)
            return 1
        if PLINE.findall(both_cpu.stdout) != PLINE.findall(both_gpu.stdout):
            print(f"FAIL job{job} PSNR changed when CUDA XPSNR was enabled", file=sys.stderr)
            fails += 1
        elif MLINE.findall(both_cpu.stdout) != MLINE.findall(both_gpu.stdout):
            print(f"FAIL job{job} SSIM changed when CUDA XPSNR was enabled", file=sys.stderr)
            fails += 1
        elif XLINE.findall(both_gpu.stdout) != XLINE.findall(cpu.stdout):
            print(f"FAIL job{job} unified XPSNR != CPU XPSNR", file=sys.stderr)
            fails += 1
        elif WSSE.findall(both_gpu.stderr) != cw:
            print(f"FAIL job{job} unified WSSE != CPU", file=sys.stderr)
            fails += 1
        else:
            print(f"OK   job{job} psnr+ssim unchanged, one-upload XPSNR matches CPU")

    # Two processes, same file, identical WSSE.
    ref = CACHE / "job0_ref_48.yuv"
    tst = CACHE / "job0_48.yuv"
    args = ["--metric-device", "gpu",  "--metrics", "xpsnr", "--fps", "60", "-n", "4",
            "-r", str(ref), "-s", "3840x2160", "-f", "420p:10",
            "-t", str(tst), "-s", "3840x2160", "-f", "420p:10"]
    a = subprocess.Popen([str(CUDA), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env={**os.environ, "PIXVQA_XPSNR_DEBUG": "1"})
    b = subprocess.Popen([str(CUDA), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env={**os.environ, "PIXVQA_XPSNR_DEBUG": "1"})
    ao, ae = a.communicate(timeout=300)
    bo, be = b.communicate(timeout=300)
    if a.returncode or b.returncode or WSSE.findall(ae) != WSSE.findall(be) or not WSSE.findall(ae):
        print("FAIL two-process WSSE", file=sys.stderr)
        fails += 1
    else:
        print(f"OK   two-process frames={len(WSSE.findall(ae))}")

    odd = ["--metric-device", "gpu",  "--metrics", "xpsnr", "--fps", "60",
           "-r", str(ref), "-s", "1600x1601", "-f", "444p:10",
           "-t", str(ref), "-s", "1600x1601", "-f", "444p:10"]
    # The cached file is 4K; the size flag is what the reader uses, so this would
    # mis-parse pixels if it ran. Rejection must happen before any read.
    rej = run(CUDA, odd)
    if rej.returncode == 0 or "Summary results" in rej.stdout or "XPSNR y:" in rej.stdout:
        print(f"FAIL odd geometry rc={rej.returncode}", file=sys.stderr)
        fails += 1
    else:
        print(f"OK   reject 1600x1601 cuda backend rc={rej.returncode}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
