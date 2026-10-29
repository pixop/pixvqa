#!/usr/bin/env python3
"""PixVQA log-structure checks for PSNR, SSIM, and MS-SSIM.

These checks use synthetic YUV generated at runtime. They do not compare
against an external tool.
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
BIN = ROOT / "build" / "cpu" / "pixvqa"
W = H = 64

MS = re.compile(
    r"\[ MS-SSIM  y:(-?[0-9.]+) u:(-?[0-9.]+) v:(-?[0-9.]+) avg:(-?[0-9.]+) \] "
)
SSIM = re.compile(
    r"\[ SSIM  y:(-?[0-9.]+) u:(-?[0-9.]+) v:(-?[0-9.]+) avg:(-?[0-9.]+) \] "
)
PSNR = re.compile(
    r"\[ PSNR  y:([0-9.]+) u:([0-9.]+) v:([0-9.]+) avg:([0-9.]+) \] "
)


def fail(msg: str) -> int:
    print("FAIL", msg, file=sys.stderr)
    return 1


def write_frame(path: Path, y: int = 512, u: int = 512, v: int = 512, dy: int = 0) -> None:
    ys = [max(0, min(1023, y + dy))] * (W * H)
    us = [u] * ((W // 2) * (H // 2))
    vs = [v] * ((W // 2) * (H // 2))
    path.write_bytes(struct.pack("<" + "H" * len(ys + us + vs), *(ys + us + vs)))


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN), *args], capture_output=True, text=True, timeout=60)


def main() -> int:
    if not BIN.exists():
        return fail(f"missing {BIN}")
    rc = 0
    with tempfile.TemporaryDirectory(prefix="pixvqa_logs_") as td:
        tmp = Path(td)
        ref = tmp / "ref.yuv"
        test = tmp / "test.yuv"
        write_frame(ref, y=400, u=500, v=600)
        write_frame(test, y=400, u=500, v=600)
        base = ["-r", str(ref), "-t", str(test), "-s", f"{W}x{H}", "-f", "420p:10", "-n", "1", "-v", "1"]

        banner = run(["--version"])
        if banner.returncode != 0 or not banner.stdout.startswith("pixvqa version "):
            rc |= fail(f"version line: {banner.stdout!r}")
        else:
            print("OK   version")

        both = run(base + ["--metrics", "psnr,ssim,msssim"])
        if both.returncode != 0:
            rc |= fail(f"combined rc={both.returncode} {both.stderr[-200:]}")
        elif "Pixop VQA" not in both.stdout:
            rc |= fail("banner missing")
        elif not PSNR.search(both.stdout) or not SSIM.search(both.stdout) or not MS.search(both.stdout):
            rc |= fail("combined log missing PSNR, SSIM, or MS-SSIM")
        elif "999.99" not in both.stdout:
            rc |= fail("identical PSNR was not 999.99")
        else:
            got = [float(x) for x in MS.findall(both.stdout)[-1]]
            if any(abs(v - 1.0) > 5.1e-7 for v in got):
                rc |= fail(f"identical MS-SSIM was not 1: {got}")
            else:
                print("OK   combined identical log")

        only_ssim = run(base + ["--metrics", "ssim"])
        only_ms = run(base + ["--metrics", "msssim"])
        if only_ssim.returncode or only_ms.returncode:
            rc |= fail("single-metric rc")
        elif MS.search(only_ssim.stdout) or not SSIM.search(only_ssim.stdout):
            rc |= fail("--ssim selected MS-SSIM or missed SSIM")
        elif SSIM.search(only_ms.stdout) or not MS.search(only_ms.stdout):
            rc |= fail("--msssim selected SSIM or missed MS-SSIM")
        else:
            print("OK   ssim and msssim select one metric each")

        bad = run(["--not-a-real-flag", *base, "--metrics", "psnr"])
        if bad.returncode == 0 or "PARSE ERROR" not in bad.stderr:
            rc |= fail(f"unsupported arg: rc={bad.returncode} err={bad.stderr!r}")
        else:
            print("OK   unsupported-arg")

        empty = tmp / "empty.yuv"
        empty.write_bytes(b"")
        emp = run(["-r", str(empty), "-t", str(empty), "-s", f"{W}x{H}", "-f", "420p:10", "--metrics", "psnr", "-v", "0"])
        if emp.returncode == 0 or "Summary:" in emp.stdout or "Total frames:" in emp.stdout:
            rc |= fail("empty file looked like a score")
        else:
            print("OK   empty-file-no-summary")

        missing = run(["-r", "/no/such_ref.yuv", "-t", "/no/such_test.yuv",
                       "-s", f"{W}x{H}", "-f", "420p:10", "--metrics", "psnr", "-v", "1"])
        if missing.returncode != 2 or "cannot open reference file '/no/such_ref.yuv'" not in missing.stderr:
            rc |= fail(f"missing file rc={missing.returncode} stderr={missing.stderr!r}")
        else:
            print("OK   missing-file")

        fifo_ref = tmp / "ref.fifo"
        fifo_test = tmp / "test.fifo"
        os.mkfifo(fifo_ref)
        os.mkfifo(fifo_test)
        proc = subprocess.Popen(
            [str(BIN), "-r", str(fifo_ref), "-t", str(fifo_test), "-s", f"{W}x{H}", "-f", "420p:10",
             "--metrics", "psnr,ssim,msssim", "-v", "1", "-n", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        blob = ref.read_bytes()
        with open(fifo_ref, "wb") as rf, open(fifo_test, "wb") as tf:
            rf.write(blob)
            tf.write(blob)
        out, err = proc.communicate(timeout=30)
        if proc.returncode != 0 or not PSNR.search(out) or not MS.search(out):
            rc |= fail(f"fifo rc={proc.returncode} err={err[-200:]}")
        else:
            print("OK   fifo")

    if rc:
        print("Log-structure checks FAILED", file=sys.stderr)
        return 1
    print("All log-structure checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
