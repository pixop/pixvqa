#!/usr/bin/env python3
"""Regression test for compact summary average metric formatting."""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.parse_metrics import (  # noqa: E402
    PU21_MSSSIM,
    PU21_PSNR,
    PU21_SSIM,
    SSIM_SUMMARY,
    validate_summary_metrics,
)

BIN = Path(__import__("os").environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def write_420p10(path: Path, w: int, h: int, y: int, u: int, v: int, frames: int) -> None:
    y_plane = struct.pack("<" + "H" * (w * h), *([y] * (w * h)))
    c_plane = struct.pack("<" + "H" * (w * h // 4), *([u] * (w * h // 4)))
    v_plane = struct.pack("<" + "H" * (w * h // 4), *([v] * (w * h // 4)))
    path.write_bytes((y_plane + c_plane + v_plane) * frames)


def run_summary(metrics: str, transfer: str, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    w = h = 64
    with tempfile.TemporaryDirectory(prefix="pixvqa_summary_fmt_") as td:
        root = Path(td)
        ref = root / "ref.yuv"
        dist = root / "dist.yuv"
        write_420p10(ref, w, h, 520, 512, 512, 2)
        write_420p10(dist, w, h, 560, 512, 516, 2)
        args = [
            str(BIN),
            "--device",
            "cpu",
            "-v",
            "0",
            "--metrics",
            metrics,
            "--hdr-transfer",
            transfer,
            "-r",
            str(ref),
            "-s",
            f"{w}x{h}",
            "-f",
            "420p:10",
            "-t",
            str(dist),
            "-s",
            f"{w}x{h}",
            "-f",
            "420p:10",
        ]
        if extra:
            args.extend(extra)
        return subprocess.run(args, capture_output=True, text=True)


def assert_parsed(proc: subprocess.CompletedProcess[str], label: str) -> str:
    if proc.returncode != 0:
        fail(f"{label}: rc={proc.returncode}\n{proc.stderr[-400:]}")
        return ""
    errors = validate_summary_metrics(proc.stdout)
    if errors:
        for item in errors:
            fail(f"{label}: {item}")
    return proc.stdout


def main() -> int:
    if not BIN.is_file():
        print(f"missing binary: {BIN}", file=sys.stderr)
        return 2

    sdr = run_summary("psnr,ssim,msssim", "sdr")
    out_sdr = assert_parsed(sdr, "sdr-native")
    if out_sdr:
        if SSIM_SUMMARY.search(out_sdr) is None:
            fail("sdr-native: could not parse SSIM summary")
        else:
            print("OK   sdr summary compact format")

    pq = run_summary("pu21-psnr,pu21-ssim,pu21-msssim", "pq")
    out_pq = assert_parsed(pq, "pq-pu21")
    if out_pq:
        for pattern, name in (
            (PU21_PSNR, "PU21-PSNR"),
            (PU21_SSIM, "PU21-SSIM"),
            (PU21_MSSSIM, "PU21-MS-SSIM"),
        ):
            if pattern.search(out_pq) is None:
                fail(f"pq-pu21: missing parsed {name}")
        pu21_lines = [
            line
            for line in out_pq.splitlines()
            if line.startswith("[ PU21-")
        ]
        if len(pu21_lines) != 3:
            fail(f"pq-pu21: expected three PU21 summary lines, got {len(pu21_lines)}")
        elif len({line.split()[1] for line in pu21_lines}) != 3:
            fail("pq-pu21: PU21 metrics must not share a summary line")
        else:
            print("OK   pq PU21 one metric per summary line")

    hlg = run_summary("pu21-psnr,pu21-ssim,pu21-msssim", "hlg", ["--hdr-peak-nits", "1000"])
    assert_parsed(hlg, "hlg-pu21")
    if hlg.returncode == 0 and not validate_summary_metrics(hlg.stdout):
        print("OK   hlg PU21 summary compact format")

    if FAILS:
        print(f"{FAILS} failure(s)", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
