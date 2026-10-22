#!/usr/bin/env python3
"""Check public PSNR, SSIM, and MS-SSIM log structure."""
from __future__ import annotations

import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "cpu" / "pixvqa"
W = H = 64

MS_LINE = re.compile(
    r"MS-SSIM\s+y:(-?[0-9.]+)\s+u:(-?[0-9.]+)\s+v:(-?[0-9.]+)\s+avg:(-?[0-9.]+)"
)
SSIM_LINE = re.compile(
    r"(?<![A-Z-])SSIM\s+y:(-?[0-9.]+)\s+u:(-?[0-9.]+)\s+v:(-?[0-9.]+)\s+avg:(-?[0-9.]+)"
)
PSNR_LINE = re.compile(
    r"PSNR\s+y:([0-9.]+)\s+u:([0-9.]+)\s+v:([0-9.]+)\s+avg:([0-9.]+)\s*\]\s*"
    r"\[\s*MSE\s+y:([0-9.]+)\s+u:([0-9.]+)\s+v:([0-9.]+)\s+avg:([0-9.]+)"
)


def fail(msg: str) -> int:
    print("FAIL", msg, file=sys.stderr)
    return 1


def write_pair(ref: Path, test: Path, frames: int = 2) -> None:
    def one(y: int) -> bytes:
        ys = [y] * (W * H)
        us = [512] * ((W // 2) * (H // 2))
        vs = [512] * ((W // 2) * (H // 2))
        return struct.pack("<" + "H" * len(ys + us + vs), *(ys + us + vs))

    ref.write_bytes(b"".join(one(400) for _ in range(frames)))
    test.write_bytes(b"".join(one(420) for _ in range(frames)))


def run(args: list[str]) -> str:
    done = subprocess.run([str(BIN), *args], capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError(f"pixvqa rc={done.returncode} stderr={done.stderr[-400:]}")
    return done.stdout


def parse_metrics(text: str) -> dict:
    ms = list(MS_LINE.finditer(text))
    ss = list(SSIM_LINE.finditer(text))
    ps = list(PSNR_LINE.finditer(text))
    return {
        "n_ms": len(ms),
        "n_ssim": len(ss),
        "n_psnr": len(ps),
        "ms_summary": ms[-1].groups() if ms else None,
        "ssim_summary": ss[-1].groups() if ss else None,
        "psnr_summary": ps[-1].groups() if ps else None,
        "ms_frames": [m.groups() for m in ms[:-1]] if len(ms) > 1 else [],
        "psnr_frames": [m.groups() for m in ps[:-1]] if len(ps) > 1 else [],
    }


def main() -> int:
    if not BIN.exists():
        return fail(f"missing {BIN}")
    rc = 0
    with tempfile.TemporaryDirectory(prefix="pixvqa_prodparse_") as td:
        tmp = Path(td)
        ref = tmp / "ref.yuv"
        test = tmp / "test.yuv"
        write_pair(ref, test, frames=2)
        args = [
            "--cs", "bt2020l", "--tf", "st2084",
            "-r", str(ref), "-s", f"{W}x{H}", "-f", "420p:10",
            "-t", str(test), "-s", f"{W}x{H}", "-f", "420p:10",
            "--metrics", "psnr,ssim,msssim", "--threads", "8", "-v", "1", "-n", "2"]
        our_text = run(args)
        (tmp / "ours.log").write_text(our_text)
        met = parse_metrics(our_text)
        ms = list(MS_LINE.finditer(our_text))
        if met["n_psnr"] < 3 or met["n_ssim"] < 3 or len(ms) < 3:
            rc |= fail(
                f"expected per-frame and summary PSNR/SSIM/MS-SSIM, "
                f"got psnr={met['n_psnr']} ssim={met['n_ssim']} msssim={len(ms)}"
            )
            print("OK   public PSNR/SSIM/MS-SSIM log structure")

        rc |= check_avg_confusion(our_text, "ours")

    if rc:
        print("Public parser checks FAILED", file=sys.stderr)
        return 1
    print("All public parser checks passed")
    return 0


def check_avg_confusion(text: str, label: str) -> int:
    """A combined line must keep PSNR, SSIM, and MS-SSIM averages distinct."""
    frame_lines = [ln for ln in text.splitlines() if re.match(r"^\d+\s*:", ln)]
    if not frame_lines:
        return fail(f"{label}: no frame lines for avg confusion check")
    line = frame_lines[0]
    if "[ PSNR" not in line or "[ SSIM" not in line or "[ MS-SSIM" not in line:
        return fail(f"{label}: combined frame line missing PSNR/SSIM/MS-SSIM: {line}")
    psnr_m = PSNR_LINE.search(line)
    ssim_m = SSIM_LINE.search(line)
    ms_m = MS_LINE.search(line)
    if not psnr_m or not ssim_m or not ms_m:
        return fail(f"{label}: PSNR/SSIM/MS-SSIM regex failed on combined line")
    bare = re.findall(r"avg:([0-9.]+)", line)
    if bare[0] != psnr_m.group(4):
        return fail(f"{label}: first bare avg: {bare[0]} != PSNR avg {psnr_m.group(4)}")
    if psnr_m.group(4) == ssim_m.group(4):
        return fail(f"{label}: PSNR avg equals SSIM avg; fixture too weak")
    print(f"OK   avg-confusion guard ({label})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
