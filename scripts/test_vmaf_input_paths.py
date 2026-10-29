#!/usr/bin/env python3
"""Input-path / decode / CUDA-combo checks for CPU VMAF.

Marks NOT EXECUTED when required binaries or fixtures are absent.
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
PIXVQA_CPU = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
PIXVQA_CUDA = Path(os.environ.get("PIXVQA_CUDA_BIN", ROOT / "build" / "cuda" / "pixvqa"))
PIXVQA_SWDEC = Path(os.environ.get("PIXVQA_SWDEC_BIN", ROOT / "build" / "swdec" / "pixvqa"))
PIXVQA_FULL = Path(os.environ.get("PIXVQA_FULL_BIN", ROOT / "build" / "full" / "pixvqa"))
MODEL = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"
W, H, N = 320, 240, 3


def write_yuv10(path: Path, n: int = N) -> None:
    with path.open("wb") as f:
        for fi in range(n):
            for row in range(H):
                for col in range(W):
                    v = (200 + fi * 5 + (row + col) % 100) & 0x3FF
                    f.write(struct.pack("<H", v))
            for seed in (512, 520):
                for row in range(H // 2):
                    for col in range(W // 2):
                        f.write(struct.pack("<H", (seed + (row + col + fi) % 8) & 0x3FF))


def parse_psnr_avg(out: str) -> float | None:
    m = re.search(r"\[ PSNR .*? avg:([0-9.]+)", out)
    return float(m.group(1)) if m else None


def main() -> int:
    results: list[str] = []

    # Raw YUV → CPU VMAF
    if PIXVQA_CPU.is_file() and MODEL.is_file():
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ref, dist = td_path / "r.yuv", td_path / "d.yuv"
            write_yuv10(ref)
            write_yuv10(dist)
            # Mild 10-bit luma distortion (preserve little-endian packing, clamp to 10-bit).
            blob = bytearray(dist.read_bytes())
            for i in range(0, min(W * H * 2, len(blob)), 2):
                v = blob[i] | (blob[i + 1] << 8)
                v = min(1023, v + 17)
                blob[i] = v & 0xFF
                blob[i + 1] = (v >> 8) & 0xFF
            dist.write_bytes(blob)
            p = subprocess.run(
                [
                    str(PIXVQA_CPU),
                    "--metrics", "vmaf",
                    "--model",
                    str(MODEL),
                    
                    "-v",
                    "0",
                    "-r",
                    str(ref),
                    "-s",
                    f"{W}x{H}",
                    "-f",
                    "420p:10",
                    "-t",
                    str(dist),
                    "-s",
                    f"{W}x{H}",
                    "-f",
                    "420p:10"],
                capture_output=True,
                text=True,
            )
            text = p.stdout + p.stderr
            if p.returncode == 0 and ("[ VMAF:" in text or "[ VMAF mean:" in text):
                results.append("PASSED raw_yuv_cpu_vmaf")
            else:
                results.append("FAILED raw_yuv_cpu_vmaf")
                print(text, file=sys.stderr)
    else:
        results.append("NOT EXECUTED raw_yuv_cpu_vmaf (missing cpu binary or model)")

    # Software HEVC → CPU VMAF
    if PIXVQA_SWDEC.is_file():
        results.append(
            "NOT EXECUTED software_hevc_cpu_vmaf "
            "(no HEVC fixture / decoded-YUV dump harness in this pass)"
        )
    else:
        results.append("NOT EXECUTED software_hevc_cpu_vmaf (build/swdec/pixvqa absent)")

    # NVDEC HEVC → CPU VMAF
    if PIXVQA_FULL.is_file():
        results.append(
            "NOT EXECUTED nvdec_hevc_cpu_vmaf "
            "(no NVDEC fixture / decoded-YUV dump harness in this pass)"
        )
    else:
        results.append("NOT EXECUTED nvdec_hevc_cpu_vmaf (build/full/pixvqa absent)")

    # CUDA metrics + CPU VMAF combo: VMAF must not change PSNR vs PSNR-only.
    # Requires a CUDA build that also linked libvmaf (--vmaf present in --help).
    cuda_has_vmaf = False
    if PIXVQA_CUDA.is_file():
        help_p = subprocess.run([str(PIXVQA_CUDA), "--help"], capture_output=True, text=True)
        cuda_has_vmaf = "msssim, vmaf" in help_p.stdout and "--model PATH" in help_p.stdout
    if PIXVQA_CUDA.is_file() and MODEL.is_file() and cuda_has_vmaf:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ref, dist = td_path / "r.yuv", td_path / "d.yuv"
            write_yuv10(ref)
            write_yuv10(dist)
            blob = bytearray(dist.read_bytes())
            for i in range(0, min(W * H * 2, len(blob)), 2):
                v = blob[i] | (blob[i + 1] << 8)
                v = min(1023, v + 7)
                blob[i] = v & 0xFF
                blob[i + 1] = (v >> 8) & 0xFF
            dist.write_bytes(blob)
            base_cmd = [
                str(PIXVQA_CUDA),
                "--metric-device", "gpu",
                "--metrics", "psnr",
                "-v",
                "0",
                "-r",
                str(ref),
                "-s",
                f"{W}x{H}",
                "-f",
                "420p:10",
                "-t",
                str(dist),
                "-s",
                f"{W}x{H}",
                "-f",
                "420p:10"]
            p0 = subprocess.run(base_cmd, capture_output=True, text=True)
            p1 = subprocess.run(
                base_cmd
                + [
                    "--metrics", "vmaf",
                    "--model",
                    str(MODEL)],
                capture_output=True,
                text=True,
            )
            if p0.returncode != 0:
                results.append(f"FAILED cuda_psnr_baseline rc={p0.returncode}")
            elif p1.returncode != 0:
                results.append(f"FAILED cuda_psnr_plus_vmaf rc={p1.returncode}")
                print(p1.stderr, file=sys.stderr)
            else:
                a = parse_psnr_avg(p0.stdout + p0.stderr)
                b = parse_psnr_avg(p1.stdout + p1.stderr)
                if a is None or b is None:
                    results.append("FAILED cuda_psnr_plus_vmaf (missing PSNR)")
                elif abs(a - b) > 1e-9:
                    results.append(f"FAILED cuda_psnr_plus_vmaf PSNR changed {a} vs {b}")
                elif "[ VMAF mean:" not in (p1.stdout + p1.stderr) and "[ VMAF:" not in (
                    p1.stdout + p1.stderr
                ):
                    results.append("FAILED cuda_psnr_plus_vmaf (missing VMAF)")
                else:
                    results.append("PASSED cuda_psnr_plus_cpu_vmaf")
    elif PIXVQA_CUDA.is_file() and not cuda_has_vmaf:
        results.append(
            "NOT EXECUTED cuda_psnr_plus_cpu_vmaf "
            "(CUDA binary present but built without libvmaf / --vmaf)"
        )
    else:
        results.append("NOT EXECUTED cuda_psnr_plus_cpu_vmaf (build/full/pixvqa absent)")

    # CPU PSNR + CPU VMAF combo (always available when VMAF-enabled CPU binary exists).
    if PIXVQA_CPU.is_file() and MODEL.is_file():
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ref, dist = td_path / "r.yuv", td_path / "d.yuv"
            write_yuv10(ref)
            write_yuv10(dist)
            blob = bytearray(dist.read_bytes())
            for i in range(0, min(W * H * 2, len(blob)), 2):
                v = blob[i] | (blob[i + 1] << 8)
                v = min(1023, v + 9)
                blob[i] = v & 0xFF
                blob[i + 1] = (v >> 8) & 0xFF
            dist.write_bytes(blob)
            base = [
                str(PIXVQA_CPU),
                "--metrics", "psnr",
                "-v",
                "0",
                "-r",
                str(ref),
                "-s",
                f"{W}x{H}",
                "-f",
                "420p:10",
                "-t",
                str(dist),
                "-s",
                f"{W}x{H}",
                "-f",
                "420p:10"]
            p0 = subprocess.run(base, capture_output=True, text=True)
            p1 = subprocess.run(
                base + ["--metrics", "vmaf", "--model", str(MODEL)],
                capture_output=True,
                text=True,
            )
            a = parse_psnr_avg(p0.stdout + p0.stderr)
            b = parse_psnr_avg(p1.stdout + p1.stderr)
            text1 = p1.stdout + p1.stderr
            if (
                p0.returncode == 0
                and p1.returncode == 0
                and a is not None
                and b is not None
                and abs(a - b) <= 1e-9
                and ("[ VMAF mean:" in text1 or "[ VMAF:" in text1)
            ):
                results.append("PASSED cpu_psnr_plus_cpu_vmaf")
            else:
                results.append("FAILED cpu_psnr_plus_cpu_vmaf")
                print(p0.stderr + p1.stderr, file=sys.stderr)
    else:
        results.append("NOT EXECUTED cpu_psnr_plus_cpu_vmaf")

    # Chroma format claims
    results.append(
        "NOT APPLICABLE numerical_422_444 "
        "(libvmaf API accepts 422/444 pictures; vmaf_v1.0.16 models are not "
        "numerically validated here for non-420 — documented only)"
    )

    for line in results:
        print(line)
    out = ROOT / "reports" / "vmaf_input_path_results.txt"
    out.write_text("\n".join(results) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if any(x.startswith("FAILED") for x in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
