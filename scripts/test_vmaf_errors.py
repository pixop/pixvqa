#!/usr/bin/env python3
"""CLI error-path checks for Phase-1 VMAF path-based --model selection."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXVQA = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
MODEL = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"


def run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(PIXVQA), *args], capture_output=True, text=True, cwd=cwd)


def write_yuv(path: Path, w: int = 64, h: int = 64) -> None:
    with path.open("wb") as f:
        f.write(bytes(w * h * 2))
        f.write(bytes((w // 2) * (h // 2) * 2))
        f.write(bytes((w // 2) * (h // 2) * 2))


def expect_fail(name: str, args: list[str], needle: str, code: int = 2, cwd: Path | None = None) -> None:
    p = run(args, cwd=cwd)
    if p.returncode != code:
        raise SystemExit(f"{name}: expected exit {code}, got {p.returncode}\n{p.stderr}{p.stdout}")
    if needle not in p.stderr and needle not in p.stdout:
        raise SystemExit(f"{name}: missing needle {needle!r}\n{p.stderr}{p.stdout}")
    print(f"OK   {name}")


def main() -> int:
    if not PIXVQA.is_file():
        print("missing pixvqa", file=sys.stderr)
        return 2
    if not MODEL.is_file():
        print(f"missing model JSON: {MODEL}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ref = td_path / "r.yuv"
        dist = td_path / "d.yuv"
        write_yuv(ref)
        write_yuv(dist)
        base = [
            "-r",
            str(ref),
            "-s",
            "64x64",
            "-f",
            "420p:10",
            "-t",
            str(dist),
            "-s",
            "64x64",
            "-f",
            "420p:10"]
        expect_fail("missing-model", ["--metrics", "vmaf", *base], "--model")
        # 64x64 is below the feature extractor. The GPU request must warn and
        # select CPU libvmaf before that failure, rather than refusing the device.
        expect_fail(
            "gpu-vmaf-falls-back-then-feature-size",
            ["--metrics", "vmaf", "--metric-device", "gpu", "--model", str(MODEL), *base],
            "using CPU libvmaf",
        )
        expect_fail(
            "missing-file",
            ["--metrics", "vmaf", "--model", str(td_path / "no_such_model.json"),  *base],
            "not found",
        )
        bad = td_path / "not_a_model.json"
        bad.write_text("{ not json for libvmaf\n", encoding="utf-8")
        expect_fail(
            "invalid-model",
            ["--metrics", "vmaf", "--model", str(bad),  *base],
            "failed to load model",
            code=1,  # thrown during start() after banner
        )
        expect_fail(
            "conflict-model-flags",
            [
                "--metrics", "vmaf",
                "--model",
                str(MODEL),
                "--vmaf-model",
                str(td_path / "other.json"),
                *base],
            "disagree",
        )
        expect_fail(
            "hdr-pq",
            ["--metrics", "vmaf", "--model", str(MODEL),  "--tf", "st2084", *base],
            "rejects explicit HDR",
        )
        expect_fail(
            "hdr-hlg",
            ["--metrics", "vmaf", "--model", str(MODEL),  "--tf", "stdb67", *base],
            "rejects explicit HDR",
        )
        expect_fail(
            "qcs-vmaf-range",
            ["--metrics", "psnr", "--qcs", "--qcs-range", "vmaf=70:100", *base],
            "not supported",
        )
        expect_fail(
            "qcs-vmaf-only",
            ["--metrics", "vmaf", "--model", str(MODEL),  "--qcs", *base],
            "not a QCS metric",
        )
        # Valid pre-existing QCS with a supported metric must still succeed
        # (geometry here is fine for PSNR; ignore VMAF).
        p = run(["--metrics", "psnr", "--qcs", *base])
        if p.returncode != 0:
            raise SystemExit(f"psnr-qcs: unexpected fail\n{p.stderr}{p.stdout}")
        if "PSNR-QCS" not in (p.stdout + p.stderr):
            raise SystemExit(f"psnr-qcs: missing QCS output\n{p.stderr}{p.stdout}")
        print("OK   psnr-qcs")
        expect_fail(
            "model-without-vmaf",
            ["--metrics", "psnr", "--model", str(MODEL), *base],
            "require vmaf in --metrics",
        )
        # Relative path: cwd = repo root, model/vmaf_...json
        # Too-small geometry for this known model should fail before processing.
        expect_fail(
            "too-small",
            ["--metrics", "vmaf", "--model", "model/vmaf_v1.0.16_3d0h.json",  *base],
            "needs at least",
            cwd=ROOT,
        )
        # Alias agreement
        p = run(
            [
                "--metrics", "vmaf",
                "--model",
                str(MODEL),
                "--vmaf-model",
                str(MODEL),
                
                *base]
        )
        # Still fails on size, but must not be a disagree error.
        if "disagree" in (p.stderr + p.stdout):
            raise SystemExit("alias agreement unexpectedly rejected")
        if "needs at least" not in (p.stderr + p.stdout):
            raise SystemExit(f"alias agreement: unexpected output\n{p.stderr}{p.stdout}")
        print("OK   alias-agree")
    print("All VMAF error checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
