#!/usr/bin/env python3
"""Run --help, --version, and every pixvqa example published in README.md."""
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
README = (ROOT / "README.md").read_text(encoding="utf-8")
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def run(bin_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(bin_path), *args], capture_output=True, text=True, timeout=60)


def check_help(bin_path: Path, kind: str) -> None:
    done = run(bin_path, ["--help"])
    text = done.stdout
    if done.returncode != 0:
        fail(f"{kind} help rc {done.returncode}")
        return
    for needle in (
        "1. Usage", "2. Input files", "3. Metrics", "4. Execution",
        "5. QCS", "6. Frame rate", "7. General", "8. Examples",
        "five-scale product", "--metrics", "msssim", "--qcs-range",
        "--metric-device", "--decode", "--threads", "--model PATH",
        "Official Netflix VMAF",
        "--metrics psnr,xpsnr,ssim,msssim,vmaf",
        "20.8333", "10.4167",
    ):
        if needle not in text:
            fail(f"{kind} help missing {needle}")
            return
    if "NVIDIA" in text or "GeForce" in text:
        fail(f"{kind} help initialized a device")
        return
    short = run(bin_path, ["-h"])
    if short.returncode != 0 or short.stdout != text:
        fail(f"{kind} -h disagrees with --help")
        return
    print(f"OK   {kind} help")


def check_version(bin_path: Path, kind: str) -> None:
    done = run(bin_path, ["--version"])
    if done.returncode != 0:
        fail(f"{kind} version rc {done.returncode} {done.stderr}")
        return
    if "pixvqa version 0.3.0-dev" not in done.stdout or f"build: {kind}" not in done.stdout:
        fail(f"{kind} version text {done.stdout!r}")
        return
    if "decode: none" not in done.stdout:
        fail(f"{kind} version missing decode capability {done.stdout!r}")
        return
    if "NVIDIA" in done.stdout or "GeForce" in done.stdout:
        fail(f"{kind} version initialized a device")
        return
    print(f"OK   {kind} version")


def write_pair(directory: Path, width: int = 176, height: int = 176) -> None:
    h, w = height, width
    ch, cw = h // 2, w // 2
    ref = np.full((h, w), 400, dtype=np.uint16)
    tst = ref.copy()
    tst[20:40, 20:40] = 460
    chroma = np.full((ch, cw), 512, dtype=np.uint16)

    def pack(y):
        return y.tobytes() + chroma.tobytes() + chroma.tobytes()

    (directory / "reference.yuv").write_bytes(pack(ref))
    (directory / "reconstructed.yuv").write_bytes(pack(tst))


def examples() -> list[str]:
    blocks = re.findall(r"```bash\n(.*?)```", README, flags=re.S)
    return [b.strip() for b in blocks if "pixvqa" in b]


def main() -> int:
    if not CPU.exists() or not CUDA.exists():
        print("build both binaries first", file=sys.stderr)
        return 2
    check_help(CPU, "cpu")
    check_help(CUDA, "cuda")
    check_version(CPU, "cpu")
    check_version(CUDA, "cuda")
    bad = run(CPU, ["--not-a-real-option"])
    if bad.returncode == 0 or "PARSE ERROR" not in bad.stderr:
        fail(f"unknown option rc={bad.returncode} {bad.stderr}")
    else:
        print("OK   unknown option")
    cpu_cuda = run(CPU, ["--metric-device", "gpu", "--metrics", "psnr", "-r", "a", "-s", "16x16", "-f", "420p:10",
                         "-t", "b", "-s", "16x16", "-f", "420p:10"])
    if cpu_cuda.returncode == 0:
        fail("CPU build accepted --metric-device gpu")
    else:
        print("OK   cpu build rejects cuda kernel")
    blocks = examples()
    if len(blocks) < 6:
        fail(f"expected the published examples, found {len(blocks)}")
    env = os.environ.copy()
    ffmpeg_bin = os.environ.get("FFMPEG")
    if ffmpeg_bin:
        env["PATH"] = str(Path(ffmpeg_bin).parent) + os.pathsep + env.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="cli_help_") as tmp:
        td = Path(tmp)
        (td / "build" / "cpu").mkdir(parents=True)
        (td / "build" / "cuda").mkdir(parents=True)
        (td / "build" / "cpu" / "pixvqa").symlink_to(CPU)
        (td / "build" / "cuda" / "pixvqa").symlink_to(CUDA)
        write_pair(td)
        # README VMAF examples reference model/*.json relative to the repo root.
        model_src = ROOT / "model"
        model_dst = td / "model"
        if model_src.is_dir():
            model_dst.mkdir(exist_ok=True)
            for p in model_src.glob("*.json"):
                (model_dst / p.name).write_bytes(p.read_bytes())
        for i, block in enumerate(blocks):
            # VMAF examples use 320x240 / 1920x1080 (SpEED rejects tiny frames).
            if "vmaf" in block:
                if "1920x1080" in block:
                    write_pair(td, 1920, 1080)
                else:
                    write_pair(td, 320, 240)
            else:
                write_pair(td, 176, 176)
            proc = subprocess.run(["bash", "-lc", block], cwd=td, capture_output=True, text=True,
                                  env=env, timeout=120)
            if proc.returncode != 0:
                fail(f"example {i} rc={proc.returncode}\n{block}\n{proc.stderr[-400:]}")
            else:
                print(f"OK   example {i}")
    if FAILS:
        print(f"{FAILS} CLI checks failed", file=sys.stderr)
        return 1
    print("All CLI help checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
