#!/usr/bin/env python3
"""Pixel equivalence of CPU and GPU HEVC decode on a short lossless clip."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "full" / "pixvqa"))
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
# NVDEC rejects widths below 144. 176 is even and inside that range.
W = H = 176
N = 4


def fail(msg: str) -> None:
    print("FAIL", msg, file=sys.stderr)
    raise SystemExit(1)


def write_yuv(path: Path, bias: int) -> None:
    with path.open("wb") as f:
        for i in range(N):
            y = bytearray()
            for row in range(H):
                for col in range(W):
                    v = (80 + bias + i * 17 + row * 3 + col) & 0x3FF
                    y += v.to_bytes(2, "little")
            u = bytearray()
            vch = bytearray()
            for row in range(H // 2):
                for col in range(W // 2):
                    u += ((400 + bias + i + row) & 0x3FF).to_bytes(2, "little")
                    vch += ((600 + i + col) & 0x3FF).to_bytes(2, "little")
            f.write(y + u + vch)


def ff(args: list[str]) -> None:
    p = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args], capture_output=True, text=True)
    if p.returncode != 0:
        fail(p.stderr or "ffmpeg failed")


def pix(args: list[str], dump: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if dump is not None:
        env["PIXVQA_DECODE_YUV"] = str(dump)
    return subprocess.run([str(BIN), *args], capture_output=True, text=True, env=env)


def sides(ref: Path, dist: Path) -> list[str]:
    return ["-r", str(ref), "-s", f"{W}x{H}", "-f", "420p:10",
            "-t", str(dist), "-s", f"{W}x{H}", "-f", "420p:10"]


def score_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines()
            if any(k in ln for k in ("PSNR", "SSIM", "XPSNR", "MS-SSIM")) and not ln.startswith("  ")]


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN)
        return 2
    with tempfile.TemporaryDirectory(prefix="pixvqa_p3gpu_") as td:
        t = Path(td)
        ref, dist = t / "ref.yuv", t / "dist.yuv"
        write_yuv(ref, 0)
        write_yuv(dist, 40)
        hevc, mp4 = t / "dist.hevc", t / "dist.mp4"
        common = ["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}", "-r", "24",
                  "-i", str(dist), "-frames:v", str(N), "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                  "-profile:v", "main10", "-x265-params", "lossless=1:bframes=0"]
        ff([*common, "-f", "hevc", str(hevc)])
        ff([*common, "-tag:v", "hvc1", str(mp4)])
        raw = t / "dist.dec.yuv"
        ff(["-i", str(hevc), "-f", "rawvideo", "-pix_fmt", "yuv420p10le", str(raw)])
        if raw.read_bytes() != dist.read_bytes():
            fail("ffmpeg decode differs from source")
        cpu_dump, gpu_dump = t / "cpu.yuv", t / "gpu.yuv"
        base = ["--metrics", "psnr", "--threads", "2"]
        cpu = pix([*base, "--decode", "cpu", *sides(ref, hevc)], cpu_dump)
        gpu = pix([*base, "--decode", "gpu", *sides(ref, hevc)], gpu_dump)
        if cpu.returncode != 0:
            fail(f"cpu decode {cpu.stderr}")
        if gpu.returncode != 0:
            fail(f"gpu decode {gpu.stderr}")
        if "distorted: cpu" not in cpu.stdout or "reference: raw" not in cpu.stdout:
            fail(f"cpu summary\n{cpu.stdout}")
        if "distorted: gpu" not in gpu.stdout or "reference: raw" not in gpu.stdout:
            fail(f"gpu summary\n{gpu.stdout}")
        cb, gb, rb = cpu_dump.read_bytes(), gpu_dump.read_bytes(), raw.read_bytes()
        print(f"bytes cpu={len(cb)} gpu={len(gb)} ffmpeg={len(rb)}")
        if cb != rb:
            fail("CPU decode pixels differ from ffmpeg")
        if gb != rb:
            differ = sum(a != b for a, b in zip(gb, rb)) + abs(len(gb) - len(rb))
            fail(f"GPU decode pixels differ from ffmpeg at {differ} bytes")
        print("OK   CPU and GPU elementary decode match ffmpeg pixels")

        mp4_dump = t / "mp4gpu.yuv"
        mp4_run = pix([*base, "--decode", "gpu", *sides(ref, mp4)], mp4_dump)
        if mp4_run.returncode != 0 or mp4_dump.read_bytes() != rb:
            fail(f"mp4 gpu {mp4_run.returncode} {mp4_run.stderr}")
        print("OK   GPU MP4 decode matches the elementary pixels")

        ref_h, dist_k = t / "ref.hevc", t / "dist.mkv"
        ref_common = ["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}", "-r", "24",
                      "-i", str(ref), "-frames:v", str(N), "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                      "-profile:v", "main10", "-x265-params", "lossless=1:bframes=0"]
        ff([*ref_common, "-f", "hevc", str(ref_h)])
        ff([*common, "-f", "matroska", str(dist_k)])
        both = pix(["--metrics", "psnr,ssim,msssim", "--threads", "2", "--decode", "gpu", *sides(ref_h, hevc)])
        raw_scores = pix(["--metrics", "psnr,ssim,msssim", "--threads", "2", *sides(ref, dist)])
        if both.returncode or raw_scores.returncode:
            fail(f"dual gpu rc {both.returncode} {raw_scores.returncode}\n{both.stderr}")
        if "reference: gpu" not in both.stdout or "distorted: gpu" not in both.stdout:
            fail(f"dual gpu summary\n{both.stdout}")
        if score_lines(both.stdout) != score_lines(raw_scores.stdout):
            fail("dual GPU scores differ from raw\n" + "\n".join(score_lines(both.stdout)))
        else:
            print("OK   both inputs on GPU match the raw scores")
        mkv = pix(["--metrics", "psnr", "--decode", "gpu", *sides(ref, dist_k)], t / "mkvgpu.yuv")
        if mkv.returncode != 0 or (t / "mkvgpu.yuv").read_bytes() != rb:
            fail(f"mkv gpu {mkv.returncode} {mkv.stderr}")
        else:
            print("OK   GPU MKV decode matches the elementary pixels")

        metrics = ["--metrics", "psnr,ssim,msssim,xpsnr", "--fps", "24", "--threads", "2"]
        cpu_m = pix([*metrics, "--decode", "cpu", "--metric-device", "cpu", *sides(ref, hevc)])
        gpu_m = pix([*metrics, "--decode", "gpu", "--metric-device", "gpu", *sides(ref, hevc)])
        if cpu_m.returncode or gpu_m.returncode:
            fail(f"metric rc cpu={cpu_m.returncode} gpu={gpu_m.returncode}\n{gpu_m.stderr}")
        if score_lines(cpu_m.stdout) != score_lines(gpu_m.stdout):
            fail("metric text differs\n" + "\n".join(score_lines(cpu_m.stdout)) + "\n!=\n" + "\n".join(score_lines(gpu_m.stdout)))
        print("OK   CPU and GPU metric text match on the decoded pair")

        auto = pix(["--metrics", "psnr", "--decode", "auto", *sides(hevc, hevc)])
        if auto.returncode != 0 or "reference: cpu" not in auto.stdout or "distorted: cpu" not in auto.stdout:
            fail(f"auto decode\n{auto.stdout}\n{auto.stderr}")
        if "999.99" not in auto.stdout:
            fail("identical auto pair was not identical")
        print("OK   auto keeps software decode when both decoders exist")
    print("All GPU decode checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
