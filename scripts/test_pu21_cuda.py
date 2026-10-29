#!/usr/bin/env python3
"""Compare CPU and CUDA PU21 on generated frames. CPU is the reference."""
from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
GPU = Path(os.environ.get("PIXVQA_CUDA_BIN", ROOT / "build" / "cuda" / "pixvqa"))
NAMES = ("PU21-PSNR", "PU21-SSIM", "PU21-MS-SSIM")
FAILS = 0
WORST = {name: 0.0 for name in NAMES}


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def write_frame(path: Path, w: int, h: int, chroma: str, bits: int, frames: list[tuple[int, int, int]]) -> None:
    cw = w if chroma == "444" else w // 2
    ch = h if chroma != "420" else h // 2
    blobs = []
    for ycode, ucode, vcode in frames:
        if bits == 8:
            blobs.append(bytes([ycode & 255]) * (w * h))
            blobs.append(bytes([ucode & 255]) * (cw * ch))
            blobs.append(bytes([vcode & 255]) * (cw * ch))
        else:
            blobs.append(struct.pack("<" + "H" * (w * h), *([ycode] * (w * h))))
            blobs.append(struct.pack("<" + "H" * (cw * ch), *([ucode] * (cw * ch))))
            blobs.append(struct.pack("<" + "H" * (cw * ch), *([vcode] * (cw * ch))))
    path.write_bytes(b"".join(blobs))


def values(text: str, name: str) -> list[float]:
    return [float(item) for item in re.findall(rf"\[ {re.escape(name)}:([0-9.]+) \]", text)]


def qcs(text: str, name: str) -> list[float]:
    return [float(item) for item in re.findall(rf"\[ {re.escape(name)}-QCS:([0-9.]+) \]", text)]


def run(binary: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary), *args], capture_output=True, text=True)


def compare(label: str, cpu: str, gpu: str, qcs_on: bool) -> None:
    if "PU21-PSNR: gpu" not in gpu and "PU21-SSIM: gpu" not in gpu and "PU21-MS-SSIM: gpu" not in gpu:
        fail(f"{label}: CUDA summary did not select gpu\n{gpu[:500]}")
        return
    for name in NAMES:
        a = values(cpu, name)
        b = values(gpu, name)
        if not a or len(a) != len(b):
            fail(f"{label}: {name} count cpu={len(a)} gpu={len(b)}")
            continue
        for index, (left, right) in enumerate(zip(a, b)):
            delta = abs(left - right)
            WORST[name] = max(WORST[name], delta)
            rel = delta / max(abs(left), 1e-12)
            if delta > 1e-4 and rel > 1e-6:
                fail(f"{label}: {name}[{index}] cpu={left} gpu={right} abs={delta:.3e} rel={rel:.3e}")
                return
        if qcs_on:
            aq = qcs(cpu, name)
            bq = qcs(gpu, name)
            if aq != bq:
                fail(f"{label}: {name}-QCS cpu={aq} gpu={bq}")
                return
    print(f"OK   {label}")


def main() -> int:
    if not CPU.is_file() or not GPU.is_file():
        print("missing binary", CPU, GPU, file=sys.stderr)
        return 2
    refused = run(CPU, ["--metrics", "pu21-psnr", "--metric-device", "gpu", "--hdr-transfer", "pq"])
    if refused.returncode == 0:
        fail("CPU build accepted --metric-device gpu")
    else:
        print("OK   CPU build still refuses a GPU request")

    cases = []
    for bits, neutral in ((8, 128), (10, 512)):
        for chroma in ("420", "422", "444"):
            for transfer, extra in (
                ("pq", []),
                ("pq", ["--hdr-peak-nits", "1000"]),
                ("hlg", ["--hdr-peak-nits", "1000"]),
                ("hlg", ["--hdr-peak-nits", "4000"]),
            ):
                cases.append((bits, chroma, transfer, extra))

    with tempfile.TemporaryDirectory(prefix="pixvqa_pu21_cuda_") as td:
        root = Path(td)
        for bits, chroma, transfer, extra in cases:
            w = h = 64
            fmt = f"{chroma}p:{bits}"
            neutral = 128 if bits == 8 else 512
            y0 = 180 if bits == 8 else 520
            y1 = 200 if bits == 8 else 620
            ref = root / f"r_{bits}_{chroma}_{transfer}_{len(extra)}.yuv"
            dist = root / f"d_{bits}_{chroma}_{transfer}_{len(extra)}.yuv"
            same = root / f"s_{bits}_{chroma}_{transfer}_{len(extra)}.yuv"
            frames = [(y0, neutral, neutral), (y1, neutral, neutral)]
            write_frame(ref, w, h, chroma, bits, frames)
            write_frame(dist, w, h, chroma, bits, [(y0 + (8 if bits == 8 else 40), neutral, neutral), (y1, neutral + 4, neutral)])
            write_frame(same, w, h, chroma, bits, frames)
            base = [
                "--metrics", "pu21-psnr,pu21-ssim,pu21-msssim",
                "--hdr-transfer", transfer, *extra,
                "-r", str(ref), "-s", f"{w}x{h}", "-f", fmt,
                "-t", str(dist), "-s", f"{w}x{h}", "-f", fmt,
            ]
            label = f"{bits}-bit {chroma} {transfer} {' '.join(extra) or 'default'} distorted"
            for qcs_on in (False, True):
                args = base + (["--qcs"] if qcs_on else [])
                cpu = run(CPU, ["--metric-device", "cpu", *args])
                gpu = run(GPU, ["--metric-device", "gpu", *args])
                if cpu.returncode != 0 or gpu.returncode != 0:
                    fail(f"{label} qcs={qcs_on} rc {cpu.returncode}/{gpu.returncode}\n{cpu.stderr[-300:]}\n{gpu.stderr[-300:]}")
                    continue
                compare(label + (" qcs" if qcs_on else ""), cpu.stdout, gpu.stdout, qcs_on)
            ident_args = [
                "--metrics", "pu21-psnr,pu21-ssim,pu21-msssim",
                "--hdr-transfer", transfer, *extra,
                "-r", str(same), "-s", f"{w}x{h}", "-f", fmt,
                "-t", str(same), "-s", f"{w}x{h}", "-f", fmt,
            ]
            cpu = run(CPU, ["--metric-device", "cpu", *ident_args])
            gpu = run(GPU, ["--metric-device", "gpu", *ident_args])
            if cpu.returncode != 0 or gpu.returncode != 0:
                fail(f"identical {label} rc {cpu.returncode}/{gpu.returncode}")
            else:
                compare("identical " + label, cpu.stdout, gpu.stdout, False)

        # Ordinary PSNR must not move when PU21 shares the GPU.
        ref = root / "psnr_ref.yuv"
        dist = root / "psnr_dist.yuv"
        write_frame(ref, 64, 64, "420", 10, [(400, 512, 512), (500, 512, 512)])
        write_frame(dist, 64, 64, "420", 10, [(420, 512, 512), (480, 520, 500)])
        common = ["--hdr-transfer", "pq", "-r", str(ref), "-s", "64x64", "-f", "420p:10",
                  "-t", str(dist), "-s", "64x64", "-f", "420p:10"]
        alone = run(GPU, ["--metrics", "psnr", "--metric-device", "gpu", *common])
        both = run(GPU, ["--metrics", "psnr,pu21-psnr", "--metric-device", "gpu", *common])
        pa = re.findall(r"\[ PSNR  y:([0-9.]+) u:([0-9.]+) v:([0-9.]+) avg:([0-9.]+) \]", alone.stdout)
        pb = re.findall(r"\[ PSNR  y:([0-9.]+) u:([0-9.]+) v:([0-9.]+) avg:([0-9.]+) \]", both.stdout)
        if alone.returncode != 0 or both.returncode != 0 or pa != pb:
            fail(f"PSNR changed beside PU21 {pa} vs {pb} rc {alone.returncode}/{both.returncode}")
        else:
            print("OK   integer PSNR unchanged beside CUDA PU21")

        # Nonzero HLG black must agree on grey and on saturated RGB. PQ stays in the matrix above.
        for label, ref_codes, dist_codes, extra in (
            ("hlg grey black 0", (480, 512, 512), (560, 512, 512), ["--hdr-black-nits", "0"]),
            ("hlg grey black 5", (480, 512, 512), (560, 512, 512), ["--hdr-black-nits", "5"]),
            ("hlg saturated black 0", (480, 512, 800), (560, 400, 700), ["--hdr-black-nits", "0"]),
            ("hlg saturated black 5", (480, 512, 800), (560, 400, 700), ["--hdr-black-nits", "5"]),
        ):
            ref = root / f"{label.replace(' ', '_')}_r.yuv"
            dist = root / f"{label.replace(' ', '_')}_d.yuv"
            write_frame(ref, 32, 32, "444", 10, [ref_codes])
            write_frame(dist, 32, 32, "444", 10, [dist_codes])
            args = ["--metrics", "pu21-psnr,pu21-ssim,pu21-msssim", "--hdr-transfer", "hlg",
                    "--hdr-peak-nits", "1000", *extra,
                    "-r", str(ref), "-s", "32x32", "-f", "444p:10",
                    "-t", str(dist), "-s", "32x32", "-f", "444p:10"]
            cpu = run(CPU, ["--metric-device", "cpu", *args])
            gpu = run(GPU, ["--metric-device", "gpu", *args])
            if cpu.returncode != 0 or gpu.returncode != 0:
                fail(f"{label} rc {cpu.returncode}/{gpu.returncode}\n{cpu.stderr[-300:]}\n{gpu.stderr[-300:]}")
            else:
                compare(label, cpu.stdout, gpu.stdout, False)

        ffmpeg = Path(os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "")
        swdec = Path(os.environ.get("PIXVQA_SWDEC_BIN", ROOT / "build" / "swdec" / "pixvqa"))
        full = Path(os.environ.get("PIXVQA_FULL_BIN", ROOT / "build" / "full" / "pixvqa"))
        if ffmpeg.is_file() and swdec.is_file() and full.is_file():
            raw = root / "hevc_ref.yuv"
            write_frame(raw, 64, 64, "420", 10, [(480, 512, 512)] * 4)
            hevc = root / "hevc_dist.mp4"
            enc = subprocess.run([
                str(ffmpeg), "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "64x64",
                "-i", str(raw), "-c:v", "libx265", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc", str(hevc),
            ], capture_output=True, text=True)
            if enc.returncode == 0:
                args = ["--metrics", "pu21-psnr,pu21-ssim,pu21-msssim", "--hdr-transfer", "pq",
                        "-r", str(raw), "-s", "64x64", "-f", "420p:10", "-t", str(hevc), "--decode", "cpu"]
                cpu = run(swdec, ["--metric-device", "cpu", *args])
                gpu = run(full, ["--metric-device", "gpu", *args])
                if cpu.returncode != 0 or gpu.returncode != 0:
                    fail(f"hevc rc {cpu.returncode}/{gpu.returncode}\n{gpu.stderr[-400:]}")
                else:
                    compare("hevc software decode", cpu.stdout, gpu.stdout, False)
            else:
                print("SKIP hevc encode", enc.stderr[-200:])
        else:
            print("SKIP hevc; ffmpeg or a decode build is absent")

        def bench(w: int, h: int, frames: int) -> None:
            ref = root / f"bench_r_{w}.yuv"
            dist = root / f"bench_d_{w}.yuv"
            y0 = 480
            write_frame(ref, w, h, "420", 10, [(y0, 512, 512)] * frames)
            write_frame(dist, w, h, "420", 10, [(y0 + 30, 520, 500)] * frames)
            args = ["--metrics", "pu21-psnr,pu21-ssim,pu21-msssim", "--hdr-transfer", "pq", "-v", "0",
                    "-r", str(ref), "-s", f"{w}x{h}", "-f", "420p:10",
                    "-t", str(dist), "-s", f"{w}x{h}", "-f", "420p:10"]
            for kind, binary, device in (("cpu", CPU, "cpu"), ("cuda", GPU, "gpu")):
                run(binary, ["--metric-device", device, *args])
                t0 = time.perf_counter()
                proc = run(binary, ["--metric-device", device, *args])
                dt = time.perf_counter() - t0
                if proc.returncode != 0:
                    fail(f"bench {w}x{h} {kind} rc {proc.returncode}")
                    return
                print(f"BENCH {w}x{h} {frames}f {kind}: {dt:.3f}s  {frames / dt:.2f} fps")

        bench(1920, 1080, 8)
        bench(3840, 2160, 4)

    print("WORST abs", " ".join(f"{name}={WORST[name]:.3e}" for name in NAMES))
    if FAILS:
        print(f"{FAILS} failure(s)", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
