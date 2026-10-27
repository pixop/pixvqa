#!/usr/bin/env python3
"""CPU versus CUDA PSNR and SSIM on 8-bit and 10-bit planar YUV."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "build" / "cpu" / "pixvqa"
CUDA = ROOT / "build" / "cuda" / "pixvqa"
W, H, N = 64, 48, 3


def fail(msg):
    print("FAIL", msg)
    sys.exit(1)


def chroma(fmt, bits):
    if fmt == "420":
        return W // 2, H // 2
    if fmt == "422":
        return W // 2, H
    if fmt == "444":
        return W, H
    raise ValueError(fmt)


def sequence(fmt, bits, distort):
    dt = np.uint8 if bits == 8 else np.uint16
    peak = 255 if bits == 8 else 1023
    yy, xx = np.mgrid[0:H, 0:W]
    y0 = ((xx * 3 + yy * 5) % (peak + 1)).astype(dt)
    cw, ch = chroma(fmt, bits)
    cy, cx = np.mgrid[0:ch, 0:cw]
    u0 = ((cx * 7 + 40) % (peak + 1)).astype(dt)
    v0 = ((cy * 9 + 80) % (peak + 1)).astype(dt)
    chunks = []
    for i in range(N):
        y, u, v = y0.copy(), u0.copy(), v0.copy()
        if distort and i == 1:
            y = np.clip(y.astype(np.int32) + 12, 0, peak).astype(dt)
            u = np.clip(u.astype(np.int32) - 5, 0, peak).astype(dt)
        chunks += [y.tobytes(), u.tobytes(), v.tobytes()]
    return b"".join(chunks)


def lines(text, token):
    return [ln for ln in text.splitlines() if token in ln]


def close_enough(cpu_text, gpu_text):
    if cpu_text == gpu_text:
        return True
    nums_c = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d+|\d+)", cpu_text)]
    nums_g = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d+|\d+)", gpu_text)]
    return len(nums_c) == len(nums_g) and all(abs(a - b) <= 1e-6 for a, b in zip(nums_c, nums_g))


def run(binary, args):
    return subprocess.run([str(binary), *args], capture_output=True, text=True, check=False)


def compare(label, fmt, bits, distort, qcs):
    tag = f"{fmt}p:{bits}"
    td = Path("/tmp/pixvqa-cuda8")
    td.mkdir(exist_ok=True)
    ref = td / f"r_{label}.yuv"
    dist = td / f"d_{label}.yuv"
    ref.write_bytes(sequence(fmt, bits, False))
    dist.write_bytes(sequence(fmt, bits, distort))
    common = ["--metrics", "psnr,ssim", "-v", "1", "-r", str(ref), "-s", f"{W}x{H}", "-f", tag,
              "-t", str(dist), "-s", f"{W}x{H}", "-f", tag]
    if qcs:
        common += ["--qcs"]
    cpu = run(CPU, common + ["--metric-device", "cpu"])
    gpu = run(CUDA, common + ["--metric-device", "gpu"])
    if cpu.returncode != 0 or gpu.returncode != 0:
        fail(f"{label} rc cpu={cpu.returncode} gpu={gpu.returncode} {gpu.stderr[-240:]}")
    if "PSNR: gpu" not in gpu.stdout or "SSIM: gpu" not in gpu.stdout:
        fail(f"{label} did not stay on the GPU")
    for token in ("[ PSNR", "[ SSIM", "PSNR-QCS", "SSIM-QCS"):
        if token.endswith("QCS") and not qcs:
            continue
        cl = lines(cpu.stdout, token)
        gl = lines(gpu.stdout, token)
        if len(cl) != len(gl) or not cl:
            fail(f"{label} {token} line count {len(cl)} vs {len(gl)}")
        for a, b in zip(cl, gl):
            if token == "[ PSNR" and a != b:
                fail(f"{label} PSNR text differs\n{a}\n{b}")
            if token != "[ PSNR" and not close_enough(a, b):
                fail(f"{label} {token} differs\n{a}\n{b}")
    if not distort:
        if "999.99" not in " ".join(lines(gpu.stdout, "[ PSNR")):
            fail(f"{label} identical PSNR was not 999.99")
    print("OK  ", label)


def main():
    if not CPU.is_file() or not CUDA.is_file():
        fail("build/cpu/pixvqa and build/cuda/pixvqa are required")
    for bits in (8, 10):
        for fmt in ("420", "422", "444"):
            for distort, kind in ((False, "identical"), (True, "distorted")):
                for qcs in (False, True):
                    compare(f"{bits}-{fmt}-{kind}-qcs{int(qcs)}", fmt, bits, distort, qcs)
    # Existing CUDA metrics must still match on both depths.
    for bits, fmt in ((8, "420"), (10, "444")):
        tag = f"{fmt}p:{bits}"
        td = Path("/tmp/pixvqa-cuda8")
        ref = td / f"other_r_{bits}.yuv"
        dist = td / f"other_d_{bits}.yuv"
        ref.write_bytes(sequence(fmt, bits, False))
        dist.write_bytes(sequence(fmt, bits, True))
        common = ["--metrics", "msssim,xpsnr", "--fps", "30", "-v", "0", "-r", str(ref), "-s", f"{W}x{H}", "-f", tag,
                  "-t", str(dist), "-s", f"{W}x{H}", "-f", tag]
        cpu = run(CPU, common + ["--metric-device", "cpu"])
        gpu = run(CUDA, common + ["--metric-device", "gpu"])
        if cpu.returncode != 0 or gpu.returncode != 0:
            fail(f"other metrics {bits} rc {cpu.returncode}/{gpu.returncode} {gpu.stderr[-200:]}")
        for token in ("[ MS-SSIM", "[ XPSNR"):
            cl = lines(cpu.stdout, token)
            gl = lines(gpu.stdout, token)
            if len(cl) != 1 or not close_enough(cl[0], gl[0]):
                fail(f"other metrics {bits} {token}\n{cl}\n{gl}")
        print("OK   other", bits, fmt)
    compressed()
    print("All CUDA PSNR/SSIM bit-depth checks passed")


def compressed():
    """Software-decoded 8-bit and 10-bit 4:2:0, then CPU versus GPU PSNR and SSIM."""
    full = ROOT / "build" / "full" / "pixvqa"
    ffmpeg = Path(os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "")
    if not full.is_file() or not ffmpeg.is_file():
        print("SKIP compressed (full binary or ffmpeg is absent)")
        return
    td = Path("/tmp/pixvqa-cuda8")
    for bits, pix in ((8, "yuv420p"), (10, "yuv420p10le")):
        raw = td / f"cmp_{bits}.yuv"
        raw.write_bytes(sequence("420", bits, True))
        enc = td / f"cmp_{bits}.mp4"
        enc_cmd = [str(ffmpeg), "-y", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{W}x{H}", "-r", "25",
                   "-i", str(raw), "-c:v", "libx264" if bits == 8 else "libx265", "-preset", "ultrafast",
                   "-crf", "28", str(enc)]
        made = subprocess.run(enc_cmd, capture_output=True, text=True)
        if made.returncode != 0:
            print(f"SKIP compressed {bits}-bit ({made.stderr.strip().splitlines()[-1][:160]})")
            continue
        tag = f"420p:{bits}"
        common = ["--metrics", "psnr,ssim", "--qcs", "-v", "1", "--decode", "cpu",
                  "-r", str(raw), "-s", f"{W}x{H}", "-f", tag, "-t", str(enc)]
        cpu = run(full, common + ["--metric-device", "cpu"])
        gpu = run(full, common + ["--metric-device", "gpu"])
        if cpu.returncode != 0 or gpu.returncode != 0:
            fail(f"compressed {bits} rc {cpu.returncode}/{gpu.returncode} {gpu.stderr[-240:]}")
        for token in ("[ PSNR", "[ SSIM", "PSNR-QCS", "SSIM-QCS"):
            cl = lines(cpu.stdout, token)
            gl = lines(gpu.stdout, token)
            if not cl or len(cl) != len(gl):
                fail(f"compressed {bits} {token} count {len(cl)} vs {len(gl)}")
            for a, b in zip(cl, gl):
                exact = token == "[ PSNR"
                if (exact and a != b) or (not exact and not close_enough(a, b)):
                    fail(f"compressed {bits} {token} differs\n{a}\n{b}")
        print("OK   compressed", bits)


if __name__ == "__main__":
    main()
