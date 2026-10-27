#!/usr/bin/env python3
"""VMAF CPU fallback when --metric-device gpu is requested.

CPU-build cases always run. CUDA execution cases run only when a full binary
built from this tree is present. A missing CUDA binary is recorded as skipped,
not as a pass.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
FULL = Path(os.environ.get("PIXVQA_FULL", ROOT / "build" / "full" / "pixvqa"))
MODEL = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"
HDR_MODEL = ROOT / "model" / "hdr_vmaf_v1.1.8" / "hdr_vmaf_v1.1.8_3d0h.json"
HDR = [
    "--hdr-transfer", "pq",
    "--hdr-primaries", "bt2020",
    "--hdr-matrix", "bt2020nc",
    "--hdr-range", "limited",
]
WARN = "pixvqa: warning: VMAF CUDA computation is unavailable; using CPU libvmaf."


def run(binary: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary), *args], capture_output=True, text=True)


def write_420p10(path: Path, w: int, h: int, frames: int, quantize: bool) -> None:
    chroma = struct.pack("<H", 512) * ((w // 2) * (h // 2))
    with path.open("wb") as f:
        for i in range(frames):
            samples = []
            for y in range(h):
                for x in range(w):
                    v = 64 + ((x * 3 + y * 5 + i * 17) % 800)
                    if quantize:
                        v = 180
                    samples.append(min(940, v))
            f.write(struct.pack("<" + "H" * len(samples), *samples))
            f.write(chroma)
            f.write(chroma)


def score_line(text: str) -> str:
    for line in text.splitlines():
        if "VMAF mean:" in line:
            return line.strip()
    return ""


def require(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(message)


def check_cpu(binary: Path, ref: Path, dist: Path) -> dict:
    common = [
        "--metrics", "vmaf", "--model", str(MODEL), "-v", "0",
        "-r", str(ref), "-s", "320x240", "-f", "420p:10",
        "-t", str(dist), "-s", "320x240", "-f", "420p:10",
    ]
    cpu = run(binary, ["--metric-device", "cpu", *common])
    auto = run(binary, ["--metric-device", "auto", *common])
    gpu = run(binary, ["--metric-device", "gpu", *common])
    require(cpu.returncode == 0, f"cpu device failed\n{cpu.stderr}")
    require(auto.returncode == 0, f"auto device failed\n{auto.stderr}")
    require(gpu.returncode == 0, f"gpu device failed\n{gpu.stderr}")
    require(WARN not in cpu.stderr and WARN not in auto.stderr, "warning on cpu or auto")
    require(WARN in gpu.stderr, f"missing fallback warning\n{gpu.stderr}")
    require("requested device: cpu" in cpu.stdout, cpu.stdout)
    require("requested device: auto" in auto.stdout, auto.stdout)
    require("requested device: gpu" in gpu.stdout, gpu.stdout)
    for proc in (cpu, auto, gpu):
        require("VMAF: cpu libvmaf" in proc.stdout, proc.stdout)
        require("backend=cpu" in proc.stdout, proc.stdout)
        require("backend=cuda" not in proc.stdout, proc.stdout)
    cpu_score = score_line(cpu.stdout)
    require(cpu_score and cpu_score == score_line(gpu.stdout) == score_line(auto.stdout),
            f"VMAF score changed across devices\n{cpu.stdout}\n{gpu.stdout}\n{auto.stdout}")

    mixed = run(binary, ["--metric-device", "gpu", "--metrics", "xpsnr,vmaf", "--fps", "60",
                         "--model", str(MODEL), "-v", "0",
                         "-r", str(ref), "-s", "320x240", "-f", "420p:10",
                         "-t", str(dist), "-s", "320x240", "-f", "420p:10"])
    require(mixed.returncode != 0, "CPU build accepted XPSNR on --metric-device gpu")
    require("no GPU metrics" in mixed.stderr or "no CPU substitute" in mixed.stderr, mixed.stderr)

    psnr = run(binary, ["--metric-device", "gpu", "--metrics", "psnr,msssim", "-v", "0",
                        "-r", str(ref), "-s", "320x240", "-f", "420p:10",
                        "-t", str(dist), "-s", "320x240", "-f", "420p:10"])
    require(psnr.returncode != 0 and "no GPU metrics" in psnr.stderr, psnr.stderr)

    decode = run(binary, ["--metrics", "vmaf", "--metric-device", "gpu", "--decode", "gpu",
                          "--input", "compressed", "--model", str(MODEL),
                          "-r", str(ref), "-t", str(dist), "-s", "320x240", "-f", "420p:10"])
    require(decode.returncode != 0, "GPU decode of a non-video file was accepted")
    require("no video stream" in decode.stderr or "GPU decoding is not compiled" in decode.stderr,
            decode.stderr)
    require("[ VMAF:" not in decode.stdout, "GPU decode failure still printed a VMAF score")
    return {
        "binary": str(binary),
        "vmaf_score_line": score_line(cpu.stdout),
        "gpu_warning": True,
        "scores_match": True,
        "cpu_build_rejects_gpu_xpsnr": True,
        "cpu_build_rejects_gpu_psnr_msssim": True,
        "cpu_build_rejects_gpu_decode": True,
    }


def check_hdr(binary: Path, ref: Path, dist: Path) -> dict:
    common = [
        "--metrics", "vmaf", "--model", str(HDR_MODEL), *HDR, "-v", "0",
        "-r", str(ref), "-s", "320x240", "-f", "420p:10",
        "-t", str(dist), "-s", "320x240", "-f", "420p:10",
    ]
    cpu = run(binary, ["--metric-device", "cpu", *common])
    auto = run(binary, ["--metric-device", "auto", *common])
    gpu = run(binary, ["--metric-device", "gpu", *common])
    require(cpu.returncode == 0, f"hdr cpu device failed\n{cpu.stderr}")
    require(auto.returncode == 0, f"hdr auto device failed\n{auto.stderr}")
    require(gpu.returncode == 0, f"hdr gpu device failed\n{gpu.stderr}")
    require(WARN not in cpu.stderr and WARN not in auto.stderr, "hdr warning on cpu or auto")
    require(WARN in gpu.stderr, f"hdr missing fallback warning\n{gpu.stderr}")
    require("CUDA HDR" not in gpu.stderr, gpu.stderr)
    for proc in (cpu, auto, gpu):
        require("VMAF: cpu libvmaf" in proc.stdout, proc.stdout)
        require("backend=cpu" in proc.stdout, proc.stdout)
        require("backend=cuda" not in proc.stdout, proc.stdout)
        require("color_ref=limited:bt2020:pq:bt2020nc" in proc.stdout, proc.stdout)
    cpu_score = score_line(cpu.stdout)
    require(cpu_score and cpu_score == score_line(gpu.stdout) == score_line(auto.stdout),
            f"HDR VMAF score changed across devices\n{cpu.stdout}\n{gpu.stdout}")
    mixed = run(binary, ["--metric-device", "gpu", "--metrics", "xpsnr,vmaf", "--fps", "60",
                         "--model", str(HDR_MODEL), *HDR, "-v", "0",
                         "-r", str(ref), "-s", "320x240", "-f", "420p:10",
                         "-t", str(dist), "-s", "320x240", "-f", "420p:10"])
    require(mixed.returncode != 0, "CPU build accepted HDR VMAF + XPSNR on --metric-device gpu")
    require("no GPU metrics" in mixed.stderr or "no CPU substitute" in mixed.stderr, mixed.stderr)
    return {"vmaf_score_line": cpu_score, "gpu_warning": True, "scores_match": True,
            "cpu_build_rejects_gpu_xpsnr": True}


def check_full(binary: Path, ref: Path, dist: Path) -> dict:
    sides = ["-r", str(ref), "-s", "320x240", "-f", "420p:10", "-t", str(dist), "-s", "320x240", "-f", "420p:10"]
    model = ["--model", str(MODEL), "-v", "0"]
    xpsnr = run(binary, ["--metric-device", "gpu", "--metrics", "xpsnr,vmaf", "--fps", "60", *model, *sides])
    both = run(binary, ["--metric-device", "gpu", "--metrics", "psnr,msssim,vmaf", *model, *sides])
    plain = run(binary, ["--metric-device", "gpu", "--metrics", "psnr,msssim", "-v", "0", *sides])
    hidden = os.environ.copy()
    hidden["CUDA_VISIBLE_DEVICES"] = ""
    alone = subprocess.run(
        [str(binary), "--metric-device", "gpu", "--metrics", "vmaf", *model, *sides],
        capture_output=True, text=True, env=hidden,
    )
    out = {
        "binary": str(binary),
        "xpsnr_vmaf_rc": xpsnr.returncode,
        "psnr_msssim_vmaf_rc": both.returncode,
        "psnr_msssim_rc": plain.returncode,
        "no_visible_cuda_vmaf_rc": alone.returncode,
    }
    require(xpsnr.returncode == 0 and WARN in xpsnr.stderr and "XPSNR: gpu" in xpsnr.stdout and "VMAF: cpu libvmaf" in xpsnr.stdout,
            f"xpsnr+vmaf\n{xpsnr.stderr}\n{xpsnr.stdout}")
    require(both.returncode == 0 and WARN in both.stderr and "PSNR: gpu" in both.stdout and "MS-SSIM: gpu" in both.stdout and "VMAF: cpu libvmaf" in both.stdout,
            f"psnr+msssim+vmaf\n{both.stderr}\n{both.stdout}")
    require(plain.returncode == 0 and WARN not in plain.stderr and "PSNR: gpu" in plain.stdout and "VMAF:" not in plain.stdout,
            f"gpu metrics without vmaf\n{plain.stderr}\n{plain.stdout}")
    require(alone.returncode == 0 and WARN in alone.stderr and "VMAF: cpu libvmaf" in alone.stdout,
            f"vmaf with no visible CUDA device\n{alone.stderr}\n{alone.stdout}")
    hdr = ["--model", str(HDR_MODEL), *HDR, "-v", "0"]
    hdr_only = run(binary, ["--metric-device", "gpu", "--metrics", "vmaf", *hdr, *sides])
    hdr_x = run(binary, ["--metric-device", "gpu", "--metrics", "xpsnr,vmaf", "--fps", "60", *hdr, *sides])
    sdr_p = run(binary, ["--metric-device", "gpu", "--metrics", "psnr,vmaf", *model, *sides])
    require(hdr_only.returncode == 0 and WARN in hdr_only.stderr and "VMAF: cpu libvmaf" in hdr_only.stdout
            and "backend=cpu" in hdr_only.stdout and "CUDA HDR" not in hdr_only.stderr,
            f"hdr vmaf gpu\n{hdr_only.stderr}\n{hdr_only.stdout}")
    require(hdr_x.returncode == 0 and WARN in hdr_x.stderr and "XPSNR: gpu" in hdr_x.stdout
            and "VMAF: cpu libvmaf" in hdr_x.stdout,
            f"hdr vmaf + xpsnr\n{hdr_x.stderr}\n{hdr_x.stdout}")
    require(sdr_p.returncode == 0 and WARN in sdr_p.stderr and "PSNR: gpu" in sdr_p.stdout
            and "VMAF: cpu libvmaf" in sdr_p.stdout,
            f"sdr vmaf + psnr\n{sdr_p.stderr}\n{sdr_p.stdout}")
    out["hdr_vmaf_rc"] = hdr_only.returncode
    out["hdr_xpsnr_vmaf_rc"] = hdr_x.returncode
    out["sdr_psnr_vmaf_rc"] = sdr_p.returncode
    out["status"] = "passed"
    return out


def main() -> int:
    if not CPU.is_file():
        print(f"missing {CPU}", file=sys.stderr)
        return 2
    if not MODEL.is_file() or not HDR_MODEL.is_file():
        print(f"missing model {MODEL} or {HDR_MODEL}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "ref.yuv"
        dist = Path(td) / "dist.yuv"
        write_420p10(ref, 320, 240, 3, False)
        write_420p10(dist, 320, 240, 3, True)
        report = {"cpu": check_cpu(CPU, ref, dist), "hdr": check_hdr(CPU, ref, dist), "cuda": None}
        if not FULL.is_file():
            report["cuda"] = {
                "status": "skipped",
                "reason": "build/full/pixvqa is absent, so XPSNR/PSNR/MS-SSIM GPU execution and NVDEC were not run",
            }
            print("SKIP cuda mixed execution: no full binary")
        else:
            report["cuda"] = check_full(FULL, ref, dist)
    out = ROOT / "reports" / "vmaf_device_fallback_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"OK vmaf device fallback score={report['cpu']['vmaf_score_line']} hdr={report['hdr']['vmaf_score_line']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
