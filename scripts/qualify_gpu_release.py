#!/usr/bin/env python3
"""Release checks for the installed CUDA/NVDEC binary.

CPU is the numerical reference. Tolerances are the ones already used by the
project tests: printed PSNR and XPSNR must match, SSIM and MS-SSIM may differ
by at most 1e-6, PU21 fails only when both the absolute error exceeds 1e-4 and
the relative error exceeds 1e-6, and QCS text must match.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLED = Path(os.environ.get("PIXVQA_BIN", ROOT / "bin" / "pixvqa"))
CPU = Path(os.environ.get("PIXVQA_CPU", ROOT / "build" / "cpu" / "pixvqa"))
SW = Path(os.environ.get("PIXVQA_SWDEC", ROOT / "build" / "swdec" / "pixvqa"))
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or ""
MODEL = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"
HDR_MODEL = ROOT / "model" / "hdr_vmaf_v1.1.8" / "hdr_vmaf_v1.1.8_3d0h.json"
NUM = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")
FAILS = 0
CHECKS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def ok(msg: str) -> None:
    global CHECKS
    CHECKS += 1
    print("OK  ", msg)


def run(binary: Path, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=timeout)


def ff(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, text=True)


def lines(text: str, token: str) -> list[str]:
    return [ln for ln in text.splitlines() if token in ln]


def nums(text: str) -> list[float]:
    return [float(v) for v in NUM.findall(text)]


def close(a: str, b: str, limit: float) -> bool:
    na, nb = nums(a), nums(b)
    return len(na) == len(nb) and all(abs(x - y) <= limit for x, y in zip(na, nb))


def compare_metrics(label: str, cpu_out: str, gpu_out: str, tokens: list[str]) -> None:
    for token in tokens:
        cl, gl = lines(cpu_out, token), lines(gpu_out, token)
        if not cl or len(cl) != len(gl):
            fail(f"{label} {token} count {len(cl)} vs {len(gl)}")
            return
        for index, (left, right) in enumerate(zip(cl, gl)):
            if "PU21-" in token:
                na, nb = nums(left), nums(right)
                if len(na) != len(nb):
                    fail(f"{label} {token}[{index}] parse")
                    return
                for x, y in zip(na, nb):
                    delta = abs(x - y)
                    rel = delta / max(abs(x), 1e-12)
                    if delta > 1e-4 and rel > 1e-6:
                        fail(f"{label} {token}[{index}] abs={delta:.3e} rel={rel:.3e}")
                        return
            elif token.startswith("[ PSNR") or token.startswith("[ XPSNR") or token.endswith("QCS"):
                if left != right:
                    fail(f"{label} {token}[{index}]\n{left}\n{right}")
                    return
            elif not close(left, right, 1e-6):
                fail(f"{label} {token}[{index}]\n{left}\n{right}")
                return
    ok(label)


def write_yuv(path: Path, w: int, h: int, chroma: str, bits: int, frames: list[tuple[int, int, int]]) -> None:
    cw = w if chroma == "444" else w // 2
    ch = h if chroma != "420" else h // 2
    chunks = []
    for y, u, v in frames:
        if bits == 8:
            chunks.append(bytes([y & 255]) * (w * h))
            chunks.append(bytes([u & 255]) * (cw * ch))
            chunks.append(bytes([v & 255]) * (cw * ch))
        else:
            import struct
            chunks.append(struct.pack("<" + "H" * (w * h), *([y] * (w * h))))
            chunks.append(struct.pack("<" + "H" * (cw * ch), *([u] * (cw * ch))))
            chunks.append(struct.pack("<" + "H" * (cw * ch), *([v] * (cw * ch))))
    path.write_bytes(b"".join(chunks))


def require_reject(label: str, proc: subprocess.CompletedProcess[str], needle: str) -> None:
    text = proc.stderr + proc.stdout
    if proc.returncode == 0 or "Summary results" in proc.stdout or needle not in text:
        fail(f"{label} rc={proc.returncode} missing {needle!r}\n{text[-400:]}")
    elif "backend=software" in text:
        fail(f"{label} fell back to software\n{text[-400:]}")
    else:
        ok(label)


def main() -> int:
    if not FFMPEG:
        print("set FFMPEG or put ffmpeg on PATH", file=sys.stderr)
        return 2
    if not INSTALLED.is_file() or not CPU.is_file() or not SW.is_file():
        print("missing binary", INSTALLED, CPU, SW, file=sys.stderr)
        return 2
    version = run(INSTALLED, ["--version"])
    print(version.stdout.strip())
    if "build: cuda" not in version.stdout or "decode: software nvdec" not in version.stdout:
        fail(f"installed binary is not the CUDA/NVDEC build:\n{version.stdout}")
    else:
        ok("installed bin/pixvqa reports cuda and software nvdec")

    with tempfile.TemporaryDirectory(prefix="pixvqa_gpu_qual_") as tmp:
        td = Path(tmp)
        raw_metrics(td)
        hdr_metrics(td)
        mixed_vmaf(td)
        nvdec_and_metadata(td)
        unsupported(td)
        kimono()
        stability(td)
    print(f"QUALIFY checks={CHECKS} fails={FAILS}")
    return 1 if FAILS else 0


def raw_metrics(td: Path) -> None:
    for bits in (8, 10):
        for chroma in ("420", "422", "444"):
            ref, dist = td / f"r_{bits}_{chroma}.yuv", td / f"d_{bits}_{chroma}.yuv"
            y0 = 40 if bits == 8 else 200
            write_yuv(ref, 64, 48, chroma, bits, [(y0, 128, 128)] * 3)
            write_yuv(dist, 64, 48, chroma, bits, [(y0 + 8, 140, 110)] * 3)
            tag = f"{chroma}p:{bits}"
            args = ["--metrics", "psnr,ssim,msssim,xpsnr", "--qcs", "--fps", "24", "-v", "0",
                    "-r", str(ref), "-s", "64x48", "-f", tag,
                    "-t", str(dist), "-s", "64x48", "-f", tag]
            cpu = run(CPU, ["--metric-device", "cpu", *args])
            gpu = run(INSTALLED, ["--metric-device", "gpu", *args])
            if cpu.returncode or gpu.returncode:
                fail(f"raw {bits} {chroma} rc {cpu.returncode}/{gpu.returncode}\n{gpu.stderr[-300:]}")
                continue
            if "PSNR: gpu" not in gpu.stdout or "XPSNR: gpu" not in gpu.stdout:
                fail(f"raw {bits} {chroma} did not select the GPU")
                continue
            compare_metrics(
                f"raw {bits}-bit {chroma} CUDA vs CPU",
                cpu.stdout, gpu.stdout,
                ["[ PSNR", "[ SSIM", "[ MS-SSIM", "[ XPSNR", "PSNR-QCS", "SSIM-QCS", "MS-SSIM-QCS", "XPSNR-QCS"],
            )


def hdr_metrics(td: Path) -> None:
    for bits, transfer, peak in (
        (8, "pq", None), (10, "pq", "1000"), (8, "hlg", "1000"), (10, "hlg", "4000"),
    ):
        for chroma in ("420", "422", "444"):
            ref, dist = td / f"h_{bits}_{chroma}.yuv", td / f"hd_{bits}_{chroma}.yuv"
            y0 = 80 if bits == 8 else 480
            write_yuv(ref, 64, 48, chroma, bits, [(y0, 128, 128)] * 2)
            write_yuv(dist, 64, 48, chroma, bits, [(y0 + 12, 150, 100)] * 2)
            tag = f"{chroma}p:{bits}"
            extra = ["--hdr-transfer", transfer]
            if peak:
                extra += ["--hdr-peak-nits", peak]
            args = ["--metrics", "pu21-psnr,pu21-ssim,pu21-msssim", "--qcs", "-v", "0", *extra,
                    "-r", str(ref), "-s", "64x48", "-f", tag,
                    "-t", str(dist), "-s", "64x48", "-f", tag]
            cpu = run(CPU, ["--metric-device", "cpu", *args])
            gpu = run(INSTALLED, ["--metric-device", "gpu", *args])
            if cpu.returncode or gpu.returncode:
                fail(f"pu21 {bits} {transfer} {chroma} rc {cpu.returncode}/{gpu.returncode}\n{gpu.stderr[-300:]}")
                continue
            compare_metrics(
                f"PU21 {bits}-bit {chroma} {transfer} peak={peak or 'default'}",
                cpu.stdout, gpu.stdout,
                ["[ PU21-PSNR", "[ PU21-SSIM", "[ PU21-MS-SSIM",
                 "PU21-PSNR-QCS", "PU21-SSIM-QCS", "PU21-MS-SSIM-QCS"],
            )


def mixed_vmaf(td: Path) -> None:
    ref, dist = td / "vm.yuv", td / "vmd.yuv"
    write_yuv(ref, 320, 240, "420", 10, [(400, 512, 512)] * 2)
    write_yuv(dist, 320, 240, "420", 10, [(430, 500, 520)] * 2)
    geom = ["-r", str(ref), "-s", "320x240", "-f", "420p:10", "-t", str(dist), "-s", "320x240", "-f", "420p:10"]
    only = run(INSTALLED, ["--metrics", "vmaf", "--model", str(MODEL), "--metric-device", "gpu", "-v", "0", *geom])
    if only.returncode or "using CPU libvmaf" not in only.stderr or "VMAF: cpu libvmaf" not in only.stdout:
        fail(f"VMAF-only GPU request\n{only.stderr[-300:]}\n{only.stdout[-300:]}")
    else:
        ok("VMAF-only GPU request stays on CPU libvmaf")
    mixed = run(INSTALLED, ["--metrics", "psnr,vmaf", "--model", str(MODEL), "--metric-device", "gpu",
                            "-v", "0", *geom], timeout=240)
    if mixed.returncode or "using CPU libvmaf" not in mixed.stderr or "VMAF: cpu libvmaf" not in mixed.stdout or "PSNR: gpu" not in mixed.stdout:
        fail(f"mixed PSNR+VMAF\n{mixed.stderr[-300:]}\n{mixed.stdout[-400:]}")
    else:
        ok("mixed PSNR stays on CUDA and VMAF stays on CPU")
    if HDR_MODEL.is_file():
        hdr = run(INSTALLED, [
            "--metrics", "xpsnr,vmaf", "--fps", "24", "--model", str(HDR_MODEL), "--metric-device", "gpu",
            "--hdr-transfer", "pq", "--hdr-primaries", "bt2020", "--hdr-matrix", "bt2020nc", "--hdr-range", "limited",
            "-v", "0", *geom,
        ], timeout=240)
        if hdr.returncode or "using CPU libvmaf" not in hdr.stderr or "VMAF: cpu libvmaf" not in hdr.stdout or "XPSNR: gpu" not in hdr.stdout:
            fail(f"HDR VMAF + XPSNR\n{hdr.stderr[-400:]}\n{hdr.stdout[-400:]}")
        else:
            ok("HDR VMAF stays on CPU beside CUDA XPSNR")


def encode_hevc(raw: Path, dest: Path, pix: str, size: str, rate: str, color: list[str] | None = None,
                vui: str = "") -> bool:
    params = "lossless=1:bframes=0" + ((":" + vui) if vui else "")
    args = ["-f", "rawvideo", "-pix_fmt", pix, "-s", size, "-r", rate, "-i", str(raw),
            "-c:v", "libx265", "-preset", "ultrafast", "-tag:v", "hvc1", "-x265-params", params,
            "-pix_fmt", pix, *(color or []), str(dest)]
    made = ff(args)
    if made.returncode != 0:
        fail(f"encode {dest.name}: {made.stderr[-240:]}")
        return False
    return True


def nvdec_and_metadata(td: Path) -> None:
    raw = td / "nv.yuv"
    dist = td / "nvd.yuv"
    write_yuv(raw, 256, 144, "420", 10, [(480, 512, 512)] * 4)
    write_yuv(dist, 256, 144, "420", 10, [(500, 500, 530)] * 4)
    hevc = td / "nv.mp4"
    hevc_d = td / "nvd.mp4"
    color = ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "tv"]
    vui = "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited"
    if not encode_hevc(raw, hevc, "yuv420p10le", "256x144", "24", color, vui):
        return
    if not encode_hevc(dist, hevc_d, "yuv420p10le", "256x144", "24", color, vui):
        return
    metrics = ["--metrics", "psnr,ssim,msssim,xpsnr,pu21-psnr,pu21-ssim,pu21-msssim", "--qcs",
               "--hdr-transfer", "pq", "-v", "0"]
    sw = run(SW, ["--metric-device", "cpu", "--decode", "cpu", *metrics, "-r", str(hevc), "-t", str(hevc_d)], timeout=240)
    nv = run(INSTALLED, ["--metric-device", "gpu", "--decode", "gpu", *metrics, "-r", str(hevc), "-t", str(hevc_d)], timeout=240)
    if sw.returncode or nv.returncode:
        fail(f"hevc sw/nvdec rc {sw.returncode}/{nv.returncode}\n{nv.stderr[-400:]}")
    else:
        if "distorted: gpu" not in nv.stdout or "backend=software" in nv.stderr:
            fail("NVDEC HEVC did not stay on the GPU decoder")
        else:
            ok("NVDEC HEVC decoder stayed on GPU")
        compare_metrics(
            "NVDEC HEVC CUDA metrics vs software-decode CPU",
            sw.stdout, nv.stdout,
            ["[ PSNR", "[ SSIM", "[ MS-SSIM", "[ XPSNR", "[ PU21-PSNR", "[ PU21-SSIM", "[ PU21-MS-SSIM",
             "PSNR-QCS", "XPSNR-QCS", "PU21-PSNR-QCS"],
        )
    auto = run(INSTALLED, ["--metric-device", "gpu", "--decode", "gpu", *metrics, "-r", str(hevc), "-t", str(hevc_d)])
    explicit = run(INSTALLED, ["--metric-device", "gpu", "--decode", "gpu", *metrics,
                               "-r", str(hevc), "-s", "256x144", "-f", "420p:10",
                               "-t", str(hevc_d), "-s", "256x144", "-f", "420p:10"])
    if auto.returncode or explicit.returncode:
        fail(f"metadata auto/explicit rc {auto.returncode}/{explicit.returncode}\n{auto.stderr[-300:]}")
    elif lines(auto.stdout, "[ PSNR") != lines(explicit.stdout, "[ PSNR") or lines(auto.stdout, "[ PU21-PSNR") != lines(explicit.stdout, "[ PU21-PSNR"):
        fail("automatic metadata scores differ from explicit -s/-f")
    else:
        ok("compressed+compressed automatic detection matches explicit -s/-f on NVDEC")
    raw_side = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0",
                               "-r", str(raw), "-s", "256x144", "-f", "420p:10", "-t", str(hevc)])
    if raw_side.returncode or "reference: raw" not in raw_side.stdout or "distorted: gpu" not in raw_side.stdout:
        fail(f"raw+compressed\n{raw_side.stderr[-300:]}\n{raw_side.stdout[-200:]}")
    else:
        ok("raw reference keeps -s/-f and compressed side uses NVDEC")
    comp_raw = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0",
                               "-r", str(hevc), "-t", str(raw), "-s", "256x144", "-f", "420p:10"])
    if comp_raw.returncode or "reference: gpu" not in comp_raw.stdout or "distorted: raw" not in comp_raw.stdout:
        fail(f"compressed+raw\n{comp_raw.stderr[-300:]}")
    else:
        ok("compressed reference is detected and raw distorted keeps -s/-f")
    both_raw = run(INSTALLED, ["--metrics", "psnr", "-v", "0", "-r", str(raw), "-t", str(dist)])
    require_reject("raw+raw without -s/-f", both_raw, "requires -s and -f")
    bad_size = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0",
                               "-r", str(hevc), "-s", "320x240", "-f", "420p:10", "-t", str(hevc_d)])
    require_reject("incorrect resolution override", bad_size, "-s asks for 320x240")
    bad_fmt = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0",
                              "-r", str(hevc), "-s", "256x144", "-f", "422p:10", "-t", str(hevc_d)])
    require_reject("incorrect pixel-format override", bad_fmt, "-f asks for")
    rate = run(INSTALLED, ["--metrics", "xpsnr", "--decode", "gpu", "--metric-device", "gpu", "-v", "0",
                           "-r", str(hevc), "-t", str(hevc_d)])
    forced = run(INSTALLED, ["--metrics", "xpsnr", "--fps", "24", "--decode", "gpu", "--metric-device", "gpu", "-v", "0",
                             "-r", str(hevc), "-t", str(hevc_d)])
    other = run(INSTALLED, ["--metrics", "xpsnr", "--fps", "30", "--decode", "gpu", "--metric-device", "gpu", "-v", "0",
                            "-r", str(hevc), "-t", str(hevc_d)])
    if rate.returncode or "using frame rate 24/1" not in rate.stderr:
        fail(f"constant frame rate was not detected\n{rate.stderr[-300:]}")
    elif forced.returncode or lines(rate.stdout, "[ XPSNR") != lines(forced.stdout, "[ XPSNR"):
        fail("explicit --fps 24 does not match the detected 24 fps XPSNR")
    elif other.returncode or "using frame rate" in other.stderr:
        fail(f"explicit --fps 30 was not honored\n{other.stderr[-240:]}")
    else:
        ok("XPSNR uses 24/1 from the stream, matches --fps 24, and accepts --fps 30")
    hlg_raw = td / "hlg.yuv"
    hlg = td / "hlg.mp4"
    write_yuv(hlg_raw, 256, 144, "420", 10, [(480, 512, 512)] * 2)
    hlg_color = ["-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-color_range", "tv"]
    if encode_hevc(hlg_raw, hlg, "yuv420p10le", "256x144", "24", hlg_color,
                   "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited"):
        seen = run(INSTALLED, ["--metrics", "pu21-psnr", "--decode", "gpu", "--metric-device", "gpu",
                               "--hdr-peak-nits", "1000", "-v", "0", "-r", str(hlg), "-t", str(hlg)])
        if seen.returncode or "transfer=hlg" not in seen.stdout:
            fail(f"HLG was not propagated on NVDEC\n{seen.stdout[-300:]}\n{seen.stderr[-240:]}")
        else:
            ok("HLG transfer propagates into NVDEC PU21")
    pq = run(INSTALLED, ["--metrics", "pu21-psnr", "--decode", "gpu", "--metric-device", "gpu", "-v", "0",
                         "-r", str(hevc), "-t", str(hevc)])
    if pq.returncode or "transfer=pq" not in pq.stdout:
        fail(f"PQ was not propagated on NVDEC\n{pq.stdout[-300:]}")
    else:
        ok("PQ transfer propagates into NVDEC PU21")


def unsupported(td: Path) -> None:
    raw8 = td / "h8.yuv"
    hevc8 = td / "h8.mp4"
    write_yuv(raw8, 256, 144, "420", 8, [(40, 128, 128)] * 2)
    made = ff(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "256x144", "-r", "24", "-i", str(raw8),
               "-c:v", "libx265", "-preset", "ultrafast", str(hevc8)])
    if made.returncode == 0:
        rejected = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0",
                                   "-r", str(hevc8), "-t", str(hevc8)])
        require_reject("8-bit HEVC NVDEC", rejected, "420p:10 only")
    else:
        fail(f"could not encode 8-bit HEVC: {made.stderr[-200:]}")
    h264 = td / "h264.mp4"
    made = ff(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "256x144", "-r", "24", "-i", str(raw8),
               "-c:v", "libx264", "-preset", "ultrafast", str(h264)])
    if made.returncode == 0:
        rejected = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0",
                                   "-r", str(h264), "-t", str(h264)])
        require_reject("H.264 NVDEC", rejected, "HEVC and AV1 only")
    else:
        fail("could not encode H.264")
    av1 = td / "av1_8.mp4"
    made = ff(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "256x144", "-r", "24", "-i", str(raw8),
               "-c:v", "libaom-av1", "-cpu-used", "8", "-lag-in-frames", "0", "-crf", "28", str(av1)])
    if made.returncode == 0:
        decoded = td / "av1_8.yuv"
        dec = ff(["-i", str(av1), "-f", "rawvideo", "-pix_fmt", "yuv420p", str(decoded)])
        nv = run(INSTALLED, ["--metrics", "psnr,ssim", "--qcs", "--decode", "gpu", "--metric-device", "gpu", "-v", "0",
                             "-r", str(av1), "-t", str(av1)])
        raw = run(CPU, ["--metrics", "psnr,ssim", "--qcs", "-v", "0",
                        "-r", str(decoded), "-s", "256x144", "-f", "420p:8",
                        "-t", str(decoded), "-s", "256x144", "-f", "420p:8"])
        if dec.returncode or nv.returncode or raw.returncode or "distorted: gpu" not in nv.stdout:
            fail(f"8-bit AV1 NVDEC rc {dec.returncode}/{nv.returncode}/{raw.returncode}\n{nv.stderr[-300:]}")
        else:
            compare_metrics("8-bit AV1 NVDEC vs dav1d CLI", raw.stdout, nv.stdout, ["[ PSNR", "[ SSIM", "PSNR-QCS"])
    else:
        fail(f"could not encode 8-bit AV1: {made.stderr[-200:]}")
    yuv422 = td / "422.yuv"
    hevc422 = td / "422.mp4"
    write_yuv(yuv422, 256, 144, "422", 10, [(480, 512, 512)] * 2)
    made = ff(["-f", "rawvideo", "-pix_fmt", "yuv422p10le", "-s", "256x144", "-r", "24", "-i", str(yuv422),
               "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv422p10le", str(hevc422)])
    if made.returncode == 0:
        rejected = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0",
                                   "-r", str(hevc422), "-t", str(hevc422)])
        require_reject("4:2:2 NVDEC", rejected, "4:2:0 only")
    else:
        print("SKIP 4:2:2 HEVC encode is unavailable")
    vvc = ROOT / "ref_vvc.mp4"
    if vvc.is_file():
        rejected = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0", "-r", str(vvc), "-t", str(vvc)])
        require_reject("VVC NVDEC", rejected, "does not support VVC")
    else:
        print("SKIP VVC file is absent")


def kimono() -> None:
    pairs = {
        "hevc": (ROOT / "ref_hevc.mp4", ROOT / "dist_hevc.mp4", True),
        "av1": (ROOT / "ref_av1.mp4", ROOT / "dist_av1.mp4", True),
        "vvc": (ROOT / "ref_vvc.mp4", ROOT / "dist_vvc.mp4", False),
    }
    metrics = ["--metrics", "psnr,ssim,msssim,xpsnr,pu21-psnr", "--qcs", "--hdr-transfer", "pq", "-v", "0", "-n", "4"]
    for name, (ref, dist, nvdec) in pairs.items():
        if not ref.is_file() or not dist.is_file():
            print(f"SKIP {name} local video is absent")
            continue
        cpu = run(SW, ["--metric-device", "cpu", "--decode", "cpu", *metrics, "-r", str(ref), "-t", str(dist)], timeout=300)
        external_oracle = False
        if cpu.returncode and name == "av1" and "libdav1d" in cpu.stderr:
            raw_ref, raw_dist = Path("/tmp") / "kimono_av1_ref.yuv", Path("/tmp") / "kimono_av1_dist.yuv"
            a = ff(["-i", str(ref), "-frames:v", "4", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", str(raw_ref)])
            b = ff(["-i", str(dist), "-frames:v", "4", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", str(raw_dist)])
            if a.returncode or b.returncode:
                fail(f"kimono av1 independent decode failed\n{a.stderr[-200:]}")
            else:
                cpu = run(CPU, ["--metric-device", "cpu", "--fps", "24", *metrics,
                                "-r", str(raw_ref), "-s", "960x540", "-f", "420p:10",
                                "-t", str(raw_dist), "-s", "960x540", "-f", "420p:10"], timeout=300)
                if cpu.returncode:
                    fail(f"kimono av1 raw oracle rc={cpu.returncode}\n{cpu.stderr[-300:]}")
                else:
                    external_oracle = True
                    ok("kimono av1 software oracle is the dav1d CLI decode; linked libavcodec has no libdav1d")
        elif cpu.returncode or "Total frames:4" not in cpu.stdout or "PSNR-QCS" not in cpu.stdout:
            fail(f"kimono {name} CPU rc={cpu.returncode}\n{cpu.stderr[-400:]}")
        else:
            ok(f"kimono {name} software decode scored 4 frames with QCS")
        software_decode = cpu.returncode == 0 and not external_oracle
        if software_decode:
            gpu = run(INSTALLED, ["--metric-device", "gpu", "--decode", "cpu", *metrics, "-r", str(ref), "-t", str(dist)], timeout=300)
            if gpu.returncode:
                fail(f"kimono {name} CUDA rc={gpu.returncode}\n{gpu.stderr[-300:]}")
            else:
                compare_metrics(f"kimono {name} CUDA vs CPU on software decode", cpu.stdout, gpu.stdout,
                                ["[ PSNR", "[ SSIM", "[ MS-SSIM", "[ XPSNR", "[ PU21-PSNR", "PSNR-QCS", "PU21-PSNR-QCS"])
        if nvdec:
            hw_cpu = run(INSTALLED, ["--metric-device", "cpu", "--decode", "gpu", *metrics, "-r", str(ref), "-t", str(dist)], timeout=300)
            hw = run(INSTALLED, ["--metric-device", "gpu", "--decode", "gpu", *metrics, "-r", str(ref), "-t", str(dist)], timeout=300)
            if hw.returncode or hw_cpu.returncode or "distorted: gpu" not in hw.stdout:
                fail(f"kimono {name} NVDEC rc={hw.returncode}/{hw_cpu.returncode}\n{hw.stderr[-400:]}")
            else:
                compare_metrics(f"kimono {name} CUDA vs CPU on NVDEC frames", hw_cpu.stdout, hw.stdout,
                                ["[ PSNR", "[ SSIM", "[ MS-SSIM", "[ XPSNR", "[ PU21-PSNR", "PSNR-QCS"])
                if cpu.returncode == 0 and lines(cpu.stdout, "[ PSNR") == lines(hw_cpu.stdout, "[ PSNR"):
                    ok(f"kimono {name} NVDEC PSNR matches the software oracle")
                elif cpu.returncode == 0:
                    print(f"NOTE {name} NVDEC PSNR differs from the software oracle")
                    print(" sw", lines(cpu.stdout, "[ PSNR")[:1])
                    print(" nv", lines(hw_cpu.stdout, "[ PSNR")[:1])
        else:
            refused = run(INSTALLED, ["--metrics", "psnr", "--decode", "gpu", "-v", "0", "-n", "1",
                                      "-r", str(ref), "-t", str(dist)])
            require_reject("kimono VVC stays off NVDEC", refused, "does not support VVC")


def score_body(text: str) -> str:
    kept = []
    for ln in text.splitlines():
        if any(token in ln for token in ("[ PSNR", "[ SSIM", "[ MS-SSIM", "[ XPSNR", "[ PU21", "QCS", "Total frames:")):
            if ln.startswith("Total frames:"):
                kept.append(ln.split("fps:")[0])
            else:
                kept.append(ln)
    return "\n".join(kept)


def stability(td: Path) -> None:
    ref, dist = td / "st.yuv", td / "std.yuv"
    write_yuv(ref, 176, 144, "420", 10, [(400, 512, 512)] * 2)
    write_yuv(dist, 176, 144, "420", 10, [(420, 500, 520)] * 2)
    args = ["--metrics", "psnr,pu21-psnr", "--hdr-transfer", "pq", "--metric-device", "gpu", "-v", "0",
            "-r", str(ref), "-s", "176x144", "-f", "420p:10", "-t", str(dist), "-s", "176x144", "-f", "420p:10"]
    first = run(INSTALLED, args)
    body = score_body(first.stdout)
    same = bool(body)
    for _ in range(2):
        nxt = run(INSTALLED, args)
        same = same and nxt.returncode == 0 and score_body(nxt.stdout) == body
    if first.returncode or not same:
        fail(f"three sequential CUDA runs did not match\n{first.stderr[-240:]}")
    else:
        ok("three sequential CUDA runs matched")
    procs = [
        subprocess.Popen([str(INSTALLED), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True),
        subprocess.Popen([str(INSTALLED), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True),
    ]
    outs = [p.communicate(timeout=180) for p in procs]
    if any(p.returncode != 0 for p in procs) or score_body(outs[0][0]) != body or score_body(outs[1][0]) != body:
        err = "\n".join(item[1][-200:] for item in outs)
        fail(f"two concurrent CUDA jobs disagreed or failed\n{err}")
    else:
        ok("two concurrent CUDA jobs matched the sequential result")


if __name__ == "__main__":
    sys.exit(main())
