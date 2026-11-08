#!/usr/bin/env python3
"""Reproducible HDR chroma-isolation test for the native PU21 metrics and CVVDP.

The 256x256 reference is neutral BT.2020 PQ 4:4:4.  Each test tile uses a
different valid BT.2020 Y'CbCr triplet selected to match the reference's
*display-linear* BT.2020 luminance after 10-bit limited-range quantization.
It intentionally changes chromaticity, not the quality-metric implementations.

The test invokes native metrics on the uncompressed YUV444P10 inputs.  CVVDP's
raw path intentionally accepts only YUV420P10, so the same pixels are also
losslessly encoded as tagged HEVC 4:4:4 before its run.  The script checks that
decoding the HEVC files reproduces the input bytes exactly.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIXVQA = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "full" / "pixvqa"))
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or ""
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or ""
W = H = 256
BITS = 10
KR, KG, KB = 0.2627, 0.6780, 0.0593
PU_STRUCTURAL_TOLERANCE = 5e-9
LUMA_ABS_TOLERANCE_NITS = 0.026
LUMA_REL_TOLERANCE = 1.4e-4
FAILS = 0

# (neutral Y' code, chromatic Y' code, Cb code, Cr code).  These were chosen
# by searching legal limited-range 10-bit code triplets for low post-PQ
# display-linear luminance error and meaningful chromatic separation.
TILES = (
    (410, 341, 664, 448),
    (450, 289, 664, 688),
    (490, 461, 376, 448),
    (530, 507, 352, 472),
    (570, 495, 424, 640),
    (610, 513, 688, 400),
    (650, 617, 376, 592),
    (690, 677, 352, 520),
)


def fail(message: str) -> None:
    global FAILS
    FAILS += 1
    print(f"FAIL {message}", file=sys.stderr)


def run(command: list[str], *, env: dict[str, str] | None = None,
        timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)


def pq_eotf(code: float) -> float:
    """ST 2084 EOTF from normalized non-linear code to cd/m2."""
    code = min(max(code, 0.0), 1.0)
    if code == 0.0:
        return 0.0
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    p = code ** (1.0 / m2)
    return 10000.0 * (max(p - c1, 0.0) / (c2 - c3 * p)) ** (1.0 / m1)


def ycbcr_to_pq_rgb(y_code: int, cb_code: int, cr_code: int) -> tuple[float, float, float]:
    """BT.2020 NCL limited-range 10-bit Y'CbCr to legal PQ RGB codes."""
    y = (y_code - 64.0) / 876.0
    cb = (cb_code - 512.0) / 896.0
    cr = (cr_code - 512.0) / 896.0
    r = y + 2.0 * (1.0 - KR) * cr
    b = y + 2.0 * (1.0 - KB) * cb
    g = (y - KR * r - KB * b) / KG
    return r, g, b


def display_luminance(y_code: int, cb_code: int, cr_code: int) -> float:
    rgb = ycbcr_to_pq_rgb(y_code, cb_code, cr_code)
    return KR * pq_eotf(rgb[0]) + KG * pq_eotf(rgb[1]) + KB * pq_eotf(rgb[2])


def validate_tiles() -> dict[str, float]:
    max_abs = 0.0
    max_rel = 0.0
    min_component = 1.0
    max_component = 0.0
    min_chroma_spread = 0.0
    for ref_y, dist_y, cb, cr in TILES:
        rgb = ycbcr_to_pq_rgb(dist_y, cb, cr)
        min_component = min(min_component, *rgb)
        max_component = max(max_component, *rgb)
        min_chroma_spread = max(min_chroma_spread, max(rgb) - min(rgb))
        ref_luma = display_luminance(ref_y, 512, 512)
        dist_luma = display_luminance(dist_y, cb, cr)
        delta = abs(dist_luma - ref_luma)
        max_abs = max(max_abs, delta)
        max_rel = max(max_rel, delta / ref_luma)
    if min_component < 0.0 or max_component > 1.0:
        fail(f"generated an illegal PQ RGB signal [{min_component}, {max_component}]")
    if max_abs > LUMA_ABS_TOLERANCE_NITS or max_rel > LUMA_REL_TOLERANCE:
        fail(f"post-quantization luminance mismatch abs={max_abs} rel={max_rel}")
    if min_chroma_spread < 0.3:
        fail(f"chromaticity change was too small (maximum RGB-code spread {min_chroma_spread})")
    return {
        "max_display_luminance_abs_error_nits": max_abs,
        "max_display_luminance_relative_error": max_rel,
        "rgb_code_min": min_component,
        "rgb_code_max": max_component,
        "max_rgb_code_spread": min_chroma_spread,
    }


def tile_for(x: int, y: int) -> tuple[int, int, int, int]:
    return TILES[((y // 32) + (x // 32)) % len(TILES)]


def write_yuv444(path: Path, distorted: bool) -> None:
    y_plane: list[int] = []
    u_plane: list[int] = []
    v_plane: list[int] = []
    for y in range(H):
        for x in range(W):
            ref_y, dist_y, cb, cr = tile_for(x, y)
            y_plane.append(dist_y if distorted else ref_y)
            u_plane.append(cb if distorted else 512)
            v_plane.append(cr if distorted else 512)
    path.write_bytes(struct.pack("<" + "H" * (W * H * 3), *(y_plane + u_plane + v_plane)))


def require_tool(path: str, name: str) -> str:
    if not path:
        raise RuntimeError(f"{name} is required; set {name.upper()} or add it to PATH")
    return path


def encode_lossless_pq(raw: Path, output: Path) -> None:
    """HEVC 4:4:4 lossless, with HEVC VUI and container PQ/BT.2020 tags."""
    command = [
        require_tool(FFMPEG, "ffmpeg"), "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "yuv444p10le", "-video_size", f"{W}x{H}",
        "-framerate", "60", "-i", str(raw), "-frames:v", "1",
        # setparams annotates the raw source without a YUV-to-YUV conversion.
        # Output color flags here would cause swscale to convert an otherwise
        # unknown raw input and defeat the lossless round-trip check.
        "-vf", "setparams=range=limited:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
        "-c:v", "libx265",
        "-x265-params", "lossless=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
        "-pix_fmt", "yuv444p10le", "-movflags", "+write_colr", str(output),
    ]
    result = run(command)
    if result.returncode:
        raise RuntimeError(f"lossless HEVC encode failed: {result.stderr[-500:]}")


def verify_lossless(source: Path, encoded: Path) -> None:
    command = [
        require_tool(FFMPEG, "ffmpeg"), "-loglevel", "error", "-i", str(encoded),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv444p10le", "-",
    ]
    binary = subprocess.run(command, capture_output=True, timeout=600)
    if binary.returncode or binary.stdout != source.read_bytes():
        raise RuntimeError("lossless HEVC round-trip did not preserve the raw YUV444P10 samples")


def verify_tags(encoded: Path) -> None:
    probe = run([
        require_tool(FFPROBE, "ffprobe"), "-v", "error",
        "-show_entries", "stream=pix_fmt,color_range,color_space,color_transfer,color_primaries",
        "-of", "default=nw=1", str(encoded),
    ])
    if probe.returncode:
        raise RuntimeError(f"ffprobe failed: {probe.stderr[-500:]}")
    for tag in ("pix_fmt=yuv444p10le", "color_range=tv", "color_space=bt2020nc",
                "color_transfer=smpte2084", "color_primaries=bt2020"):
        if tag not in probe.stdout:
            raise RuntimeError(f"lossless HEVC is missing {tag}: {probe.stdout}")


def parse_native(stdout: str, stderr: str) -> dict[str, object]:
    conventional = {}
    for name in ("PSNR", "SSIM", "MS-SSIM"):
        match = re.search(rf"\[ {re.escape(name)}\s+y:([0-9.]+) u:([0-9.]+) v:([0-9.]+) avg:([0-9.]+) \]", stdout)
        if not match:
            raise RuntimeError(f"missing conventional {name}: {stdout[-1000:]}")
        conventional[name] = dict(zip(("y", "u", "v", "avg"), map(float, match.groups())))
    psnr = re.search(r"\[ PU21-PSNR r:([0-9.]+) g:([0-9.]+) b:([0-9.]+) avg:([0-9.]+) \]", stdout)
    if not psnr:
        raise RuntimeError(f"missing PU21-PSNR: {stdout[-1000:]}")
    structural: dict[str, float] = {}
    for name in ("PU21-SSIM", "PU21-MS-SSIM"):
        match = re.search(rf"\[ {re.escape(name)} y:([0-9.]+) u:n/a v:n/a avg:([0-9.]+) \]", stdout)
        if not match or match.group(1) != match.group(2):
            raise RuntimeError(f"{name} did not report equal luminance y/avg and n/a chroma")
        structural[name] = float(match.group(1))
    trace = dict(re.findall(r"qcs_sample (pu21-(?:psnr|ssim|msssim)) ([0-9.eE+-]+)", stderr))
    if set(trace) != {"pu21-psnr", "pu21-ssim", "pu21-msssim"}:
        raise RuntimeError(f"missing PU21 trace: {stderr[-1000:]}")
    return {
        "conventional": conventional,
        "pu21_psnr": dict(zip(("r", "g", "b", "avg"), map(float, psnr.groups()))),
        "pu21_structural_printed": structural,
        "pu21_structural_precise": {key: float(value) for key, value in trace.items()},
    }


def score_native(ref: Path, dist: Path) -> dict[str, object]:
    env = dict(os.environ, PIXVQA_QCS_TRACE="1")
    result = run([
        str(PIXVQA), "--metrics", "psnr,ssim,msssim,pu21-psnr,pu21-ssim,pu21-msssim",
        "--device", "cpu", "--qcs", "--hdr-transfer", "pq", "--hdr-primaries", "bt2020",
        "--hdr-matrix", "bt2020nc", "--hdr-range", "limited", "-v", "0",
        "-r", str(ref), "-s", f"{W}x{H}", "-f", "444p:10",
        "-t", str(dist), "-s", f"{W}x{H}", "-f", "444p:10",
    ], env=env)
    if result.returncode:
        raise RuntimeError(f"native score failed: {result.stderr[-1000:]}")
    return parse_native(result.stdout, result.stderr)


def score_cvvdp(ref: Path, dist: Path) -> dict[str, object]:
    result = run([
        str(PIXVQA), "--metrics", "cvvdp", "--device", "gpu", "--cvvdp-python", sys.executable,
        "--cvvdp-interpretation", "pq", "--cvvdp-display", "standard_hdr_pq",
        "-r", str(ref), "-t", str(dist), "-n", "1",
    ], timeout=900)
    if result.returncode:
        raise RuntimeError(f"CVVDP score failed: {result.stderr[-1000:]}")
    try:
        record = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"CVVDP did not emit JSON: {result.stdout[-1000:]}") from exc
    if record.get("metric") != "cvvdp" or record.get("device") != "cuda":
        raise RuntimeError(f"unexpected CVVDP record: {record}")
    return {"jod": float(record["jod"]), "device": record["device"], "elapsed_seconds": record["elapsed_seconds"]}


def main() -> int:
    if not PIXVQA.is_file():
        print(f"missing {PIXVQA}", file=sys.stderr)
        return 2
    try:
        luma = validate_tiles()
        with tempfile.TemporaryDirectory(prefix="pixvqa_pu21_chroma_") as directory:
            work = Path(directory)
            ref_raw, dist_raw = work / "reference.yuv", work / "distorted.yuv"
            write_yuv444(ref_raw, distorted=False)
            write_yuv444(dist_raw, distorted=True)
            native = score_native(ref_raw, dist_raw)
            ref_video, dist_video = work / "reference.mp4", work / "distorted.mp4"
            encode_lossless_pq(ref_raw, ref_video)
            encode_lossless_pq(dist_raw, dist_video)
            verify_lossless(ref_raw, ref_video)
            verify_lossless(dist_raw, dist_video)
            verify_tags(ref_video)
            verify_tags(dist_video)
            cvvdp = score_cvvdp(ref_video, dist_video)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        fail(str(exc))
        return 1

    exact = native["pu21_structural_precise"]
    ssim_error = 1.0 - exact["pu21-ssim"]
    msssim_error = 1.0 - exact["pu21-msssim"]
    if ssim_error < 0 or ssim_error > PU_STRUCTURAL_TOLERANCE:
        fail(f"PU21-SSIM={exact['pu21-ssim']:.17g}, error={ssim_error:.3e}")
    if msssim_error < 0 or msssim_error > PU_STRUCTURAL_TOLERANCE:
        fail(f"PU21-MS-SSIM={exact['pu21-msssim']:.17g}, error={msssim_error:.3e}")
    if native["pu21_psnr"]["avg"] >= 10.0:
        fail(f"PU21-PSNR did not detect chroma difference: {native['pu21_psnr']['avg']}")
    if cvvdp["jod"] <= 0.1:
        fail(f"CVVDP did not detect chroma difference: {cvvdp['jod']}")

    result = {
        "fixture": {
            "width": W, "height": H, "format": "yuv444p10le limited BT.2020 NCL PQ",
            "tiles": TILES, **luma,
        },
        "native": native,
        "cvvdp": cvvdp,
        "tolerance": {
            "PU21_structural_unity": PU_STRUCTURAL_TOLERANCE,
            "post_quantization_luminance_abs_nits": LUMA_ABS_TOLERANCE_NITS,
            "post_quantization_luminance_relative": LUMA_REL_TOLERANCE,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if FAILS:
        return 1
    print("PASS PU21 chroma isolation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
