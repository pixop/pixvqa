"""Build the public-capability inventory from the source tree.

Every constraint carries a needle that must still occur in the cited file.
The suite fails if a needle disappears, so the inventory cannot drift silently.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def _text(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8", errors="replace")


def _flags(source: str) -> list:
    found = re.findall(r'a == "(--?[A-Za-z0-9][A-Za-z0-9-]*)"', source)
    hdr = re.findall(r"(--(?:ref-|dist-)?hdr-[a-z-]+)", source)
    ordered = []
    for name in found + hdr:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


CONSTRAINTS = [
    {
        "id": "metrics-required",
        "file": "src/main.cpp",
        "needle": "pixvqa: --metrics is required",
        "rule": "A run without one of the eight metric names exits 2 and prints no score.",
    },
    {
        "id": "metric-names",
        "file": "src/main.cpp",
        "needle": "pu21-psnr, pu21-ssim, pu21-msssim",
        "rule": "Public metrics are psnr, xpsnr, ssim, msssim, vmaf, pu21-psnr, pu21-ssim, pu21-msssim.",
    },
    {
        "id": "duplicate-metric-is-additive",
        "file": "src/main.cpp",
        "needle": "opt.enable_psnr = true",
        "rule": "Repeating a metric name, or passing --metrics twice, enables each named metric. It does not replace the earlier list.",
    },
    {
        "id": "vmaf-model",
        "file": "src/main.cpp",
        "needle": "vmaf requires --model",
        "rule": "vmaf requires a readable model. --model and --vmaf-model are aliases and must agree. Either one without vmaf is an error.",
    },
    {
        "id": "vmaf-hdr",
        "file": "src/main.cpp",
        "needle": "vmaf rejects HDR transfer",
        "rule": "An SDR VMAF model is refused when the effective transfer is PQ or HLG. An explicit SDR transfer scores stored samples. An HDR model uses a separate explicit-colorimetry check.",
    },
    {
        "id": "xpsnr-fps",
        "file": "src/main.cpp",
        "needle": "xpsnr requires --fps",
        "rule": "xpsnr requires --fps as a positive integer or num/den. A decimal is truncated toward zero.",
    },
    {
        "id": "qcs",
        "file": "src/main.cpp",
        "needle": "--qcs-range requires --qcs",
        "rule": "--qcs applies to every selected metric. A range requires A < B. vmaf defaults to [70, 100] for every model.",
    },
    {
        "id": "threads",
        "file": "src/main.cpp",
        "needle": "--threads expects auto or a positive integer",
        "rule": "--threads auto or N, with N from 1 to 1000000. auto does not change the software decoder pool.",
    },
    {
        "id": "decoder-pool",
        "file": "src/main.cpp",
        "needle": "kSoftwareDecoderThreads = 8",
        "rule": "Software decoding uses 8 frame threads per compressed input. NVDEC uses 1. --threads applies to metric workers.",
    },
    {
        "id": "frames-positive",
        "file": "src/main.cpp",
        "needle": "--frames expects a positive integer",
        "rule": "--frames and -n require a positive integer. A non-numeric token is an error, not a crash. Omitted, every frame is read.",
    },
    {
        "id": "formats",
        "file": "src/yuv_io.cpp",
        "needle": "return bit_depth == 8 || bit_depth == 10",
        "rule": "Declared layouts are 420, 422, or 444 at 8 or 10 bits. 4:2:0 requires even width and height. nv12, rgb24, and 12-bit are rejected.",
    },
    {
        "id": "geometry-match",
        "file": "src/main.cpp",
        "needle": "reference and test -s/-f disagree",
        "rule": "The two sides must declare the same width, height, bit depth, and chroma.",
    },
    {
        "id": "input-kinds",
        "file": "src/main.cpp",
        "needle": "--input compressed will not read it as raw YUV",
        "rule": "--input auto probes. raw does not probe. compressed errors when FFmpeg finds no video stream.",
    },
    {
        "id": "no-allowlist",
        "file": "src/main.cpp",
        "needle": "There is no codec",
        "rule": "Software decoding is selected by codec id. There is no codec or container allowlist.",
    },
    {
        "id": "decode-policy",
        "file": "src/main.cpp",
        "needle": "AV1 software decoding requires libdav1d or libaom-av1",
        "rule": "CPU AV1 uses libdav1d, then libaom-av1. The native av1 decoder is not used. GPU AV1 uses av1_cuvid. Explicit GPU is not replaced by software.",
    },
    {
        "id": "vvc-gpu",
        "file": "src/decode.cpp",
        "needle": "GPU decoding does not support VVC",
        "rule": "VVC software uses the decoder named vvc. --decode gpu rejects VVC. --decode auto stays on software.",
    },
    {
        "id": "hevc-gpu",
        "file": "src/decode.cpp",
        "needle": "GPU decoding supports HEVC at 420p:10 only",
        "rule": "HEVC NVDEC is 10-bit 4:2:0 on the existing P010 path. Other GPU codecs are rejected.",
    },
    {
        "id": "gpu-metrics",
        "file": "src/main.cpp",
        "needle": "no CPU substitute",
        "rule": "--metric-device gpu does not fall back to the CPU for PSNR, SSIM, MS-SSIM, or XPSNR. Those four accept 8-bit and 10-bit. VMAF has no GPU implementation and uses CPU libvmaf, with a warning, including when it is the only metric. PU21 has no GPU implementation.",
    },
    {
        "id": "auto-metric-profile",
        "file": "src/main.cpp",
        "needle": "opt.format.width >= 3840",
        "rule": "--metric-device auto uses the CPU except one profile: 10-bit, at least 3840x2160, GPU decoding, no VMAF, CUDA build.",
    },
    {
        "id": "hdr-order",
        "file": "src/main.cpp",
        "needle": "Order for every field: per-input option, shared option, stream metadata",
        "rule": "HDR precedence is per-input override, shared override, stream metadata, then the documented assumption.",
    },
    {
        "id": "hdr-same-scope",
        "file": "src/main.cpp",
        "needle": "--tf and --hdr-transfer both set",
        "rule": "Two controls of the same scope are an error: --tf with --hdr-transfer, and the ref and dist equivalents, and --hlg-peak with --hdr-peak-nits.",
    },
    {
        "id": "code-domain",
        "file": "src/main.cpp",
        "needle": "compare stored sample codes",
        "rule": "PSNR, SSIM, MS-SSIM, and XPSNR compare stored sample codes. Changing the HDR transfer does not change that arithmetic.",
    },
    {
        "id": "pairing",
        "file": "src/main.cpp",
        "needle": "ref/test frame count mismatch",
        "rule": "Frames are paired in presentation order. A count mismatch is an error. Timestamps do not drop, repeat, or resynchronize.",
    },
    {
        "id": "empty-input",
        "file": "src/main.cpp",
        "needle": "no frames read",
        "rule": "An empty input exits without a metric summary.",
    },
    {
        "id": "short-yuv",
        "file": "src/yuv_io.cpp",
        "needle": "short YUV frame read",
        "rule": "A partial raw frame is an error.",
    },
    {
        "id": "removed-flags",
        "file": "src/main.cpp",
        "needle": "was removed; use --metrics",
        "rule": "The old single-metric flags and the old device flags are rejected.",
    },
]


CODECS = [
    {"codec": "h264", "containers": ["mp4"], "layouts": ["420p:8"], "cpu": "software h264", "gpu": "unsupported"},
    {"codec": "hevc", "containers": ["mpegts", "mov"], "layouts": ["420p:10"], "cpu": "software hevc", "gpu": "nvdec hevc, 420p:10 only"},
    {"codec": "av1", "containers": ["mp4", "mkv"], "layouts": ["420p:8", "420p:10"], "cpu": "libdav1d", "gpu": "av1_cuvid"},
    {"codec": "vvc", "containers": ["mp4"], "layouts": ["420p:10"], "cpu": "native vvc", "gpu": "unsupported on this NVDEC path"},
    {"codec": "vp9", "containers": ["webm"], "layouts": ["420p:8"], "cpu": "software vp9", "gpu": "unsupported"},
    {"codec": "ffv1", "containers": ["mkv"], "layouts": ["422p:10", "444p:8"], "cpu": "software ffv1", "gpu": "unsupported"},
    {"codec": "mpeg2video", "containers": ["mpegts"], "layouts": ["420p:8"], "cpu": "software mpeg2video", "gpu": "unsupported"},
    {"codec": "mjpeg", "containers": ["avi"], "layouts": ["420p:8"], "cpu": "software mjpeg", "gpu": "unsupported"},
    {"codec": "prores", "containers": ["mov"], "layouts": ["422p:10"], "cpu": "software prores", "gpu": "unsupported"},
]


def build_inventory(root: Path) -> dict:
    main = _text(root, "src/main.cpp")
    missing = []
    for item in CONSTRAINTS:
        if item["needle"] not in _text(root, item["file"]):
            missing.append(item["id"])
    files = ["src/main.cpp", "src/decode.cpp", "src/yuv_io.cpp", "include/pixvqa/types.hpp"]
    return {
        "source": "src/main.cpp, src/decode.cpp, src/yuv_io.cpp, include/pixvqa/types.hpp",
        "source_sha256_16": {rel: _sha(root / rel) for rel in files},
        "options": _flags(main),
        "metrics": ["psnr", "xpsnr", "ssim", "msssim", "vmaf", "pu21-psnr", "pu21-ssim", "pu21-msssim"],
        "defaults": {
            "--decode": "auto",
            "--input": "auto",
            "--metric-device": "auto",
            "--threads": "auto",
            "-f bits": "10",
            "-v": "1",
            "--frames": "all",
            "qcs psnr and xpsnr": "[30, 50]",
            "qcs ssim and msssim": "[0.8, 1.0]",
            "qcs vmaf": "[70, 100]",
            "qcs pu21-psnr": "[30, 50]",
            "qcs pu21-ssim and pu21-msssim": "[0.8, 1.0]",
            "software decoder threads": 8,
            "nvdec decoder threads": 1,
        },
        "decoder_backends": {
            "cpu": "FFmpeg software decoder for the stream codec id. AV1 prefers libdav1d.",
            "gpu": "NVDEC. HEVC at 10-bit 4:2:0, AV1 at 8-bit or 10-bit 4:2:0.",
            "auto": "Software when that decoder is linked. AV1 NVDEC only when software AV1 is absent. VVC stays on software.",
        },
        "metric_backends": {
            "cpu": "Every selected metric.",
            "gpu": "PSNR, SSIM, MS-SSIM, and XPSNR when the layout is supported. VMAF stays on CPU libvmaf and warns, including when it is the only GPU request. PU21 is rejected when it is the only GPU request.",
            "auto": "CPU, except the single 4K HEVC GPU-decode profile.",
        },
        "codecs": CODECS,
        "pixel_formats": ["420p:8", "420p:10", "422p:8", "422p:10", "444p:8", "444p:10"],
        "hdr": {
            "transfers": ["sdr", "pq", "hlg", "linear"],
            "legacy_transfers": {"st2084": "pq", "stdb67": "hlg"},
            "primaries": ["bt2020", "bt709"],
            "matrix": ["bt2020nc", "bt709"],
            "range": ["limited", "full"],
            "chroma_loc": ["left", "center", "topleft"],
            "chroma_up": ["bilinear", "nearest"],
            "peak_nits": "(0, 10000]",
            "black_nits": "[0, peak)",
            "hlg_gamma": "(0, 10]",
            "precedence": ["per-input override", "shared override", "stream metadata", "documented assumption"],
        },
        "unsupported": [
            "nv12, rgb24, and 12-bit as declared or software-decoded layouts",
            "GPU decoding of VVC, H.264, VP9, FFV1, MPEG-2, MJPEG, and ProRes",
            "HEVC NVDEC other than 10-bit 4:2:0",
            "native av1 decoder",
            "VMAF on an effective PQ or HLG transfer",
            "pu21-xpsnr and pu21-vmaf",
            "nonzero --seek",
            "--wsssim, --wspsnr, --rfilter, --tfilter, --geom, --mt, --fif",
            "bt601 as a PQ or HLG color system",
            "HEVC in AVI with the FFmpeg build used for the P5 oracle",
        ],
        "constraints": CONSTRAINTS,
        "missing_needles": missing,
    }


def check_inventory(root: Path) -> tuple:
    inv = build_inventory(root)
    if inv["missing_needles"]:
        return False, "inventory needles missing: " + ", ".join(inv["missing_needles"])
    required = [
        "--metrics", "--decode", "--input", "--metric-device", "--threads", "--model",
        "--vmaf-model", "--qcs", "--qcs-range", "--fps", "--frames", "--hdr-transfer",
        "--help", "--version", "--help-hdr",
    ]
    absent = [name for name in required if name not in inv["options"]]
    if absent:
        return False, "options not found in src/main.cpp: " + ", ".join(absent)
    return True, "inventory needles and public options match the source"
