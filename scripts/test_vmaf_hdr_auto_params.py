#!/usr/bin/env python3
"""Regression checks for bundled HDR VMAF's model-specific color assumptions."""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "swdec" / "pixvqa"))
HDR_MODEL = ROOT / "model" / "hdr_vmaf_v1.1.8" / "hdr_vmaf_v1.1.8_3d0h.json"
SDR_MODEL = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"
W, H, N = 320, 240, 2
HDR = [
    "--hdr-transfer", "pq",
    "--hdr-primaries", "bt2020",
    "--hdr-matrix", "bt2020nc",
    "--hdr-range", "limited",
]


def fail(message: str) -> None:
    raise SystemExit(message)


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(BIN), *args], capture_output=True, text=True, timeout=180)


def write_yuv(path: Path, bias: int) -> None:
    with path.open("wb") as out:
        for frame in range(N):
            for y in range(H):
                for x in range(W):
                    value = 64 + ((bias + frame * 13 + x * 7 + y * 5) % 876)
                    out.write(struct.pack("<H", value))
            out.write(struct.pack("<H", 512) * ((W // 2) * (H // 2)))
            out.write(struct.pack("<H", 512) * ((W // 2) * (H // 2)))


def raw_base(ref: Path, dist: Path, fmt: str = "420p:10") -> list[str]:
    return [
        "-r", str(ref), "-s", f"{W}x{H}", "-f", fmt,
        "-t", str(dist), "-s", f"{W}x{H}", "-f", fmt,
        "-n", str(N), "-v", "0",
    ]


def vmaf_args(model: Path, inputs: list[str], extra: list[str] | None = None) -> list[str]:
    return ["--metrics", "vmaf", "--model", str(model), *(extra or []), *inputs]


def vmaf_mean(text: str) -> str:
    found = re.search(r"\[ VMAF mean:([^ ]+)", text)
    if not found:
        fail(f"missing VMAF summary:\n{text[-800:]}")
    return found.group(1)


def expect_ok(label: str, args: list[str], needles: tuple[str, ...] = ()) -> str:
    done = run(args)
    text = done.stdout + done.stderr
    if done.returncode != 0:
        fail(f"{label}: rc={done.returncode}\n{text[-1200:]}")
    for needle in needles:
        if needle not in text:
            fail(f"{label}: missing {needle!r}\n{text[-1200:]}")
    print(f"OK   {label}")
    return text


def expect_fail(label: str, args: list[str], needle: str) -> None:
    done = run(args)
    text = done.stdout + done.stderr
    if done.returncode == 0 or needle not in text:
        fail(f"{label}: expected {needle!r}, rc={done.returncode}\n{text[-1200:]}")
    print(f"OK   {label}")


def encode(ffmpeg: str, raw: Path, output: Path, *, trc: str, primaries: str,
           matrix: str, color_range: str) -> None:
    # FFmpeg 8 takes transfer and primaries from the frames, so output options
    # alone leave them unset; an output matrix or range that differs from the
    # frames makes the auto-inserted scaler convert the samples. Tag the frames
    # with setparams and request the same values from the encoder.
    tags = [(key, flag, value) for key, flag, value in (
        ("color_trc", "-color_trc", trc),
        ("color_primaries", "-color_primaries", primaries),
        ("colorspace", "-colorspace", matrix),
        ("range", "-color_range", color_range),
    ) if value]
    params = [f"{key}={value}" for key, _, value in tags]
    color = ["-vf", "setparams=" + ":".join(params)] if params else []
    for _, flag, value in tags:
        color += [flag, value]
    done = subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}", "-r", "24",
            "-i", str(raw), "-frames:v", str(N), "-c:v", "ffv1", "-pix_fmt", "yuv420p10le",
            *color, str(output),
        ],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        fail(f"ffmpeg encode failed:\n{done.stderr}")
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
        capture_output=True,
    )
    if decoded.returncode != 0 or decoded.stdout != raw.read_bytes()[: N * W * H * 3]:
        fail(f"fixture {output.name} does not decode to the raw source samples")
    ffprobe = shutil.which("ffprobe", path=str(Path(ffmpeg).parent)) or shutil.which("ffprobe")
    if not ffprobe or not params:
        return
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=color_transfer,color_primaries,color_space,color_range", "-of", "json", str(output)],
        capture_output=True, text=True,
    )
    stream = (json.loads(probe.stdout or "{}").get("streams") or [{}])[0]
    for key, value in (("color_transfer", trc), ("color_primaries", primaries),
                       ("color_space", matrix), ("color_range", color_range)):
        if value and stream.get(key) != value:
            fail(f"fixture {output.name} lost {key}={value}: ffprobe reports {stream.get(key)!r}")


def compressed_checks(ref: Path, dist: Path, directory: Path) -> None:
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        print("SKIP compressed metadata checks: ffmpeg is unavailable")
        return

    def inputs(tag: str, *, trc: str, primaries: str, matrix: str, color_range: str) -> list[str]:
        reference = directory / f"{tag}_ref.mkv"
        distorted = directory / f"{tag}_dist.mkv"
        encode(ffmpeg, ref, reference, trc=trc, primaries=primaries, matrix=matrix,
               color_range=color_range)
        encode(ffmpeg, dist, distorted, trc=trc, primaries=primaries, matrix=matrix,
               color_range=color_range)
        return ["-r", str(reference), "-t", str(distorted), "-n", str(N), "-v", "0"]

    tagged = expect_ok(
        "compressed tagged PQ",
        vmaf_args(HDR_MODEL, inputs("pq", trc="smpte2084", primaries="bt2020",
                                    matrix="bt2020nc", color_range="tv")),
    )
    if "warning: HDR VMAF" in tagged:
        fail(f"compressed tagged PQ unexpectedly warned:\n{tagged[-1200:]}")

    expect_ok(
        "compressed missing metadata",
        vmaf_args(HDR_MODEL, inputs("missing", trc="", primaries="", matrix="", color_range="")),
    )
    expect_ok(
        "compressed HLG metadata warning",
        vmaf_args(HDR_MODEL, inputs("hlg", trc="arib-std-b67", primaries="bt2020",
                                    matrix="bt2020nc", color_range="tv")),
        ("Reference input metadata transfer is hlg", "Distorted input metadata transfer is hlg"),
    )
    expect_ok(
        "compressed color metadata warnings",
        vmaf_args(HDR_MODEL, inputs("color", trc="smpte2084", primaries="bt709",
                                    matrix="bt709", color_range="pc")),
        ("metadata primaries is bt709", "metadata matrix is bt709", "metadata range is full"),
    )


def main() -> int:
    if not BIN.is_file() or not HDR_MODEL.is_file() or not SDR_MODEL.is_file():
        print("missing PixVQA binary or bundled model", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="pixvqa_hdr_vmaf_auto_") as temp:
        root = Path(temp)
        ref, dist = root / "ref.yuv", root / "dist.yuv"
        write_yuv(ref, 0)
        write_yuv(dist, 37)
        base = raw_base(ref, dist)

        automatic = expect_ok("raw bundled HDR model without HDR flags", vmaf_args(HDR_MODEL, base))
        explicit = expect_ok("raw bundled HDR model with explicit HDR flags",
                             vmaf_args(HDR_MODEL, base, HDR))
        if vmaf_mean(automatic) != vmaf_mean(explicit):
            fail("valid PQ score changed between automatic and explicit HDR parameters")
        print("OK   automatic and explicit HDR VMAF scores match")

        expect_ok(
            "explicit conflicting HDR options warn",
            vmaf_args(
                HDR_MODEL, base,
                ["--hdr-transfer", "hlg", "--hdr-primaries", "bt709",
                 "--hdr-matrix", "bt709", "--hdr-range", "full"],
            ),
            ("explicit transfer is hlg", "explicit primaries is bt709",
             "explicit matrix is bt709", "explicit range is full"),
        )
        combined = expect_ok(
            "HDR VMAF plus PU21 keeps HLG",
            [
                "--metrics", "vmaf,pu21-psnr", "--model", str(HDR_MODEL),
                "--hdr-transfer", "hlg", *base,
            ],
            ("[ PU21-PSNR:", "display reference: hlg",
             "color_ref=limited:bt2020:pq:bt2020nc", "explicit transfer is hlg"),
        )
        if "display reference: pq" in combined:
            fail("bundled HDR VMAF labels leaked into PU21 HDR resolution")

        custom = root / "custom_hdr_model.json"
        custom.write_bytes(HDR_MODEL.read_bytes() + b"\n")
        expect_fail("custom HDR model remains explicit", vmaf_args(custom, base), "explicit PQ transfer")
        expect_ok("custom HDR model accepts explicit tuple", vmaf_args(custom, base, HDR))

        expect_fail(
            "bundled HDR VMAF rejects 8-bit",
            vmaf_args(HDR_MODEL, raw_base(ref, dist, "420p:8")),
            "HDR VMAF supports limited-range YUV420p10 only",
        )
        expect_fail(
            "bundled HDR VMAF rejects 4:2:2",
            vmaf_args(HDR_MODEL, raw_base(ref, dist, "422p:10")),
            "HDR VMAF supports limited-range YUV420p10 only",
        )
        expect_fail(
            "SDR VMAF still rejects PQ",
            vmaf_args(SDR_MODEL, base, ["--hdr-transfer", "pq"]),
            "rejects explicit HDR",
        )
        compressed_checks(ref, dist, root)
    print("All bundled HDR VMAF automatic-parameter checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
