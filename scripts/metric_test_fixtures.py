#!/usr/bin/env python3
"""Shared fixtures for metric CLI conformance tests."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def write_yuv(path: Path, *, frames: int = 2, size: int = 64) -> None:
    path.write_bytes(b"\x00\x00" * (size * size * 3 // 2 * frames))


def run_ffmpeg(ffmpeg: str, args: list[str]) -> None:
    proc = subprocess.run([ffmpeg, "-v", "error", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-400:] or "ffmpeg failed")


def make_pq_mp4(ffmpeg: str, path: Path, *, size: int = 64, frames: int = 2) -> None:
    run_ffmpeg(
        ffmpeg,
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}x{size}:rate=24",
            "-frames:v",
            str(frames),
            "-vf",
            "setparams=color_trc=smpte2084:color_primaries=bt2020:colorspace=bt2020nc:range=tv",
            "-c:v",
            "libx264",
            "-qp",
            "28",
            "-pix_fmt",
            "yuv420p10le",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
            "-color_range",
            "tv",
            str(path),
        ],
    )


def make_hlg_mp4(ffmpeg: str, path: Path, *, size: int = 64, frames: int = 2) -> None:
    run_ffmpeg(
        ffmpeg,
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}x{size}:rate=24",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-qp",
            "28",
            "-pix_fmt",
            "yuv420p10le",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "arib-std-b67",
            "-colorspace",
            "bt2020nc",
            "-color_range",
            "tv",
            str(path),
        ],
    )


def make_sdr_mp4(ffmpeg: str, path: Path, *, size: int = 64, frames: int = 2) -> None:
    run_ffmpeg(
        ffmpeg,
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}x{size}:rate=24",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-qp",
            "28",
            "-pix_fmt",
            "yuv420p10le",
            str(path),
        ],
    )


def make_no_color_metadata_mp4(ffmpeg: str, path: Path, *, size: int = 64, frames: int = 2) -> None:
    """H.264 without VUI color metadata (strip tags after encode)."""
    tmp = path.with_suffix(".tmp.mp4")
    run_ffmpeg(
        ffmpeg,
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}x{size}:rate=24",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-qp",
            "28",
            "-pix_fmt",
            "yuv420p10le",
            "-bsf:v",
            "filter_units=remove_types=6",
            str(tmp),
        ],
    )
    tmp.replace(path)


def write_corrupt_mp4(path: Path) -> None:
    path.write_bytes(b"\x00" * 64 + b"not-an-mp4")


def cvvdp_mock_main_py() -> str:
    return '''import json
import sys

args = sys.argv[1:]

def value(flag):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None

if "--check" in args:
    print("pixvqa-check ok")
    raise SystemExit(0)

interpretation = value("--interpretation") or "pq"
display = value("--display") or "standard_hdr_pq"
device = value("--device") or "auto"
jod = 1.0
transient = "--transient-output" in args
out_json = value("--output-json")
output_json = None if transient else out_json

if value("--protocol") == "lines":
    lines = [
        "pixvqa-protocol 1",
        "pixvqa-value metric cvvdp",
        f"pixvqa-value jod {jod!r}",
        "pixvqa-value frames_evaluated 2",
        f"pixvqa-value display {display}",
        f"pixvqa-value interpretation {interpretation}",
        f"pixvqa-value device {device}",
        f"pixvqa-value output_json {output_json if output_json else 'none'}",
        "pixvqa-end",
    ]
    print("\\n".join(lines))
    raise SystemExit(0)

payload = {
    "metric": "cvvdp",
    "jod": jod,
    "display": display,
    "interpretation": interpretation,
    "device": device,
    "output_json": output_json,
}
print(json.dumps(payload, separators=(",", ":")))
'''


def install_cvvdp_mock(tmp_path: Path) -> Path:
    """Install or refresh the pixvqa_cvvdp mock under tmp_path/mock (idempotent)."""
    mock_root = tmp_path / "mock"
    mock_root.mkdir(parents=True, exist_ok=True)
    pkg = mock_root / "pixvqa_cvvdp"
    pkg.mkdir(exist_ok=True)
    main_py = cvvdp_mock_main_py()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__main__.py").write_text(main_py, encoding="utf-8")
    return mock_root


def cvvdp_mock_env(tmp_path: Path) -> dict[str, str]:
    mock_root = install_cvvdp_mock(tmp_path)
    env = os.environ.copy()
    env["PIXVQA_LEARNED_CONFIG"] = "none"
    env["PYTHONPATH"] = str(mock_root)
    return env


def verify_cvvdp_mock_repeatable(*, iterations: int = 5) -> None:
    """Regression: repeated mock install in one directory must not raise."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pixvqa_mock_repeat_") as tmp:
        root = Path(tmp)
        for _ in range(iterations):
            install_cvvdp_mock(root)


def base_env(tmp_path: Path, *, mock_cvvdp: bool) -> dict[str, str]:
    if mock_cvvdp:
        return cvvdp_mock_env(tmp_path)
    env = os.environ.copy()
    env["PIXVQA_LEARNED_CONFIG"] = "none"
    return env


class FixtureContext:
    def __init__(self, tmp: Path, ffmpeg: str | None) -> None:
        self.tmp = tmp
        self.raw = tmp / "ref.yuv"
        self.raw_b = tmp / "test.yuv"
        self.pq = tmp / "pq.mp4"
        self.hlg = tmp / "hlg.mp4"
        self.sdr = tmp / "sdr.mp4"
        self.nometa = tmp / "nometa.mp4"
        self.corrupt = tmp / "corrupt.mp4"
        self.missing = tmp / "missing.yuv"
        self.python = sys.executable
        write_yuv(self.raw)
        write_yuv(self.raw_b)
        write_corrupt_mp4(self.corrupt)
        if ffmpeg:
            for maker, dest in (
                (make_pq_mp4, self.pq),
                (make_hlg_mp4, self.hlg),
                (make_sdr_mp4, self.sdr),
                (make_no_color_metadata_mp4, self.nometa),
            ):
                try:
                    maker(ffmpeg, dest)
                except RuntimeError:
                    # FFmpeg -y overwrites; re-run after failed partial output is safe.
                    if dest.is_file():
                        dest.unlink(missing_ok=True)
                    maker(ffmpeg, dest)

    def subst(self, token: str) -> str:
        key = token.strip("{}")
        if hasattr(self, key):
            val = getattr(self, key)
            return str(val)
        return token

    def expand_args(self, args: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        for a in args:
            if a.startswith("{") and a.endswith("}"):
                out.append(self.subst(a))
            else:
                out.append(a)
        return out
