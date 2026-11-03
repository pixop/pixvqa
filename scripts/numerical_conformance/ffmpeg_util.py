"""FFmpeg encode/decode helpers for fixture generation."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import FPS, HEIGHT, WIDTH
from .color_science import yuv420p10_frame_bytes


def atomic_tmp_path(out: Path) -> Path:
    return out.with_name(out.name + ".tmp")


def container_format_flags(final_path: Path) -> list[str]:
    suffix = final_path.suffix.lower()
    if suffix == ".mp4":
        return ["-f", "mp4"]
    if suffix == ".mkv":
        return ["-f", "matroska"]
    return []


def run_ffmpeg(ffmpeg: str, args: list[str], *, label: str = "ffmpeg") -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{label}: {proc.stderr[-800:] or proc.stdout[-800:]}")
    return proc


def raw_to_hevc_lossless(
    ffmpeg: str, raw: Path, out: Path, *, family: str,
) -> None:
    meta = _family_x265_params(family, lossless=True)
    tmp = atomic_tmp_path(out)
    try:
        run_ffmpeg(
            ffmpeg,
            [
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p10le",
                "-s",
                f"{WIDTH}x{HEIGHT}",
                "-r",
                FPS.split("/")[0],
                "-i",
                str(raw),
                "-frames:v",
                str(frames_in_raw(raw)),
                "-c:v",
                "libx265",
                "-x265-params",
                meta,
                *container_format_flags(out),
                str(tmp),
            ],
            label=f"lossless-{family}",
        )
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def raw_to_hevc_lossy(ffmpeg: str, raw: Path, out: Path, *, family: str, qp: int = 32) -> None:
    meta = _family_x265_params(family, lossless=False)
    tmp = atomic_tmp_path(out)
    try:
        run_ffmpeg(
            ffmpeg,
            [
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p10le",
                "-s",
                f"{WIDTH}x{HEIGHT}",
                "-r",
                FPS.split("/")[0],
                "-i",
                str(raw),
                "-frames:v",
                str(frames_in_raw(raw)),
                "-c:v",
                "libx265",
                "-qp",
                str(qp),
                "-x265-params",
                meta,
                "-tag:v",
                "hvc1",
                *container_format_flags(out),
                str(tmp),
            ],
            label=f"lossy-{family}",
        )
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def decode_to_yuv(ffmpeg: str, src: Path, dst: Path) -> None:
    tmp = atomic_tmp_path(dst)
    try:
        run_ffmpeg(
            ffmpeg,
            [
                "-y",
                "-i",
                str(src),
                "-pix_fmt",
                "yuv420p10le",
                "-f",
                "rawvideo",
                str(tmp),
            ],
            label=f"decode-{src.name}",
        )
        tmp.replace(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def frames_in_raw(raw: Path) -> int:
    fb = yuv420p10_frame_bytes(WIDTH, HEIGHT)
    size = raw.stat().st_size
    if size % fb != 0:
        raise ValueError(f"{raw}: invalid raw size {size}")
    return size // fb


def ffprobe_stream(ffmpeg: str, path: Path) -> dict[str, str]:
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not Path(ffprobe).is_file() and "ffmpeg" in ffmpeg:
        import shutil

        ffprobe = shutil.which("ffprobe") or ffprobe
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames,pix_fmt,codec_name,color_transfer,color_primaries,color_space",
            "-count_frames",
            "-of",
            "default=nw=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def ffprobe_color(ffmpeg: str, path: Path) -> dict[str, str]:
    return ffprobe_stream(ffmpeg, path)


def _family_x265_params(family: str, *, lossless: bool) -> str:
    base = "bframes=0:keyint=60:min-keyint=60"
    if lossless:
        base = "lossless=1:" + base
    if family == "sdr":
        return base + ":colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"
    if family == "pq":
        return base + ":colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited"
    if family == "hlg":
        return base + ":colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited"
    raise ValueError(family)
