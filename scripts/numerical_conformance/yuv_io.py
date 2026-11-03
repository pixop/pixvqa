"""Planar YUV420 10-bit LE read/write (streaming, bounded memory)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .color_science import encode_family_rgb, linear_rgb_to_ycbcr10_limited, yuv420p10_frame_bytes
from .config import HEIGHT, WIDTH
from .scene import scene_linear_rgb


def encode_frame_yuv420p10_planes(
    frame_index: int,
    frame_count: int,
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = scene_linear_rgb(frame_index, frame_count=frame_count)
    nlrgb = encode_family_rgb(linear, family)
    y, cb, cr = linear_rgb_to_ycbcr10_limited(nlrgb)
    cb420 = cb[0::2, 0::2]
    cr420 = cr[0::2, 0::2]
    return y, cb420, cr420


def write_planes_to_handle(
    handle,
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
) -> None:
    handle.write(np.ascontiguousarray(y, dtype="<u2").tobytes())
    handle.write(np.ascontiguousarray(cb, dtype="<u2").tobytes())
    handle.write(np.ascontiguousarray(cr, dtype="<u2").tobytes())


def stream_reference_yuv(
    path: Path,
    *,
    family: str,
    frame_count: int,
) -> None:
    """Write reference YUV one frame at a time (no full-sequence RGB retention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            for i in range(frame_count):
                y, cb, cr = encode_frame_yuv420p10_planes(i, frame_count, family)
                write_planes_to_handle(handle, y, cb, cr)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def rgb_frames_to_yuv(
    linear_frames: list[np.ndarray], family: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Legacy list path (tests / small frame counts only)."""
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for linear in linear_frames:
        nlrgb = encode_family_rgb(linear, family)
        y, cb, cr = linear_rgb_to_ycbcr10_limited(nlrgb)
        cb420 = cb[0::2, 0::2]
        cr420 = cr[0::2, 0::2]
        out.append((y, cb420, cr420))
    return out


def write_yuv420p10(path: Path, frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as f:
            for y, cb, cr in frames:
                write_planes_to_handle(f, y, cb, cr)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_yuv420p10(path: Path, width: int, height: int, frames: int) -> bytes:
    need = yuv420p10_frame_bytes(width, height) * frames
    data = path.read_bytes()
    if len(data) != need:
        raise ValueError(f"{path}: expected {need} bytes, got {len(data)}")
    return data


def raw_yuv_frame_count(path: Path, width: int = WIDTH, height: int = HEIGHT) -> int:
    fb = yuv420p10_frame_bytes(width, height)
    size = path.stat().st_size
    if size % fb != 0:
        raise ValueError(f"{path}: size {size} not multiple of frame size {fb}")
    return size // fb


def files_identical(a: Path, b: Path, *, chunk: int = 1 << 20) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            ba, bb = fa.read(chunk), fb.read(chunk)
            if ba != bb:
                return False
            if not ba:
                return True
