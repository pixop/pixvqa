"""Planar YUV420 10-bit LE read/write."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .color_science import linear_rgb_to_ycbcr10_limited, yuv420p10_frame_bytes
from .config import HEIGHT, WIDTH


def rgb_frames_to_yuv(
    linear_frames: list[np.ndarray], family: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from .color_science import encode_family_rgb

    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for linear in linear_frames:
        nlrgb = encode_family_rgb(linear, family)
        y, cb, cr = linear_rgb_to_ycbcr10_limited(nlrgb)
        # Subsample chroma 420.
        cb420 = cb[0::2, 0::2]
        cr420 = cr[0::2, 0::2]
        out.append((y, cb420, cr420))
    return out


def write_yuv420p10(path: Path, frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for y, cb, cr in frames:
            f.write(y.astype("<u2").tobytes())
            f.write(cb.astype("<u2").tobytes())
            f.write(cr.astype("<u2").tobytes())


def read_yuv420p10(path: Path, width: int, height: int, frames: int) -> bytes:
    need = yuv420p10_frame_bytes(width, height) * frames
    data = path.read_bytes()
    if len(data) != need:
        raise ValueError(f"{path}: expected {need} bytes, got {len(data)}")
    return data


def frames_identical(a: bytes, b: bytes) -> bool:
    return a == b
