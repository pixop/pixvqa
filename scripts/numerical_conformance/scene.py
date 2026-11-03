"""Deterministic synthetic UHD scene (5 frames)."""
from __future__ import annotations

import numpy as np

from .config import FRAMES, HEIGHT, WIDTH


def scene_linear_rgb(frame_index: int) -> np.ndarray:
    """Return (H,W,3) scene-linear RGB in 0..1 with motion and test patterns."""
    h, w = HEIGHT, WIDTH
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    t = frame_index / max(FRAMES - 1, 1)
    # Horizontal luma gradient + vertical colour shift.
    base = 0.12 + 0.55 * xs + 0.08 * np.sin(2 * np.pi * (ys + t))
    r = np.clip(base + 0.25 * np.sin(2 * np.pi * (xs * 8 + t)), 0, 1)
    g = np.clip(base + 0.15 * np.cos(2 * np.pi * (ys * 6 - t)), 0, 1)
    b = np.clip(base + 0.2 * np.sin(2 * np.pi * ((xs + ys) * 4)), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    # Dark corner (shadow detail).
    mask_dark = (xs < 0.15) & (ys < 0.15)
    rgb[mask_dark] *= 0.02
    # Bright highlight patch.
    cx, cy = 0.72 + 0.05 * t, 0.35
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    rgb[dist < 0.06] = np.clip(rgb[dist < 0.06] + 0.85, 0, 1)
    # Fine vertical stripes (texture).
    stripe = 0.04 * (np.sin(xs * w * 0.35) > 0.95)
    rgb[..., 1:2] += stripe[..., None]
    # Moving edge block.
    bx0 = int((0.25 + 0.4 * t) * w)
    bx1 = bx0 + w // 16
    rgb[:, bx0:bx1, :] = np.array([0.95, 0.05, 0.05])
    return np.clip(rgb, 0.0, 1.0)


def all_scene_frames() -> list[np.ndarray]:
    return [scene_linear_rgb(i) for i in range(FRAMES)]
