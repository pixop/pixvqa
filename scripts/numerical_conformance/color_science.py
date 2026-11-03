"""Documented SDR/PQ/HLG encoding from scene-linear RGB (see README.md)."""
from __future__ import annotations

import numpy as np

# BT.709 limited-range 10-bit matrix (same Kr,Kg,Kb as BT.2020 for our synthetic RGB path).
_KR, _KG, _KB = 0.2627, 0.6780, 0.0593


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def linear_rgb_to_ycbcr10_limited(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rgb: (H,W,3) linear display-referred 0..1 -> Y,Cb,Cr uint16 10-bit limited."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = _KR * r + _KG * g + _KB * b
    cb = (b - y) / (1.0 - _KB) + 0.5
    cr = (r - y) / (1.0 - _KR) + 0.5
    y_n = _clip01(y)
    cb_n = _clip01(cb)
    cr_n = _clip01(cr)
    y_q = np.round(y_n * 876.0 + 64.0).astype(np.uint16)
    cb_q = np.round(cb_n * 896.0 + 64.0).astype(np.uint16)
    cr_q = np.round(cr_n * 896.0 + 64.0).astype(np.uint16)
    return y_q, cb_q, cr_q


def pq_oetf(linear: np.ndarray) -> np.ndarray:
    """ST 2084 PQ OETF: linear display 0..1 -> non-linear PQ code 0..1."""
    m1, m2 = 2610.0 / 4096.0 / 4.0, 2523.0 / 4096.0 * 128.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 4096.0, 2392.0 / 4096.0
    x = np.clip(linear, 0.0, None)
    xp = np.power(x / 10000.0, m1)
    num = c1 + c2 * xp
    den = 1.0 + c3 * xp
    return np.power(np.clip(num / den, 0.0, None), m2)


def hlg_oetf(linear: np.ndarray) -> np.ndarray:
    """BT.2100 HLG OETF on normalized signal 0..1 (scene linear)."""
    a, b = 0.17883277, 0.28466892
    c = 0.55991073
    x = np.clip(linear, 0.0, None)
    out = np.empty_like(x)
    lo = x <= 1.0 / 12.0
    out[lo] = np.sqrt(3.0 * x[lo])
    hi = ~lo
    out[hi] = a * np.log(12.0 * x[hi] - b) + c
    return np.clip(out, 0.0, 1.0)


def encode_family_rgb(linear_rgb: np.ndarray, family: str) -> np.ndarray:
    """Map scene-linear RGB to non-linear R'G'B' before YUV for each transfer family."""
    if family == "sdr":
        # BT.709 OETF (gamma 2.4) on linear, peak 100 nits mapped to 0..1 reference white.
        return _clip01(np.power(_clip01(linear_rgb), 1.0 / 2.4))
    if family == "pq":
        return pq_oetf(linear_rgb * 10000.0)  # linear_rgb 0..1 -> 0..10000 nits
    if family == "hlg":
        return hlg_oetf(linear_rgb)
    raise ValueError(family)


def yuv420p10_frame_bytes(width: int, height: int) -> int:
    return (width * height + 2 * (width // 2) * (height // 2)) * 2
