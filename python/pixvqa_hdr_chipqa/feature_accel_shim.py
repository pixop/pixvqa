"""Parity-preserving ChipQA/hdr_utils accelerations injected before upstream hdr_chipqa.py.

All changes are adapter-local monkey-patches; upstream sources and model weights are untouched.
Verified: bit-identical feature vectors vs unpatched path on the 60f SDR benchmark clip.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gamma

_installed = False
_YUV_BULK: dict[tuple[str, int, int, int], bytes] = {}

# Same lattice as upstream estimateggdparam (list comprehension, not np.arange).
_EGGD_GAM = np.asarray([x / 1000.0 for x in range(200, 10000, 1)], dtype=np.float64)
_EGGD_R_GAM = (gamma(1.0 / _EGGD_GAM) * gamma(3.0 / _EGGD_GAM)) / ((gamma(2.0 / _EGGD_GAM)) ** 2)


def _fast_paired_product(new_im: np.ndarray) -> tuple[np.ndarray, ...]:
    # np.roll already returns a new array; upstream .copy() is redundant.
    shift1 = np.roll(new_im, 1, axis=1)
    shift2 = np.roll(new_im, 1, axis=0)
    shift3 = np.roll(np.roll(new_im, 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im, 1, axis=0), -1, axis=1)
    return (
        shift1 * new_im,
        shift2 * new_im,
        shift3 * new_im,
        shift4 * new_im,
    )


def _fast_estimateggdparam(vec: np.ndarray) -> tuple[float, float]:
    flat = np.ravel(vec)
    sigma_sq = np.mean(flat**2)
    sigma = float(np.sqrt(sigma_sq))
    e = np.mean(np.abs(flat))
    rho = sigma_sq / (e**2 + 1e-6)
    array_position = int((np.abs(rho - _EGGD_R_GAM)).argmin())
    return float(_EGGD_GAM[array_position]), sigma


def _fast_extract_subband_feats(mscncoefs: np.ndarray, *, aggd_features: Any) -> np.ndarray:
    alpha_m, sigma = _fast_estimateggdparam(mscncoefs)
    pps1, pps2, pps3, pps4 = _fast_paired_product(mscncoefs)
    alpha1, n1, _bl1, _br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, n2, _bl2, _br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, n3, _bl3, _br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, n4, _bl4, _br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array(
        [
            alpha_m,
            sigma,
            alpha1,
            n1,
            lsq1**2,
            rsq1**2,
            alpha2,
            n2,
            lsq2**2,
            rsq2**2,
            alpha3,
            n3,
            lsq3**2,
            rsq3**2,
            alpha4,
            n4,
            lsq4**2,
            rsq4**2,
        ]
    )


def _yuv_read_bulk(
    filename: str,
    frame_num: int,
    height: int,
    width: int,
    bit_depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single bulk read per file; frame slices match upstream yuv_read/fread layout."""
    key = (filename, height, width, bit_depth)
    if key not in _YUV_BULK:
        with open(filename, "rb") as handle:
            _YUV_BULK[key] = handle.read()
    raw = _YUV_BULK[key]
    if bit_depth == 8:
        frame_bytes = int(height * width * 1.5)
        offset = int(frame_num * frame_bytes)
        y_count = height * width
        uv_count = height * width // 4
        y1 = np.frombuffer(raw, dtype=np.uint8, count=y_count, offset=offset)
        u1 = np.frombuffer(raw, dtype=np.uint8, count=uv_count, offset=offset + y_count)
        v1 = np.frombuffer(raw, dtype=np.uint8, count=uv_count, offset=offset + y_count + uv_count)
    elif bit_depth in (10, 12):
        frame_bytes = height * width * 3
        offset = int(frame_num * frame_bytes)
        y_count = height * width
        uv_count = height * width // 4
        y1 = np.frombuffer(raw, dtype=np.uint16, count=y_count, offset=offset)
        u1 = np.frombuffer(raw, dtype=np.uint16, count=uv_count, offset=offset + y_count * 2)
        v1 = np.frombuffer(raw, dtype=np.uint16, count=uv_count, offset=offset + y_count * 2 + uv_count * 2)
    else:
        raise ValueError(f"unsupported bit_depth {bit_depth}")
    # Mirror upstream reshape after fread (nelements, 1) -> (h, w).
    y = np.reshape(np.array(y1), (height, width))
    u = np.reshape(np.array(u1), (height // 2, width // 2)).repeat(2, axis=0).repeat(2, axis=1)
    v = np.reshape(np.array(v1), (height // 2, width // 2)).repeat(2, axis=0).repeat(2, axis=1)
    return y, u, v


def install() -> None:
    """Apply monkey-patches once per feature-extraction subprocess."""
    global _installed
    if _installed:
        return
    import ChipQA.save_stats as save_stats

    def extract_subband_feats(mscncoefs: np.ndarray) -> np.ndarray:
        return _fast_extract_subband_feats(mscncoefs, aggd_features=save_stats.aggd_features)

    save_stats.paired_product = _fast_paired_product
    save_stats.estimateggdparam = _fast_estimateggdparam
    save_stats._extract_subband_feats = extract_subband_feats
    save_stats.extract_subband_feats = extract_subband_feats

    from utils import hdr_utils

    hdr_utils.yuv_read = _yuv_read_bulk
    _installed = True
