"""Per-metric numerical tolerance policies."""
from __future__ import annotations

TOLERANCE = {
    "psnr_avg": 0.02,
    "ssim_avg": 1e-5,
    "pu21_psnr_avg": 0.05,
    "cvvdp_jod": 1e-4,
}


def close(name: str, a: float, b: float) -> bool:
    tol = TOLERANCE.get(name, 1e-6)
    return abs(a - b) <= tol
