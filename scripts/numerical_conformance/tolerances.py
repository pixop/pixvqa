"""Per-metric numerical tolerance policies (explicit; no silent defaults)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

ToleranceMode = Literal["same_backend_repro", "cross_backend_parity", "reference_parity"]


@dataclass(frozen=True)
class ToleranceSpec:
    abs_tol: float | None
    rel_tol: float | None
    unit: str
    rationale: str
    mode: ToleranceMode


# Same-backend reproducibility (approved baseline regression on one device/build).
SAME_BACKEND: dict[str, ToleranceSpec] = {
    "psnr_avg": ToleranceSpec(
        0.02,
        None,
        "dB",
        "Native PSNR summary avg; FP64 CPU path reference on fixed fixtures.",
        "same_backend_repro",
    ),
    "ssim_avg": ToleranceSpec(
        1e-5,
        None,
        "SSIM index",
        "Plane-weighted SSIM avg; stable on deterministic fixtures.",
        "same_backend_repro",
    ),
    "msssim_avg": ToleranceSpec(
        1e-5,
        None,
        "MS-SSIM index",
        "Multi-scale SSIM avg; same windowing as SSIM family.",
        "same_backend_repro",
    ),
    "xpsnr_avg": ToleranceSpec(
        0.02,
        None,
        "dB",
        "Temporal XPSNR summary avg at fixed fps.",
        "same_backend_repro",
    ),
    "vmaf_mean": ToleranceSpec(
        0.01,
        None,
        "VMAF",
        "libvmaf temporal mean pool on fixed model and inputs.",
        "same_backend_repro",
    ),
    "pu21_psnr_avg": ToleranceSpec(
        0.05,
        None,
        "dB",
        "PU21-PSNR pooled RGB avg (256 peak encoding).",
        "same_backend_repro",
    ),
    "pu21_ssim_avg": ToleranceSpec(
        1e-5,
        None,
        "PU21 SSIM",
        "PU21-encoded luma SSIM avg.",
        "same_backend_repro",
    ),
    "pu21_msssim_avg": ToleranceSpec(
        1e-5,
        None,
        "PU21 MS-SSIM",
        "PU21-encoded luma MS-SSIM avg.",
        "same_backend_repro",
    ),
    "cvvdp_jod": ToleranceSpec(
        1e-4,
        None,
        "JOD",
        "Learned CVVDP JSON/mixed summary JOD on fixed display.",
        "same_backend_repro",
    ),
    "cvqa_quality": ToleranceSpec(
        1e-4,
        None,
        "quality_score",
        "CVQA-HDR-NR mixed summary quality_score scale.",
        "same_backend_repro",
    ),
}

# Cross-backend / cross-path parity (not baseline regression).
CROSS_BACKEND: dict[str, ToleranceSpec] = {
    "psnr_avg": ToleranceSpec(
        0.05,
        None,
        "dB",
        "CPU vs GPU native PSNR may differ slightly in FP paths.",
        "cross_backend_parity",
    ),
}

REFERENCE_PARITY: dict[str, ToleranceSpec] = {
    "cvvdp_jod": ToleranceSpec(
        0.02,
        None,
        "JOD",
        "BerlinSeqs independent reference vs PixVQA CVVDP (optional asset).",
        "reference_parity",
    ),
}


def catalog_export() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in SAME_BACKEND.items():
        out[name] = {
            "abs_tol": spec.abs_tol,
            "rel_tol": spec.rel_tol,
            "unit": spec.unit,
            "mode": spec.mode,
            "rationale": spec.rationale,
        }
    return out


def get_spec(name: str, *, mode: ToleranceMode = "same_backend_repro") -> ToleranceSpec:
    if mode == "cross_backend_parity" and name in CROSS_BACKEND:
        return CROSS_BACKEND[name]
    if mode == "reference_parity" and name in REFERENCE_PARITY:
        return REFERENCE_PARITY[name]
    if name not in SAME_BACKEND:
        raise KeyError(f"no tolerance policy for metric field {name!r}")
    return SAME_BACKEND[name]


def within_tolerance(
    name: str,
    observed: float,
    expected: float,
    *,
    mode: ToleranceMode = "same_backend_repro",
) -> tuple[bool, dict[str, Any]]:
    if not math.isfinite(observed) or not math.isfinite(expected):
        return False, {
            "field": name,
            "observed": observed,
            "expected": expected,
            "reason": "non-finite",
        }
    spec = get_spec(name, mode=mode)
    abs_err = abs(observed - expected)
    detail: dict[str, Any] = {
        "field": name,
        "observed": observed,
        "expected": expected,
        "abs_error": abs_err,
        "abs_tol": spec.abs_tol,
        "rel_tol": spec.rel_tol,
        "unit": spec.unit,
        "mode": spec.mode,
    }
    if spec.abs_tol is not None and abs_err <= spec.abs_tol:
        return True, detail
    if spec.rel_tol is not None and expected != 0.0:
        rel_err = abs_err / abs(expected)
        detail["rel_error"] = rel_err
        if rel_err <= spec.rel_tol:
            return True, detail
    return False, detail


def close(name: str, a: float, b: float, *, mode: ToleranceMode = "same_backend_repro") -> bool:
    ok, _ = within_tolerance(name, a, b, mode=mode)
    return ok
