"""Metric applicability derived from src/main.cpp kMetricCaps and docs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Family = Literal["sdr", "pq", "hlg"]
Scenario = Literal["raw_raw", "raw_comp", "comp_comp"]

CVVDP_DISPLAY = {"sdr": "standard_fhd", "pq": "standard_hdr_pq", "hlg": "standard_hdr_hlg"}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    full_reference: bool
    python_backend: bool
    raw_ok: bool
    groups: tuple[str, ...]  # empty = individual only


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("psnr", True, False, True, ("sdr", "pq", "hlg")),
    MetricSpec("ssim", True, False, True, ("sdr", "pq", "hlg")),
    MetricSpec("msssim", True, False, True, ("sdr", "pq", "hlg")),
    MetricSpec("xpsnr", True, False, True, ("sdr", "pq", "hlg")),
    MetricSpec("vmaf", True, False, True, ("sdr", "pq")),
    MetricSpec("pu21-psnr", True, False, True, ("pq", "hlg")),
    MetricSpec("pu21-ssim", True, False, True, ("pq", "hlg")),
    MetricSpec("pu21-msssim", True, False, True, ("pq", "hlg")),
    MetricSpec("cvvdp", True, True, True, ("sdr", "pq", "hlg")),
    MetricSpec("cvqa-hdr-nr", False, True, False, ("sdr", "pq")),
)

GROUP_METRICS = {
    "sdr": ("psnr", "xpsnr", "ssim", "msssim", "vmaf", "cvvdp", "cvqa-hdr-nr"),
    "pq": ("psnr", "xpsnr", "ssim", "msssim", "vmaf", "pu21-psnr", "pu21-ssim", "pu21-msssim", "cvvdp", "cvqa-hdr-nr"),
    "hlg": ("psnr", "xpsnr", "ssim", "msssim", "pu21-psnr", "pu21-ssim", "pu21-msssim", "cvvdp"),
}


def metric_supported(metric: str, family: Family) -> bool:
    if metric in GROUP_METRICS.get(family, ()):
        return True
    for spec in METRICS:
        if spec.name == metric and family in spec.groups:
            return True
    return False


def scenario_paths(fx, scenario: Scenario) -> tuple[str, str, bool, bool]:
    """Return ref path, test path, ref_is_raw, test_is_raw."""
    if scenario == "raw_raw":
        return str(fx.reference_yuv), str(fx.distorted_yuv), True, True
    if scenario == "raw_comp":
        return str(fx.reference_yuv), str(fx.distorted_lossy), True, False
    if scenario == "comp_comp":
        return str(fx.reference_lossless), str(fx.distorted_lossy), False, False
    raise ValueError(scenario)


def incompatible_reason(metric: str, family: Family, scenario: Scenario) -> str | None:
    if not metric_supported(metric, family):
        return f"metric {metric} not in --metrics {family} group"
    if metric == "cvqa-hdr-nr" and scenario != "raw_comp" and scenario != "comp_comp":
        return "cvqa-hdr-nr requires compressed test"
    if metric == "cvqa-hdr-nr" and scenario == "raw_raw":
        return "cvqa-hdr-nr requires compressed test"
    return None
