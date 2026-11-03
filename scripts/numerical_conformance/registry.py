"""Metric applicability derived from src/main.cpp kMetricCaps and docs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .metric_caps import GROUP_METRICS, includes_cvqa

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

def metric_supported(metric: str, family: Family) -> bool:
    if metric in GROUP_METRICS.get(family, ()):
        return True
    for spec in METRICS:
        if spec.name == metric and family in spec.groups:
            return True
    return False


def needs_nr_temporal(metric: str, scenario: Scenario) -> bool:
    if not includes_cvqa(metric):
        return False
    if scenario == "raw_raw":
        return False
    return True


def resolve_media_paths(
    fx,
    scenario: Scenario,
    metric: str,
) -> tuple[str, str, bool, bool]:
    """Return ref path, test path, ref_is_raw, test_is_raw."""
    nr = needs_nr_temporal(metric, scenario)
    if scenario == "raw_raw":
        return str(fx.reference_yuv), str(fx.distorted_yuv), True, True
    if scenario == "raw_comp":
        if nr:
            return str(fx.reference_yuv_60), str(fx.distorted_lossy_60), True, False
        return str(fx.reference_yuv), str(fx.distorted_lossy), True, False
    if scenario == "comp_comp":
        if nr:
            return str(fx.reference_lossless_60), str(fx.distorted_lossy_60), False, False
        return str(fx.reference_lossless), str(fx.distorted_lossy), False, False
    raise ValueError(scenario)


def scenario_paths(fx, scenario: Scenario) -> tuple[str, str, bool, bool]:
    return resolve_media_paths(fx, scenario, "psnr")


def incompatible_reason(metric: str, family: Family, scenario: Scenario) -> str | None:
    if not metric_supported(metric, family):
        return f"metric {metric} not in --metrics {family} group"
    if metric == "cvqa-hdr-nr" and scenario != "raw_comp" and scenario != "comp_comp":
        return "cvqa-hdr-nr requires compressed test"
    if metric == "cvqa-hdr-nr" and scenario == "raw_raw":
        return "cvqa-hdr-nr requires compressed test"
    return None


def cvvdp_skip_reason(metric: str, family: Family, scenario: Scenario) -> str | None:
    """PixVQA rejects SDR raw-side CVVDP (pq/hlg interpretation only for raw YUV)."""
    from .metric_caps import expand_metrics

    names = expand_metrics(metric)
    if "cvvdp" not in names:
        return None
    if family != "sdr":
        return None
    if scenario in ("raw_raw", "raw_comp"):
        return (
            "cvvdp raw YUV supports only pq/hlg interpretation; "
            "SDR raw reference/test is unsupported by pixvqa"
        )
    return None
