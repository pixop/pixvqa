"""Mirror of src/main.cpp kMetricCaps for harness command generation."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Cap:
    name: str
    full_reference: bool
    honors_frames: bool


CAPS: dict[str, Cap] = {
    "psnr": Cap("psnr", True, True),
    "xpsnr": Cap("xpsnr", True, True),
    "ssim": Cap("ssim", True, True),
    "msssim": Cap("msssim", True, True),
    "vmaf": Cap("vmaf", True, True),
    "pu21-psnr": Cap("pu21-psnr", True, True),
    "pu21-ssim": Cap("pu21-ssim", True, True),
    "pu21-msssim": Cap("pu21-msssim", True, True),
    "cvvdp": Cap("cvvdp", True, True),
    "cvqa-hdr-nr": Cap("cvqa-hdr-nr", False, False),
}


GROUP_METRICS = {
    "sdr": ("psnr", "xpsnr", "ssim", "msssim", "vmaf", "cvvdp", "cvqa-hdr-nr"),
    "pq": (
        "psnr",
        "xpsnr",
        "ssim",
        "msssim",
        "vmaf",
        "pu21-psnr",
        "pu21-ssim",
        "pu21-msssim",
        "cvvdp",
        "cvqa-hdr-nr",
    ),
    "hlg": ("psnr", "xpsnr", "ssim", "msssim", "pu21-psnr", "pu21-ssim", "pu21-msssim", "cvvdp"),
}


def expand_metrics(metric: str) -> tuple[str, ...]:
    if metric.startswith("group:"):
        return GROUP_METRICS[metric.split(":", 1)[1]]
    return (metric,)


def selected_caps(metric: str) -> tuple[Cap, ...]:
    return tuple(CAPS[n] for n in expand_metrics(metric))


def includes_cvqa(metric: str) -> bool:
    return "cvqa-hdr-nr" in expand_metrics(metric)


def includes_cvvdp(metric: str) -> bool:
    return "cvvdp" in expand_metrics(metric)


def includes_native(metric: str) -> bool:
    learned = {"cvvdp", "cvqa-hdr-nr"}
    return any(n not in learned for n in expand_metrics(metric))


def includes_learned(metric: str) -> bool:
    return any(n in ("cvvdp", "cvqa-hdr-nr") for n in expand_metrics(metric))


def includes_vmaf(metric: str, family: str) -> bool:
    return "vmaf" in expand_metrics(metric) and family != "hlg"
