"""Map PixVQA metric selections to parsed baseline field keys."""
from __future__ import annotations

from .metric_caps import expand_metrics, includes_vmaf
from .parse_metrics import ParsedRun

# Keys stored in result rows and baseline case entries.
FIELD_PSNR = "psnr_avg"
FIELD_SSIM = "ssim_avg"
FIELD_MSSSIM = "msssim_avg"
FIELD_XPSNR = "xpsnr_avg"
FIELD_VMAF = "vmaf_mean"
FIELD_PU21_PSNR = "pu21_psnr_avg"
FIELD_PU21_SSIM = "pu21_ssim_avg"
FIELD_PU21_MSSSIM = "pu21_msssim_avg"
FIELD_CVVDP = "cvvdp_jod"
FIELD_CVQA = "cvqa_quality"

_METRIC_TO_FIELD: dict[str, str] = {
    "psnr": FIELD_PSNR,
    "ssim": FIELD_SSIM,
    "msssim": FIELD_MSSSIM,
    "xpsnr": FIELD_XPSNR,
    "vmaf": FIELD_VMAF,
    "pu21-psnr": FIELD_PU21_PSNR,
    "pu21-ssim": FIELD_PU21_SSIM,
    "pu21-msssim": FIELD_PU21_MSSSIM,
    "cvvdp": FIELD_CVVDP,
    "cvqa-hdr-nr": FIELD_CVQA,
}

ALL_NUMERIC_FIELDS: tuple[str, ...] = tuple(_METRIC_TO_FIELD.values())


def field_for_metric(name: str) -> str | None:
    return _METRIC_TO_FIELD.get(name)


def required_fields_for_case(metric: str, family: str) -> tuple[str, ...]:
    names = expand_metrics(metric)
    fields: list[str] = []
    for name in names:
        if name == "vmaf" and not includes_vmaf(metric, family):
            continue
        key = field_for_metric(name)
        if key and key not in fields:
            fields.append(key)
    return tuple(fields)


def read_field(parsed: ParsedRun, field: str) -> float | None:
    return getattr(parsed, _PARSED_ATTR[field])


def write_fields_to_row(row: dict, parsed: ParsedRun) -> None:
    for field in ALL_NUMERIC_FIELDS:
        val = read_field(parsed, field)
        if val is not None:
            row[field] = val


def write_selected_fields_to_row(row: dict, parsed: ParsedRun, metric: str, family: str) -> None:
    for field in required_fields_for_case(metric, family):
        val = read_field(parsed, field)
        if val is not None:
            row[field] = val


def baseline_case_entry(row: dict[str, Any]) -> dict[str, Any]:
    metric = row["metric"]
    family = row["family"]
    out: dict[str, Any] = {
        "status": row["status"],
        "metric": metric,
        "family": family,
    }
    for field in required_fields_for_case(metric, family):
        if field in row:
            out[field] = row[field]
    return out


def allowed_baseline_case_keys(metric: str, family: str) -> frozenset[str]:
    return frozenset({"status", "metric", "family", *required_fields_for_case(metric, family)})


_PARSED_ATTR: dict[str, str] = {
    FIELD_PSNR: "psnr_avg",
    FIELD_SSIM: "ssim_avg",
    FIELD_MSSSIM: "msssim_avg",
    FIELD_XPSNR: "xpsnr_avg",
    FIELD_VMAF: "vmaf_mean",
    FIELD_PU21_PSNR: "pu21_psnr_avg",
    FIELD_PU21_SSIM: "pu21_ssim_avg",
    FIELD_PU21_MSSSIM: "pu21_msssim_avg",
    FIELD_CVVDP: "cvvdp_jod",
    FIELD_CVQA: "cvqa_quality",
}
