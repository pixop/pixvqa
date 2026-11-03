"""Parse PixVQA stdout for numerical assertions."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

TOTAL_FRAMES = re.compile(r"Total frames:(\d+)")

_NUMERIC_TOKEN = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PLANE_VALUE = rf"(?:{_NUMERIC_TOKEN}|inf|nan|n/a)"

PSNR_SUMMARY = re.compile(
    rf"PSNR\s+y:({_PLANE_VALUE})\s+u:({_PLANE_VALUE})\s+v:({_PLANE_VALUE})\s+avg:({_PLANE_VALUE})",
    re.I,
)
SSIM_SUMMARY = re.compile(
    rf"SSIM\s+y:({_PLANE_VALUE})\s+u:({_PLANE_VALUE})\s+v:({_PLANE_VALUE})\s+avg:({_PLANE_VALUE})",
    re.I,
)
MSSSIM_SUMMARY = re.compile(
    rf"MS-SSIM\s+y:({_PLANE_VALUE})\s+u:({_PLANE_VALUE})\s+v:({_PLANE_VALUE})\s+avg:({_PLANE_VALUE})",
    re.I,
)
XPSNR_LINE = re.compile(
    rf"\[\s*XPSNR\s+y:({_PLANE_VALUE})\s+u:({_PLANE_VALUE})\s+v:({_PLANE_VALUE})\s+"
    rf"avg:({_PLANE_VALUE})(?:\s+min:({_PLANE_VALUE}))?\s*\]",
    re.I,
)
VMAF_MEAN = re.compile(r"\[\s*VMAF\s+mean:([0-9.eE+-]+)", re.I)
PU21_PSNR = re.compile(
    rf"PU21-PSNR\s+r:({_PLANE_VALUE})\s+g:({_PLANE_VALUE})\s+b:({_PLANE_VALUE})\s+avg:({_PLANE_VALUE})",
    re.I,
)
PU21_SSIM = re.compile(
    rf"PU21-SSIM\s+y:({_PLANE_VALUE})\s+u:n/a\s+v:n/a\s+avg:({_PLANE_VALUE})",
    re.I,
)
PU21_MSSSIM = re.compile(
    rf"PU21-MS-SSIM\s+y:({_PLANE_VALUE})\s+u:n/a\s+v:n/a\s+avg:({_PLANE_VALUE})",
    re.I,
)
CVVDP_JOD = re.compile(r"\[ CVVDP jod:\s*([0-9.eE+-]+)", re.I)
CVVDP_JSON_JOD = re.compile(r'"jod"\s*:\s*([0-9.eE+-]+)')
CVQA_MIXED_SUMMARY = re.compile(
    r"\[\s*CVQA-HDR-NR\s+quality_score:([0-9.eE+-]+)",
    re.I,
)


@dataclass
class ParsedRun:
    returncode: int
    total_frames: int | None
    psnr_avg: float | None
    ssim_avg: float | None
    msssim_avg: float | None
    xpsnr_avg: float | None
    vmaf_mean: float | None
    pu21_psnr_avg: float | None
    pu21_ssim_avg: float | None
    pu21_msssim_avg: float | None
    cvvdp_jod: float | None
    cvqa_quality: float | None
    hdr_inferred: bool
    hdr_stream: bool


def _parse_json_line_objects(text: str) -> list[dict]:
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _cvvdp_jod_from_text(text: str) -> float | None:
    m = CVVDP_JOD.search(text)
    if m:
        return float(m.group(1))
    jm = CVVDP_JSON_JOD.search(text)
    if jm:
        return float(jm.group(1))
    for obj in _parse_json_line_objects(text):
        if "jod" in obj and isinstance(obj["jod"], (int, float)):
            return float(obj["jod"])
    return None


def _finite_metric_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _cvqa_quality_from_text(text: str) -> float | None:
    mixed = list(CVQA_MIXED_SUMMARY.finditer(text))
    if mixed:
        val = _finite_metric_float(mixed[-1].group(1))
        if val is not None:
            return val
    for obj in _parse_json_line_objects(text):
        metric = obj.get("metric")
        if metric not in (None, "cvqa-hdr-nr"):
            continue
        if "quality_score" not in obj:
            continue
        if metric == "cvqa-hdr-nr" or "raw_model_score" in obj:
            val = _finite_metric_float(str(obj["quality_score"]))
            if val is not None:
                return val
    return None


def _optional_avg(match: re.Match[str] | None, group: int = 4) -> float | None:
    if not match:
        return None
    return _finite_metric_float(match.group(group))


def _xpsnr_avg_from_stdout(stdout: str) -> float | None:
    matches = list(XPSNR_LINE.finditer(stdout))
    if not matches:
        return None
    return _finite_metric_float(matches[-1].group(4))


def parse_run(stdout: str, stderr: str, rc: int) -> ParsedRun:
    combined = stdout + stderr
    tf = TOTAL_FRAMES.search(stdout)
    psnr = _last(PSNR_SUMMARY, stdout)
    ssim = _last(SSIM_SUMMARY, stdout)
    msssim = _last(MSSSIM_SUMMARY, stdout)
    xpsnr_avg = _xpsnr_avg_from_stdout(stdout)
    vmaf = _last(VMAF_MEAN, stdout)
    pu21 = _last(PU21_PSNR, stdout)
    pu21_ssim = _last(PU21_SSIM, stdout)
    pu21_mss = _last(PU21_MSSSIM, stdout)
    jod_val = _cvvdp_jod_from_text(combined)
    cvqa_q = _cvqa_quality_from_text(combined)
    return ParsedRun(
        returncode=rc,
        total_frames=int(tf.group(1)) if tf else None,
        psnr_avg=_optional_avg(psnr),
        ssim_avg=_optional_avg(ssim),
        msssim_avg=_optional_avg(msssim),
        xpsnr_avg=xpsnr_avg,
        vmaf_mean=_optional_avg(vmaf, 1),
        pu21_psnr_avg=_optional_avg(pu21),
        pu21_ssim_avg=_optional_avg(pu21_ssim, 2),
        pu21_msssim_avg=_optional_avg(pu21_mss, 2),
        cvvdp_jod=jod_val if jod_val is None or math.isfinite(jod_val) else None,
        cvqa_quality=cvqa_q,
        hdr_inferred="transfer=pq (inferred)" in combined or "transfer=hlg (inferred)" in combined,
        hdr_stream="(stream)" in combined and "HDR:" in combined,
    )


def _last(pat: re.Pattern[str], text: str):
    matches = list(pat.finditer(text))
    return matches[-1] if matches else None
