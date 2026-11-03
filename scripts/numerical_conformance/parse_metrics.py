"""Parse PixVQA stdout for numerical assertions."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

PSNR_SUMMARY = re.compile(
    r"PSNR\s+y:([0-9.]+)\s+u:([0-9.]+)\s+v:([0-9.]+)\s+avg:([0-9.]+)"
)
SSIM_SUMMARY = re.compile(
    r"SSIM\s+y:([0-9.]+)\s+u:([0-9.]+)\s+v:([0-9.]+)\s+avg:([0-9.]+)"
)
PU21_PSNR = re.compile(
    r"PU21-PSNR\s+r:([0-9.]+)\s+g:([0-9.]+)\s+b:([0-9.]+)\s+avg:([0-9.]+)"
)
CVVDP_JOD = re.compile(r"\[ CVVDP jod:\s*([0-9.]+)", re.I)
CVVDP_JSON_JOD = re.compile(r'"jod"\s*:\s*([0-9.eE+-]+)')
CVQA_MIXED_SUMMARY = re.compile(
    r"\[\s*CVQA-HDR-NR\s+quality_score:([0-9.eE+-]+)",
    re.I,
)
TOTAL_FRAMES = re.compile(r"Total frames:(\d+)")


@dataclass
class ParsedRun:
    returncode: int
    total_frames: int | None
    psnr_avg: float | None
    ssim_avg: float | None
    pu21_psnr_avg: float | None
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


def parse_run(stdout: str, stderr: str, rc: int) -> ParsedRun:
    combined = stdout + stderr
    tf = TOTAL_FRAMES.search(stdout)
    psnr = _last(PSNR_SUMMARY, stdout)
    ssim = _last(SSIM_SUMMARY, stdout)
    pu21 = _last(PU21_PSNR, stdout)
    jod_val = _cvvdp_jod_from_text(combined)
    cvqa_q = _cvqa_quality_from_text(combined)
    return ParsedRun(
        returncode=rc,
        total_frames=int(tf.group(1)) if tf else None,
        psnr_avg=float(psnr.group(4)) if psnr else None,
        ssim_avg=float(ssim.group(4)) if ssim else None,
        pu21_psnr_avg=float(pu21.group(4)) if pu21 else None,
        cvvdp_jod=jod_val,
        cvqa_quality=cvqa_q,
        hdr_inferred="transfer=pq (inferred)" in combined or "transfer=hlg (inferred)" in combined,
        hdr_stream="(stream)" in combined and "HDR:" in combined,
    )


def _last(pat: re.Pattern[str], text: str):
    matches = list(pat.finditer(text))
    return matches[-1] if matches else None
