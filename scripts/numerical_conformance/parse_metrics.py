"""Parse PixVQA stdout for numerical assertions."""
from __future__ import annotations

import json
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


def _cvqa_quality_from_text(text: str) -> float | None:
    for obj in _parse_json_line_objects(text):
        if obj.get("metric") == "cvqa-hdr-nr" and "quality_score" in obj:
            return float(obj["quality_score"])
        if "quality_score" in obj and "raw_model_score" in obj:
            return float(obj["quality_score"])
    return None


def parse_run(stdout: str, stderr: str, rc: int) -> ParsedRun:
    combined = stdout + stderr
    tf = TOTAL_FRAMES.search(stdout)
    psnr = _last(PSNR_SUMMARY, stdout)
    ssim = _last(SSIM_SUMMARY, stdout)
    pu21 = _last(PU21_PSNR, stdout)
    jod_val = _cvvdp_jod_from_text(combined)
    cvqa_q = _cvqa_quality_from_text(stdout)
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
