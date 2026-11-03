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
CVVDP_JSON_JOD = re.compile(r'"jod"\s*:\s*([0-9.]+)')
TOTAL_FRAMES = re.compile(r"Total frames:(\d+)")


@dataclass
class ParsedRun:
    returncode: int
    total_frames: int | None
    psnr_avg: float | None
    ssim_avg: float | None
    pu21_psnr_avg: float | None
    cvvdp_jod: float | None
    hdr_inferred: bool
    hdr_stream: bool


def parse_run(stdout: str, stderr: str, rc: int) -> ParsedRun:
    text = stdout + stderr
    tf = TOTAL_FRAMES.search(stdout)
    psnr = _last(PSNR_SUMMARY, stdout)
    ssim = _last(SSIM_SUMMARY, stdout)
    pu21 = _last(PU21_PSNR, stdout)
    jod = CVVDP_JOD.search(stdout)
    if not jod:
        jm = CVVDP_JSON_JOD.search(stdout.strip())
        jod_val = float(jm.group(1)) if jm else None
    else:
        jod_val = float(jod.group(1))
    return ParsedRun(
        returncode=rc,
        total_frames=int(tf.group(1)) if tf else None,
        psnr_avg=float(psnr.group(4)) if psnr else None,
        ssim_avg=float(ssim.group(4)) if ssim else None,
        pu21_psnr_avg=float(pu21.group(4)) if pu21 else None,
        cvvdp_jod=jod_val,
        hdr_inferred="transfer=pq (inferred)" in stdout or "transfer=hlg (inferred)" in stdout,
        hdr_stream="(stream)" in stdout and "HDR:" in stdout,
    )


def _last(pat: re.Pattern[str], text: str):
    matches = list(pat.finditer(text))
    return matches[-1] if matches else None
