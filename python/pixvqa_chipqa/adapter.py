"""ChipQA SDR reference adapter: upstream feature extract + bundled SVR inference."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pixvqa_chipqa import PINNED_UPSTREAM_COMMIT
from pixvqa_chipqa.inference import (
    InferenceError,
    SetupError,
    features_to_npy,
    git_revision,
    predict_from_npy,
    resolve_python,
    validate_repo,
)

_HDR_TRANSFER_REJECT = frozenset({
    "smpte2084",
    "arib-std-b67",
    "pq",
    "hlg",
    "smpte428",
})


def _run_ffprobe(ffprobe: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [ffprobe, "-v", "error", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise SetupError(f"could not launch ffprobe ({ffprobe!r}): {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        raise SetupError(f"ffprobe failed: {tail[-500:] or proc.returncode}")
    return proc


def inspect_video(path: Path, ffprobe: str = "ffprobe") -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SetupError(f"video not found: {path}")
    if path.suffix.lower() in (".yuv", ".raw"):
        raise SetupError("chipqa SDR expects compressed 8-bit video (upstream chipqa.py via OpenCV)")
    proc = _run_ffprobe(
        ffprobe,
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,color_transfer",
            "-of",
            "json",
            str(path),
        ],
    )
    try:
        streams = json.loads(proc.stdout).get("streams") or []
    except json.JSONDecodeError as exc:
        raise SetupError("ffprobe stream output is not valid JSON") from exc
    if not streams:
        raise SetupError("video has no video stream")
    stream = streams[0]
    tr_raw = stream.get("color_transfer")
    tr_norm = str(tr_raw).strip().lower() if tr_raw not in (None, "") else None
    if tr_norm in _HDR_TRANSFER_REJECT:
        raise SetupError("chipqa is SDR-only; PQ/HLG/HDR transfer rejected")
    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "transfer": tr_raw if tr_raw is not None else "unknown",
    }


def validate(
    repo: Path,
    video: Path,
    *,
    check_video: bool = True,
    ffprobe: str = "ffprobe",
    feature_python: str | None = None,
    inference_python: str | None = None,
) -> dict:
    meta = validate_repo(repo)
    feat_py = resolve_python(feature_python, "PIXVQA_CHIPQA_FEATURE_PYTHON")
    inf_py = resolve_python(inference_python, "PIXVQA_CHIPQA_INFERENCE_PYTHON", required=False)
    if check_video:
        vmeta = inspect_video(video, ffprobe=ffprobe)
    else:
        vmeta = {}
    return {
        **meta,
        "video": str(video.expanduser().resolve()),
        "input": vmeta,
        "feature_python": feat_py,
        "inference_python": inf_py or None,
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
    }


def extract_features(
    repo: Path,
    video: Path,
    features_z: Path,
    feature_python: str | None = None,
    timeout: int = 3600,
) -> None:
    repo = repo.expanduser().resolve()
    video = video.expanduser().resolve()
    features_z = features_z.expanduser().resolve()
    extractor = repo / "chipqa.py"
    if not extractor.is_file():
        raise SetupError(f"upstream chipqa.py missing: {extractor}")
    py = resolve_python(feature_python, "PIXVQA_CHIPQA_FEATURE_PYTHON")
    cmd = [py, str(extractor), "--input_file", str(video), "--results_file", str(features_z)]
    try:
        proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise InferenceError(f"ChipQA feature extraction timed out after {timeout}s") from exc
    except OSError as exc:
        raise SetupError(f"could not start ChipQA feature extraction: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise InferenceError(f"ChipQA feature extraction failed: {err[-1200:] or proc.returncode}")
    if not features_z.is_file() or features_z.stat().st_size == 0:
        raise InferenceError("ChipQA feature extraction produced no output file")


def run(
    repo: Path,
    video: Path,
    *,
    feature_python: str | None = None,
    inference_python: str | None = None,
    ffprobe: str = "ffprobe",
    timeout: int = 3600,
) -> dict:
    info = validate(repo, video, ffprobe=ffprobe, feature_python=feature_python,
                    inference_python=inference_python)
    repo = Path(info["repo"])
    video = Path(info["video"])
    feat_py = info["feature_python"]
    with tempfile.TemporaryDirectory(prefix="pixvqa_chipqa_") as tmp:
        features_z = Path(tmp) / "features.z"
        features_npy = Path(tmp) / "features.npy"
        extract_features(repo, video, features_z, feature_python=feat_py, timeout=timeout)
        features_to_npy(features_z, features_npy, feat_py)
        score, infer_meta = predict_from_npy(features_npy, repo, inference_python)
    return {
        "metric": "chipqa",
        "quality_score": score,
        "video": str(video),
        "video_metadata": info["input"],
        "upstream_repo": str(repo),
        "upstream_commit": git_revision(repo) or PINNED_UPSTREAM_COMMIT,
        "backend": "upstream-chipqa.py+testing.py",
        "temporal_semantics": "pooled-video",
        "feature_python": feat_py,
        **infer_meta,
    }
