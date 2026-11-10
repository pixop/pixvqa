"""Post-install readiness checks including model load probes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pixvqa_learned_install.download import sha256_file
from pixvqa_learned_install.registry import (
    ASSETS,
    CVQA_NR_CHECKPOINT_SHA256,
    CVQA_NR_PROFILE_SHA256,
    REPOS,
    SIMPLEVQA_CHECKPOINT_SHA256,
    SIMPLEVQA_GDRIVE_ID,
)
from pixvqa_video.ffmpeg_bins import FFmpegBinError, validate_pinned_bins


class ValidationError(RuntimeError):
    pass


def _check_file(path: Path, expected: str, label: str, issues: list[str]) -> None:
    if not path.is_file():
        issues.append(f"missing {label}: {path}")
        return
    actual = sha256_file(path)
    if actual != expected:
        issues.append(f"{label} SHA-256 mismatch at {path} (got {actual[:12]}…)")


def validate_assets(root: Path) -> list[str]:
    issues: list[str] = []
    nr_repo = root / REPOS["cvqa-hdr-nr"].default_dir
    _check_file(nr_repo / "NR_HDR_VQA.pth", CVQA_NR_CHECKPOINT_SHA256, "CVQA-HDR NR checkpoint", issues)
    _check_file(nr_repo / "NR_HDR_VQA.npy", CVQA_NR_PROFILE_SHA256, "CVQA-HDR NR profile", issues)

    sv_repo = root / REPOS["simplevqa"].default_dir
    _check_file(
        sv_repo / ASSETS["simplevqa_checkpoint"].relative_path,
        SIMPLEVQA_CHECKPOINT_SHA256,
        "SimpleVQA checkpoint",
        issues,
    )

    hdr_repo = root / REPOS["hdr-chipqa"].default_dir
    for key in ("hdr_chipqa_scaler", "hdr_chipqa_svr"):
        spec = ASSETS[key]
        _check_file(hdr_repo / spec.relative_path, spec.sha256, spec.name, issues)
    return issues


def probe_cvqa_load(python: str, checkpoint: Path, profile: Path) -> str | None:
    code = """
import sys, torch, numpy as np
ckpt, prof = sys.argv[1:3]
sd = torch.load(ckpt, map_location="cpu", weights_only=True)
if not isinstance(sd, dict) or not sd:
    raise SystemExit("empty or non-dict checkpoint")
arr = np.load(prof, allow_pickle=True)
if getattr(arr, "size", 0) == 0:
    raise SystemExit("empty calibration profile")
print("ok")
"""
    try:
        proc = subprocess.run(
            [python, "-c", code, str(checkpoint), str(profile)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"CVQA-HDR NR load probe failed: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[:500]
        return f"CVQA-HDR NR load probe failed: {tail}"
    return None


def probe_simplevqa_load(python: str, checkpoint: Path) -> str | None:
    code = """
import sys, importlib.util, torch
missing = [m for m in ("torch", "torchvision", "pytorchvideo", "cv2", "PIL", "numpy")
           if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("SimpleVQA missing required imports: " + ", ".join(missing))
ckpt = sys.argv[1]
sd = torch.load(ckpt, map_location="cpu", weights_only=True)
if not isinstance(sd, dict) or not sd:
    raise SystemExit("empty or non-dict checkpoint")
print("ok")
"""
    try:
        proc = subprocess.run(
            [python, "-c", code, str(checkpoint)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"SimpleVQA load probe failed: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[:500]
        return f"SimpleVQA load probe failed: {tail}"
    return None


def probe_hdr_chipqa_load(inference_python: str, repo: Path) -> str | None:
    code = """
import sys, joblib
from pathlib import Path
repo = Path(sys.argv[1])
for name in ("hdrchipqa_livehdr_fitted_scaler.z", "hdrchipqa_livehdr_trained_svr.z"):
    obj = joblib.load(repo / name)
    if obj is None:
        raise SystemExit(f"null object in {name}")
print("ok")
"""
    try:
        proc = subprocess.run(
            [inference_python, "-c", code, str(repo)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"HDR-ChipQA SVR load probe failed: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[:500]
        return f"HDR-ChipQA SVR load probe failed: {tail}"
    return None


def probe_cvvdp_import(python: str) -> str | None:
    code = "import pycvvdp, torch; print('ok')"
    try:
        proc = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"CVVDP import probe failed: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[:500]
        return f"CVVDP import probe failed: {tail}"
    return None


def probe_ffmpeg_bins(learned_python: str) -> str | None:
    """Verify stamped ffmpeg/ffprobe beside the managed learned interpreter."""
    if not learned_python or not Path(learned_python).is_file():
        return None
    try:
        validate_pinned_bins(learned_python=learned_python)
    except FFmpegBinError as exc:
        return str(exc)
    return None


def validate_readiness(
    root: Path,
    *,
    learned_python: str,
    feature_python: str,
    inference_python: str,
    skip_load: bool = False,
    check_ffmpeg: bool = True,
) -> tuple[bool, list[str]]:
    issues = validate_assets(root)
    repo_issues = []
    for key, spec in REPOS.items():
        dest = root / spec.default_dir
        if not (dest / spec.marker).is_file():
            repo_issues.append(f"missing upstream {key} checkout: {dest}")
    issues.extend(repo_issues)

    gdrive_note = []
    sv_ckpt = root / REPOS["simplevqa"].default_dir / "ckpts" / "UGC_BVQA_model.pth"
    if not sv_ckpt.is_file():
        gdrive_note.append(
            "SimpleVQA checkpoint not verified: download from Google Drive "
            f"(id {SIMPLEVQA_GDRIVE_ID}) or rerun installer with network access."
        )

    if check_ffmpeg and learned_python:
        err = probe_ffmpeg_bins(learned_python)
        if err:
            issues.append(err)

    if skip_load:
        return not issues, issues + gdrive_note

    if learned_python and Path(learned_python).is_file():
        err = probe_cvvdp_import(learned_python)
        if err:
            issues.append(err)
        nr_repo = root / REPOS["cvqa-hdr-nr"].default_dir
        ckpt, prof = nr_repo / "NR_HDR_VQA.pth", nr_repo / "NR_HDR_VQA.npy"
        if ckpt.is_file() and prof.is_file():
            err = probe_cvqa_load(learned_python, ckpt, prof)
            if err:
                issues.append(err)
        if sv_ckpt.is_file():
            err = probe_simplevqa_load(learned_python, sv_ckpt)
            if err:
                issues.append(err)

    hdr_repo = root / REPOS["hdr-chipqa"].default_dir
    if inference_python and Path(inference_python).is_file() and hdr_repo.is_dir():
        err = probe_hdr_chipqa_load(inference_python, hdr_repo)
        if err:
            issues.append(err)

    return not issues, issues + gdrive_note
