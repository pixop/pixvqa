"""Build baseline candidate provenance metadata."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cli_invoke import parse_cvqa_asset_paths
from .config import ROOT
from .tolerances import catalog_export

# PyPI / wheel names tried for CVVDP backend identity (first match wins for canonical key).
CVVDP_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("cvvdp", "cvvdp_version"),
    ("pixvqa-cvvdp", "pixvqa_cvvdp_version"),
    ("colorvideovdp", "colorvideovdp_version"),
)

PYTHON_PACKAGE_SNAPSHOT = (
    "cvvdp",
    "pixvqa-cvvdp",
    "pixvqa-cvqahdr",
    "colorvideovdp",
    "torch",
    "torchvision",
    "open_clip_torch",
    "numpy",
)


def _git_head(repo: Path | None = None) -> str:
    cwd = repo or ROOT
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _fixture_digest(paths: dict[str, str]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for family, root in paths.items():
        p = Path(root)
        manifest = p / "manifest.json"
        if manifest.is_file():
            data = manifest.read_bytes()
            digests[family] = hashlib.sha256(data).hexdigest()
        else:
            digests[family] = "missing-manifest"
    return digests


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _python_package_version(python: str | None, distribution: str) -> str | None:
    """Resolve a distribution version in the configured learned interpreter only."""
    if not python:
        return None
    code = (
        "import importlib.metadata as m\n"
        f"print(m.version({distribution!r}))\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            version = proc.stdout.strip()
            return version or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _pip_versions_snapshot(python: str | None) -> dict[str, str | None]:
    return {name: _python_package_version(python, name) for name in PYTHON_PACKAGE_SNAPSHOT}


def _collect_cvvdp_package_versions(learned_python: str | None) -> dict[str, Any]:
    cvvdp: dict[str, Any] = {}
    for dist, field in CVVDP_DISTRIBUTIONS:
        ver = _python_package_version(learned_python, dist)
        if ver:
            cvvdp[field] = ver
    if not cvvdp_has_package_identity(cvvdp):
        cvvdp["package_version_missing"] = (
            "could not resolve cvvdp, pixvqa-cvvdp, or colorvideovdp version from learned python"
        )
    return cvvdp


def cvvdp_has_package_identity(cvvdp: dict[str, Any] | None) -> bool:
    if not cvvdp or not isinstance(cvvdp, dict):
        return False
    if cvvdp.get("package_version_missing"):
        return False
    return any(k.endswith("_version") and cvvdp.get(k) for k in cvvdp)


def collect_learned_provenance(
    *,
    learned_python: str | None,
    learned_config: str | None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "cvvdp": {},
        "cvqa_hdr_nr": {},
        "python_packages": {},
    }
    block["python_packages"] = _pip_versions_snapshot(learned_python)
    block["cvvdp"] = _collect_cvvdp_package_versions(learned_python)

    if not learned_config:
        block["status"] = "learned_config_not_set"
        return block

    cfg = Path(learned_config)
    block["learned_config_path"] = str(cfg)
    if not cfg.is_file():
        block["status"] = "learned_config_missing"
        return block

    cvqa_paths = parse_cvqa_asset_paths(cfg)
    cvqa: dict[str, Any] = {}
    for key, path in cvqa_paths.items():
        cvqa[f"{key}_path"] = str(path)
    checkpoint = cvqa_paths.get("checkpoint")
    profile = cvqa_paths.get("profile")
    repository = cvqa_paths.get("repository")

    if checkpoint:
        cvqa["checkpoint_sha256"] = _sha256_file(checkpoint)
        if cvqa["checkpoint_sha256"] is None:
            cvqa["checkpoint_sha256_missing"] = f"not readable: {checkpoint}"
    else:
        cvqa["checkpoint_sha256_missing"] = "checkpoint unset in config"

    if profile:
        cvqa["profile_sha256"] = _sha256_file(profile)
        if cvqa["profile_sha256"] is None:
            cvqa["profile_sha256_missing"] = f"not readable: {profile}"
    else:
        cvqa["profile_sha256_missing"] = "profile unset in config"

    if repository:
        cvqa["repository_path"] = str(repository)
        commit = _git_head(repository)
        cvqa["compressed_vqa_hdr_commit"] = commit if commit != "unknown" else None
        if cvqa["compressed_vqa_hdr_commit"] is None:
            cvqa["compressed_vqa_hdr_commit_missing"] = (
                "no git HEAD in repository path (not a checkout or git unavailable)"
            )
    else:
        cvqa["repository_missing"] = "repository unset in config"

    block["cvqa_hdr_nr"] = cvqa
    block["status"] = "collected"
    return block


def build_provenance(
    *,
    pixvqa_version: str,
    learned_python: str | None,
    learned_config: str | None,
    fixtures: dict[str, str],
    device_note: str | None = None,
    encoding_note: str = "YUV420 10-bit limited; HEVC lossy MP4 + MKV lossless per fixture README",
) -> dict[str, Any]:
    learned = collect_learned_provenance(
        learned_python=learned_python,
        learned_config=learned_config,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pixvqa_commit": _git_head(),
        "pixvqa_version": pixvqa_version,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "learned_python": learned_python,
        "learned_config": learned_config,
        "learned_provenance": learned,
        "tolerance_profile": "same_backend_repro_v1",
        "tolerances": catalog_export(),
        "fixtures": fixtures,
        "fixture_manifest_sha256": _fixture_digest(fixtures),
        "device_note": device_note,
        "encoding_note": encoding_note,
    }
