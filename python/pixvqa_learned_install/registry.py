"""Pinned upstream references and artifact digests for learned metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Reuse adapter pins where available.
from pixvqa_cvqahdr import PINNED_COMMIT as CVQA_HDR_COMMIT
from pixvqa_cvqahdr import PINNED_UPSTREAM as CVQA_HDR_URL
from pixvqa_cvvdp import PINNED_COMMIT as CVVDP_COMMIT
from pixvqa_cvvdp import PINNED_UPSTREAM as CVVDP_URL
from pixvqa_hdr_chipqa import PINNED_UPSTREAM_COMMIT as HDR_CHIPQA_COMMIT
from pixvqa_hdr_chipqa import SCALER_NAME as HDR_CHIPQA_SCALER_NAME
from pixvqa_hdr_chipqa import SCALER_SHA256 as HDR_CHIPQA_SCALER_SHA
from pixvqa_hdr_chipqa import SVR_NAME as HDR_CHIPQA_SVR_NAME
from pixvqa_hdr_chipqa import SVR_SHA256 as HDR_CHIPQA_SVR_SHA

CVQA_NR_CHECKPOINT_SHA256 = (
    "b37659902ce96e2f49de22e5e0c865c694921e3b8e3e55b770adc3be7cac8b88"
)
CVQA_NR_PROFILE_SHA256 = (
    "601a7314e802a6867c33784355587bd892ad49990d2074cc64fe1bd7cdf6d49f"
)

SIMPLEVQA_URL = "https://github.com/sunwei925/SimpleVQA"
SIMPLEVQA_COMMIT = "017edda50e8cdbeb48113a001dbb9af0dbd2030b"
SIMPLEVQA_CHECKPOINT_NAME = "UGC_BVQA_model.pth"
SIMPLEVQA_CHECKPOINT_SHA256 = (
    "8a9546ec164953ff74698f413e859260e5f396b9c1ea646a26c309b926c0421d"
)
SIMPLEVQA_GDRIVE_ID = "137XJdq3reNMJ9tkBNqKUYTY_dTlcwXc3"

HDR_CHIPQA_URL = "https://github.com/JoshuaEbenezer/hdr_chipqa"

# Dropbox NR assets (Apache-2.0 upstream; weights linked from upstream README).
CVQA_NR_CHECKPOINT_URL = (
    "https://www.dropbox.com/scl/fi/t9auox7p47yjf0crwxyso/NR_HDR_VQA.pth"
    "?rlkey=3hpgvoiq484lt80moh9j6tmkx&dl=1"
)
CVQA_NR_PROFILE_URL = (
    "https://www.dropbox.com/scl/fi/ezsv3buh353ny3y71y4wk/NR_HDR_VQA.npy"
    "?rlkey=domijnhpjfsjp9tyvvyfp5zf9&dl=1"
)


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    commit: str
    default_dir: str
    marker: str  # file that must exist inside checkout


@dataclass(frozen=True)
class AssetSpec:
    name: str
    relative_path: str  # relative to repo root when in-repo
    sha256: str
    url: str | None
    min_bytes: int
    kind: str  # pth, npy, joblib, other


REPOS: dict[str, RepoSpec] = {
    "cvvdp": RepoSpec(
        "cvvdp",
        CVVDP_URL.rstrip(".git"),
        CVVDP_COMMIT,
        "deps/ColorVideoVDP",
        "pycvvdp/__init__.py",
    ),
    "cvqa-hdr-nr": RepoSpec(
        "cvqa-hdr-nr",
        CVQA_HDR_URL,
        CVQA_HDR_COMMIT,
        "deps/CompressedVQA-HDR",
        "NR/NR_model.py",
    ),
    "simplevqa": RepoSpec(
        "simplevqa",
        SIMPLEVQA_URL,
        SIMPLEVQA_COMMIT,
        "deps/SimpleVQA",
        "test_demo.py",
    ),
    "hdr-chipqa": RepoSpec(
        "hdr-chipqa",
        HDR_CHIPQA_URL,
        HDR_CHIPQA_COMMIT,
        "deps/hdr_chipqa",
        "hdr_chipqa.py",
    ),
}

ASSETS: dict[str, AssetSpec] = {
    "cvqa_nr_checkpoint": AssetSpec(
        "CompressedVQA-HDR NR checkpoint",
        "NR_HDR_VQA.pth",
        CVQA_NR_CHECKPOINT_SHA256,
        CVQA_NR_CHECKPOINT_URL,
        1_000_000,
        "pth",
    ),
    "cvqa_nr_profile": AssetSpec(
        "CompressedVQA-HDR NR calibration profile",
        "NR_HDR_VQA.npy",
        CVQA_NR_PROFILE_SHA256,
        CVQA_NR_PROFILE_URL,
        100,
        "npy",
    ),
    "simplevqa_checkpoint": AssetSpec(
        "SimpleVQA UGC checkpoint",
        f"ckpts/{SIMPLEVQA_CHECKPOINT_NAME}",
        SIMPLEVQA_CHECKPOINT_SHA256,
        None,  # Google Drive — handled separately
        1_000_000,
        "pth",
    ),
    "hdr_chipqa_scaler": AssetSpec(
        "HDR-ChipQA scaler",
        HDR_CHIPQA_SCALER_NAME,
        HDR_CHIPQA_SCALER_SHA,
        None,
        100,
        "joblib",
    ),
    "hdr_chipqa_svr": AssetSpec(
        "HDR-ChipQA SVR",
        HDR_CHIPQA_SVR_NAME,
        HDR_CHIPQA_SVR_SHA,
        None,
        100,
        "joblib",
    ),
}


def repo_root_paths(root: Path) -> dict[str, Path]:
    return {key: (root / spec.default_dir).resolve() for key, spec in REPOS.items()}


def default_config_paths(root: Path, learned_python: str, feature_python: str, inference_python: str) -> dict[str, str]:
    """Absolute paths for generated learned_metrics.local.toml entries."""
    cfg_dir = root / "config"
    rel = lambda p: str((root / p).resolve())  # noqa: E731
    return {
        "learned.python": learned_python,
        "cvqa-hdr-nr.repository": rel("deps/CompressedVQA-HDR"),
        "cvqa-hdr-nr.checkpoint": rel("deps/CompressedVQA-HDR/NR_HDR_VQA.pth"),
        "cvqa-hdr-nr.profile": rel("deps/CompressedVQA-HDR/NR_HDR_VQA.npy"),
        "simplevqa.repository": rel("deps/SimpleVQA"),
        "simplevqa.checkpoint": rel(f"deps/SimpleVQA/ckpts/{SIMPLEVQA_CHECKPOINT_NAME}"),
        "hdr-chipqa.repository": rel("deps/hdr_chipqa"),
        "hdr-chipqa.feature_python": feature_python,
        "hdr-chipqa.inference_python": inference_python,
    }
