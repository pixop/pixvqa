"""HDR-ChipQA PQ reference adapter using upstream yuv_read (no implicit range clipping)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pixvqa_chipqa.inference import InferenceError, SetupError, git_revision
from pixvqa_hdr_chipqa import FIXTURE_LABEL, PINNED_UPSTREAM_COMMIT
from pixvqa_chipqa.inference import resolve_python
from pixvqa_hdr_chipqa.inference import features_to_npy, predict_from_npy, validate_repo

_HLG_REJECT = frozenset({"arib-std-b67", "hlg"})


def frame_bytes_yuv420p10le(width: int, height: int) -> int:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise SetupError("YUV420 requires positive even width and height")
    return width * height * 3


def validate_yuv(
    path: Path,
    width: int,
    height: int,
    *,
    bit_depth: int = 10,
    color_space: str = "BT2020",
    transfer: str | None = None,
    fps_num: int | None = None,
    fps_den: int | None = None,
) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SetupError(f"raw YUV not found: {path}")
    if bit_depth != 10:
        raise SetupError("hdr-chipqa supports 10-bit YUV420 only")
    if color_space != "BT2020":
        raise SetupError("hdr-chipqa requires BT2020 matrix (PQ-coded YUV domain)")
    tr = (transfer or "").strip().lower()
    if tr in _HLG_REJECT:
        raise SetupError("hdr-chipqa is PQ-only; HLG is unsupported until independently validated")
    if tr and tr not in ("smpte2084", "pq", "st2084", "bt2020", "unknown", "unspecified", ""):
        if "hlg" in tr or "b67" in tr:
            raise SetupError("hdr-chipqa is PQ-only; HLG rejected")
    frame_bytes = frame_bytes_yuv420p10le(width, height)
    size = path.stat().st_size
    if size <= 0 or size % frame_bytes:
        raise SetupError(
            f"{path} is not an exact multiple of {width}x{height} YUV420p10le frame size ({frame_bytes} B)"
        )
    frames = size // frame_bytes
    if frames < 5:
        raise SetupError("hdr-chipqa needs at least 5 frames for temporal buffer warmup")
    fps = fps_num / fps_den if fps_num and fps_den and fps_num > 0 and fps_den > 0 else None
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "frames_in_file": frames,
        "bit_depth": bit_depth,
        "color_space": color_space,
        "transfer": transfer or "pq-coded-yuv-domain",
        "fps": fps,
        "storage": "little-endian uint16, 10-bit codes in low bits (upstream yuv_read)",
        "fixture_note": FIXTURE_LABEL,
    }


def _inline_extract_script(
    repo: Path,
    argv: list[str],
    parent_pythonpath: str,
) -> str:
    return f"""
import os, runpy, sys, tempfile
from pathlib import Path
import pixvqa_chipqa.numpy_legacy_shim  # noqa: F401
hdr = Path({str(repo)!r})
tmp_pkg = hdr / "ChipQA" / "ChipQA_tmp"
td = Path(tempfile.mkdtemp(prefix="pixvqa-hdr-chipqa-path-"))
(td / "ChipQA").symlink_to(tmp_pkg, target_is_directory=True)
for p in (str(td), str(hdr), {parent_pythonpath!r}):
    if p not in sys.path:
        sys.path.insert(0, p)
import ChipQA.save_stats as _ss
if not hasattr(_ss, "extract_subband_feats"):
    _ss.extract_subband_feats = _ss._extract_subband_feats
os.chdir(hdr)
sys.argv = [{", ".join(repr(x) for x in argv)}]
runpy.run_path(str(hdr / "hdr_chipqa.py"), run_name="__main__")
"""


def extract_features(
    repo: Path,
    yuv: Path,
    features_z: Path,
    width: int,
    height: int,
    feature_python: str | None = None,
    timeout: int = 3600,
) -> None:
    repo = repo.expanduser().resolve()
    yuv = yuv.expanduser().resolve()
    py = resolve_python(feature_python, "PIXVQA_HDR_CHIPQA_FEATURE_PYTHON")
    parent = str(Path(__file__).resolve().parent.parent)
    argv = [
        "hdr_chipqa.py",
        "--input_file",
        str(yuv),
        "--results_file",
        str(features_z),
        "--width",
        str(width),
        "--height",
        str(height),
        "--bit_depth",
        "10",
        "--color_space",
        "BT2020",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [py, "-c", _inline_extract_script(repo, argv, parent)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise InferenceError(f"HDR-ChipQA feature extraction timed out after {timeout}s") from exc
    except OSError as exc:
        raise SetupError(f"could not start HDR-ChipQA feature extraction: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise InferenceError(f"HDR-ChipQA feature extraction failed: {err[-1500:] or proc.returncode}")
    if not features_z.is_file() or features_z.stat().st_size == 0:
        raise InferenceError("HDR-ChipQA feature extraction produced no output")


def validate(
    repo: Path,
    yuv: Path,
    width: int,
    height: int,
    *,
    transfer: str | None = None,
    fps_num: int | None = None,
    fps_den: int | None = None,
    feature_python: str | None = None,
    inference_python: str | None = None,
    check_yuv: bool = True,
) -> dict:
    meta = validate_repo(repo)
    feat_py = resolve_python(feature_python, "PIXVQA_HDR_CHIPQA_FEATURE_PYTHON")
    inf_py = resolve_python(inference_python, "PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON", required=False)
    ymeta = (
        validate_yuv(yuv, width, height, transfer=transfer, fps_num=fps_num, fps_den=fps_den)
        if check_yuv
        else {}
    )
    return {
        **meta,
        "video": str(yuv.expanduser().resolve()),
        "input": ymeta,
        "feature_python": feat_py,
        "inference_python": inf_py or None,
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
    }


def run(
    repo: Path,
    yuv: Path,
    width: int,
    height: int,
    *,
    transfer: str | None = None,
    fps_num: int | None = None,
    fps_den: int | None = None,
    feature_python: str | None = None,
    inference_python: str | None = None,
    timeout: int = 3600,
) -> dict:
    info = validate(
        repo, yuv, width, height, transfer=transfer, fps_num=fps_num, fps_den=fps_den,
        feature_python=feature_python, inference_python=inference_python,
    )
    repo = Path(info["repo"])
    yuv = Path(info["video"])
    feat_py = info["feature_python"]
    with tempfile.TemporaryDirectory(prefix="pixvqa_hdr_chipqa_") as tmp:
        features_z = Path(tmp) / "features.z"
        features_npy = Path(tmp) / "features.npy"
        extract_features(repo, yuv, features_z, width, height, feature_python=feat_py, timeout=timeout)
        features_to_npy(features_z, features_npy, feat_py)
        score, infer_meta = predict_from_npy(features_npy, repo, inference_python)
    return {
        "metric": "hdr-chipqa",
        "quality_score": score,
        "video": str(yuv),
        "video_metadata": info["input"],
        "upstream_repo": str(repo),
        "upstream_commit": git_revision(repo) or PINNED_UPSTREAM_COMMIT,
        "backend": "upstream-hdr_chipqa.py+testing.py",
        "temporal_semantics": "pooled-video",
        "color_space": "BT2020",
        "transfer": "PQ-coded YUV domain (no EOTF in reader)",
        "feature_python": feat_py,
        **infer_meta,
    }
