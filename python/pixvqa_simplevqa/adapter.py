"""Run the upstream SimpleVQA single-video demo with explicit model provenance.

No PyTorch code is vendored. The official demo is executed in a temporary,
isolated working directory to satisfy its fixed relative ckpts/ path.
This is a parity baseline, not a native implementation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


class SetupError(RuntimeError):
    """Missing/invalid inputs or required runtime dependencies."""


class InferenceError(RuntimeError):
    """The upstream implementation did not complete with a valid score."""


REQUIRED_IMPORTS = ("torch", "torchvision", "pytorchvideo", "cv2", "PIL", "numpy")
MODEL_FILES = ("test_demo.py", "model/UGC_BVQA_model.py")
SCORE_RE = re.compile(r"^\s*(.+),\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_video(path: Path, ffprobe: str = "ffprobe") -> dict:
    if not path.is_file():
        raise SetupError(f"video not found: {path}")
    if path.suffix.lower() in (".yuv", ".raw"):
        raise SetupError("upstream SimpleVQA demo expects compressed video, not raw YUV")
    try:
        proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,avg_frame_rate,codec_name,width,height,color_transfer",
         "-of", "json", str(path)],
        capture_output=True, text=True,
        )
    except OSError as exc:
        raise SetupError(f"could not launch ffprobe: {exc}") from exc
    if proc.returncode:
        raise SetupError(f"ffprobe failed: {proc.stderr[-500:]}")
    try:
        streams = json.loads(proc.stdout)["streams"]
        stream = streams[0]
        rate = stream["avg_frame_rate"].split("/")
        fps = float(rate[0]) / float(rate[1])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("invalid frame rate")
    except (KeyError, IndexError, ValueError, ZeroDivisionError, TypeError) as exc:
        raise SetupError("video metadata has no valid video stream/frame rate") from exc
    # Upstream uses int(round(fps)) and int(frame_count / rounded_fps).
    # Short sequences cause a zero-size temporal branch or invalid padding.
    count = stream.get("nb_frames")
    if count not in (None, "N/A"):
        try:
            if int(count) < int(round(fps)):
                raise SetupError("upstream SimpleVQA requires at least one full second of video")
        except ValueError as exc:
            raise SetupError("invalid video frame count") from exc
    return {"codec": stream.get("codec_name"), "width": stream.get("width"),
            "height": stream.get("height"), "fps": fps,
            "frame_count": int(count) if count not in (None, "N/A") else None,
            "transfer": stream.get("color_transfer", "unknown")}


def validate(repo: Path, checkpoint: Path, video: Path, *,
             check_imports: bool = True, probe_video: bool = True,
             ffprobe: str = "ffprobe") -> dict:
    repo, checkpoint, video = (p.expanduser().resolve() for p in (repo, checkpoint, video))
    if not repo.is_dir():
        raise SetupError(f"upstream SimpleVQA checkout not found: {repo}")
    for name in MODEL_FILES:
        if not (repo / name).is_file():
            raise SetupError(f"upstream SimpleVQA missing {name}")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise SetupError(f"SimpleVQA trained checkpoint missing/empty: {checkpoint}")
    if check_imports:
        missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
        if missing:
            raise SetupError("missing required Python modules: " + ", ".join(missing))
    meta = inspect_video(video, ffprobe=ffprobe) if probe_video else {}
    # Upstream uses cv2 and BGR/RGB conversion with no HDR normalization.
    if meta.get("transfer") in ("smpte2084", "arib-std-b67"):
        raise SetupError("SimpleVQA reference backend supports SDR only; PQ/HLG rejected")
    return {"repo": str(repo), "checkpoint": str(checkpoint), "video": str(video), "input": meta}


def parse_score(text: str, expected_video: Path) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise InferenceError("upstream result must contain exactly one video score")
    match = SCORE_RE.fullmatch(lines[0])
    if not match or Path(match.group(1)).resolve() != expected_video.resolve():
        raise InferenceError("upstream result does not match the requested video")
    score = float(match.group(2))
    if not math.isfinite(score):
        raise InferenceError("upstream predicted a non-finite score")
    return score


def run(repo: Path, checkpoint: Path, video: Path, *,
        device: str = "cpu", ffprobe: str = "ffprobe",
        timeout: int = 3600) -> dict:
    if device not in ("cpu", "cuda"):
        raise SetupError("device must be cpu or cuda")
    if timeout <= 0:
        raise SetupError("timeout must be positive")
    info = validate(repo, checkpoint, video, ffprobe=ffprobe)
    repo, checkpoint, video = Path(info["repo"]), Path(info["checkpoint"]), Path(info["video"])
    # Check dependencies before launching the heavyweight upstream demo.
    # The demo internally loads slowfast_r50(pretrained=True); users must
    # separately provision its pretrained checkpoint/cache. Never suppress
    # failures or silently initialize randomly.
    with tempfile.TemporaryDirectory(prefix="pixvqa_simplevqa_") as tmp:
        root = Path(tmp)
        (root / "test_demo.py").symlink_to(repo / "test_demo.py")
        (root / "model").symlink_to(repo / "model", target_is_directory=True)
        (root / "ckpts").mkdir()
        (root / "ckpts" / "UGC_BVQA_model.pth").symlink_to(checkpoint)
        out = root / "result.txt"
        argv = [sys.executable, str(root / "test_demo.py"),
                "--method_name", "single-scale", "--dist", str(video),
                "--output", str(out)]
        if device == "cuda":
            argv.append("--is_gpu")
        try:
            proc = subprocess.run(argv, cwd=root, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise InferenceError(f"SimpleVQA upstream timed out after {timeout}s") from exc
        if proc.returncode:
            raise InferenceError(
                f"SimpleVQA upstream failed (exit {proc.returncode}): {proc.stderr[-1000:]}")
        if not out.is_file():
            raise InferenceError("SimpleVQA upstream produced no result file")
        score = parse_score(out.read_text(encoding="utf-8"), video)
    return {
        "metric": "simplevqa", "quality_score": score, "device": device,
        "checkpoint_sha256": digest(checkpoint),
        "upstream_repo": str(repo), "upstream_commit": git_revision(repo),
        "video": str(video), "video_metadata": info["input"],
        "backend": "upstream-test_demo", "temporal_semantics": "pooled-video",
        "slowfast_pretrained_source": "upstream PyTorchVideo pretrained cache/download",
    }


def git_revision(repo: Path) -> str | None:
    proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None
