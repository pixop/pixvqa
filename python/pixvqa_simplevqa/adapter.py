"""Run the upstream SimpleVQA single-video demo with explicit model provenance.

No PyTorch code is vendored. The official demo is executed in a temporary,
isolated working directory to satisfy its fixed relative ckpts/ path.
This is a parity baseline, not a native implementation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
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
# Pip distribution names for metadata.version (no import of torch / cv2 required).
_RUNTIME_DIST_NAMES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("pytorchvideo", "pytorchvideo"),
    ("numpy", "numpy"),
    ("opencv_python", "opencv-python"),
    ("pillow", "Pillow"),
)
MODEL_FILES = ("test_demo.py", "model/UGC_BVQA_model.py")
SCORE_RE = re.compile(r"^\s*(.+),\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")

# PQ / HLG and common alias strings from ffprobe color_transfer (SDR-only reference).
_HDR_TRANSFER_REJECT = frozenset({
    "smpte2084",
    "arib-std-b67",
    "pq",
    "hlg",
    "smpte428",
})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_stack_versions() -> dict[str, str | None]:
    """Best-effort versions for experiment manifests (pip metadata only; no heavy imports).

    Values are None when the distribution is not installed in the current interpreter
    (e.g. minimal CI). CUDA / compiled feature flags are not queried here.
    """
    versions: dict[str, str | None] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    for key, dist in _RUNTIME_DIST_NAMES:
        try:
            versions[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _run_ffprobe(ffprobe: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke ffprobe; OSError and failures become SetupError."""
    cmd = [ffprobe, "-v", "error", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SetupError(f"could not launch ffprobe ({ffprobe!r}): {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        if len(tail) > 500:
            tail = tail[-500:]
        raise SetupError(f"ffprobe failed ({' '.join(cmd[:6])}…): {tail or proc.returncode}")
    return proc


def _parse_fps(stream: dict) -> float:
    rate = str(stream.get("avg_frame_rate") or "").split("/")
    if len(rate) != 2:
        raise ValueError("avg_frame_rate")
    fps = float(rate[0]) / float(rate[1])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps")
    return fps


def _parse_duration_seconds(raw: object) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "n/a":
        return None
    try:
        d = float(s)
    except ValueError:
        return None
    if not math.isfinite(d) or d < 0:
        return None
    return d


def _parse_nb_frames(raw: object) -> int | None:
    """Frame count from stream metadata, or None if unknown / N/A."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().upper() == "N/A":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n < 0:
        return None
    return n


def _format_duration_seconds(format_block: dict) -> float | None:
    """Prefer floating `format.duration` from ffprobe JSON."""
    val = format_block.get("duration")
    got = _parse_duration_seconds(val)
    if got is not None:
        return got
    tags = format_block.get("tags")
    if isinstance(tags, dict):
        for tag in ("DURATION", "duration"):
            if tag not in tags:
                continue
            s = str(tags[tag]).strip()
            if not s:
                continue
            # e.g. 0:00:01.23 or plain seconds
            if s.count(":") >= 2:
                parts = s.split(":")
                try:
                    sec = float(parts[-1])
                    total = sec + int(parts[-2]) * 60
                    if len(parts) >= 3:
                        total += int(parts[-3]) * 3600
                    if math.isfinite(total) and total >= 0:
                        return total
                except (ValueError, TypeError, IndexError):
                    pass
            else:
                got2 = _parse_duration_seconds(s)
                if got2 is not None:
                    return got2
    return None


def inspect_video(path: Path, ffprobe: str = "ffprobe") -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SetupError(f"video not found: {path}")
    suffix = path.suffix.lower()
    # Upstream reads via OpenCV; extension-only raw containers are unsupported (ambiguous transport).
    if suffix in (".yuv", ".raw"):
        raise SetupError(
            "upstream SimpleVQA demo expects compressed video with a known codec, not raw YUV/raw pixel"
        )

    stream_proc = _run_ffprobe(
        ffprobe,
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate,codec_name,width,height,color_transfer",
            "-of",
            "json",
            str(path),
        ],
    )
    try:
        stream_root = json.loads(stream_proc.stdout)
    except json.JSONDecodeError as exc:
        raise SetupError("ffprobe stream output is not valid JSON") from exc
    if not isinstance(stream_root, dict):
        raise SetupError("ffprobe stream output JSON root is not an object")
    streams = stream_root.get("streams")
    if not isinstance(streams, list) or not streams:
        raise SetupError("video has no video stream for SimpleVQA preflight")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise SetupError("ffprobe stream record is not a JSON object")

    format_proc = _run_ffprobe(
        ffprobe,
        ["-show_entries", "format=duration", "-of", "json", str(path)],
    )
    try:
        fmt_root = json.loads(format_proc.stdout)
    except json.JSONDecodeError as exc:
        raise SetupError("ffprobe format output is not valid JSON") from exc
    if not isinstance(fmt_root, dict):
        raise SetupError("ffprobe format output JSON root is not an object")
    fmt_val = fmt_root.get("format")
    if fmt_val is None:
        fmt: dict = {}
    elif isinstance(fmt_val, dict):
        fmt = fmt_val
    else:
        raise SetupError("ffprobe format record is not a JSON object")
    duration_s = _format_duration_seconds(fmt)

    try:
        fps = _parse_fps(stream)
    except (KeyError, IndexError, ValueError, ZeroDivisionError, TypeError) as exc:
        raise SetupError("video metadata has no valid video stream/frame rate") from exc

    rounded_fps = max(1, int(round(fps)))
    nb_parsed = _parse_nb_frames(stream.get("nb_frames"))
    if stream.get("nb_frames") not in (None, "N/A") and nb_parsed is None:
        raise SetupError("invalid or malformed nb_frames in video stream metadata")

    # Reject sub-second or invalid clips when duration is known.
    if duration_s is not None:
        if duration_s <= 0:
            raise SetupError("invalid video duration (non-positive)")
        if duration_s < 1.0:
            raise SetupError("upstream SimpleVQA requires at least one full second of video (duration < 1s)")

    if nb_parsed is not None:
        if nb_parsed < rounded_fps:
            raise SetupError("upstream SimpleVQA requires at least one full second of video (frame count too low)")
    elif duration_s is not None:
        approx_frames = duration_s * fps
        if approx_frames < float(rounded_fps):
            raise SetupError("upstream SimpleVQA requires at least one full second of video (estimated frames too low)")
    else:
        raise SetupError(
            "cannot verify video length for SimpleVQA preflight (missing nb_frames and duration)"
        )

    tr_raw = stream.get("color_transfer")
    tr_norm = str(tr_raw).strip().lower() if tr_raw not in (None, "") else None
    if tr_norm in _HDR_TRANSFER_REJECT:
        raise SetupError("SimpleVQA reference backend supports SDR only; PQ/HLG rejected")
    # Missing/unknown transfer: still allow SDR workflows; HDR must not be labeled as unknown in practice.

    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "frame_count": nb_parsed,
        "duration_seconds": duration_s,
        "transfer": tr_raw if tr_raw is not None else "unknown",
    }


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
    tr = meta.get("transfer")
    if isinstance(tr, str) and tr.strip().lower() in _HDR_TRANSFER_REJECT:
        raise SetupError("SimpleVQA reference backend supports SDR only; PQ/HLG rejected")
    out: dict[str, object] = {
        "repo": str(repo),
        "checkpoint": str(checkpoint),
        "video": str(video),
        "input": meta,
        "runtime_stack": runtime_stack_versions(),
    }
    return out


def parse_score(text: str, expected_video: Path) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise InferenceError("upstream result must contain exactly one video score line")
    match = SCORE_RE.fullmatch(lines[0])
    if not match:
        raise InferenceError("upstream result line is not valid path,score CSV")
    try:
        score = float(match.group(2))
    except ValueError as exc:
        raise InferenceError("upstream score is not a valid number") from exc
    if not math.isfinite(score):
        raise InferenceError("upstream predicted a non-finite score")
    path_a = Path(match.group(1)).expanduser().resolve()
    path_b = expected_video.expanduser().resolve()
    if path_a != path_b:
        raise InferenceError("upstream result does not match the requested video path")
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
    video_in = str(video)
    # Check dependencies before launching the heavyweight upstream demo.
    # The demo internally loads slowfast_r50(pretrained=True); that call may
    # trigger PyTorchVideo/hub download and large full-video allocations — not
    # initiated by this wrapper, but not safe on huge inputs. Never suppress
    # failures or silently accept a randomly initialized backbone.
    with tempfile.TemporaryDirectory(prefix="pixvqa_simplevqa_") as tmp:
        root = Path(tmp)
        try:
            (root / "test_demo.py").symlink_to(repo / "test_demo.py")
            (root / "model").symlink_to(repo / "model", target_is_directory=True)
        except OSError as exc:
            raise SetupError(f"could not stage upstream symlinks: {exc}") from exc
        (root / "ckpts").mkdir()
        try:
            (root / "ckpts" / "UGC_BVQA_model.pth").symlink_to(checkpoint)
        except OSError as exc:
            raise SetupError(f"could not stage checkpoint symlink: {exc}") from exc
        out = root / "result.txt"
        argv = [sys.executable, str(root / "test_demo.py"),
                "--method_name", "single-scale", "--dist", video_in,
                "--output", str(out)]
        if device == "cuda":
            argv.append("--is_gpu")
        try:
            proc = subprocess.run(argv, cwd=root, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            err = (exc.stderr or "").strip()
            if len(err) > 800:
                err = err[-800:]
            raise InferenceError(
                f"SimpleVQA upstream timed out after {timeout}s" + (f": {err}" if err else "")
            ) from exc
        except OSError as exc:
            raise SetupError(f"could not start upstream SimpleVQA process: {exc}") from exc
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            out_snip = (proc.stdout or "").strip()
            if len(err) > 1200:
                err = err[-1200:]
            if not err and out_snip:
                if len(out_snip) > 600:
                    out_snip = out_snip[-600:]
                err = f"(stderr empty; stdout tail) {out_snip}"
            raise InferenceError(
                f"SimpleVQA upstream failed (exit {proc.returncode}): {err or 'no error output'}")
        if not out.is_file():
            raise InferenceError("SimpleVQA upstream produced no result file")
        try:
            score = parse_score(out.read_text(encoding="utf-8"), video)
        except (OSError, UnicodeError) as exc:
            raise InferenceError(f"could not read upstream result file: {exc}") from exc
    return {
        "metric": "simplevqa", "quality_score": score, "device": device,
        "checkpoint_sha256": digest(checkpoint),
        "upstream_repo": str(repo), "upstream_commit": git_revision(repo),
        "video": str(video), "video_metadata": info["input"],
        "backend": "upstream-test_demo", "temporal_semantics": "pooled-video",
        "slowfast_pretrained_source": "upstream PyTorchVideo pretrained cache/download",
        "runtime_stack": info.get("runtime_stack") or runtime_stack_versions(),
    }


def git_revision(repo: Path) -> str | None:
    proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None
