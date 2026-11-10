"""Resolve pinned FFmpeg/ffprobe for learned-metric decoding (no PATH fallback)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PINNED_FFMPEG_TAG = "n8.1.3"
PINNED_FFMPEG_COMMIT = "1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7"
STAMP_FILENAME = "PIXVQA_LEARNED_FFMPEG_COMMIT"


class FFmpegBinError(RuntimeError):
    """Pinned FFmpeg/ffprobe is missing or incompatible."""


def _version_ok(bin_path: Path, label: str) -> bool:
    if not (bin_path.is_file() or bin_path.is_symlink()):
        return False
    # venv wrappers set LD_LIBRARY_PATH themselves; do not strip the caller env.
    try:
        proc = subprocess.run(
            [str(bin_path), "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    text = proc.stdout or proc.stderr or ""
    return PINNED_FFMPEG_TAG in text


def _stamp_ok(bin_dir: Path) -> bool:
    stamp = bin_dir.parent / STAMP_FILENAME
    if not stamp.is_file():
        return False
    try:
        return stamp.read_text(encoding="utf-8").strip() == PINNED_FFMPEG_COMMIT
    except OSError:
        return False


def _interpreter_bin_dir() -> Path | None:
    """Return the logical bin directory for sys.executable (do not follow python symlink)."""
    if not sys.executable:
        return None
    # .venv/bin/python -> /usr/bin/python3.12 must still resolve bins under .venv/bin.
    return Path(sys.executable).parent


def _candidate_bin_dirs() -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen or not resolved.is_dir():
            return
        seen.add(resolved)
        ordered.append(resolved)

    for key in (
        "PIXVQA_LEARNED_ENV_DIR",
        "VIRTUAL_ENV",
    ):
        raw = os.environ.get(key, "").strip()
        if raw:
            add(Path(raw).expanduser() / "bin")

    add(_interpreter_bin_dir())

    learned_python = os.environ.get("PIXVQA_LEARNED_PYTHON", "").strip()
    if learned_python:
        add(Path(learned_python).expanduser().parent)

    return ordered


def _resolve_one(kind: str, override_env: str) -> str:
    override = os.environ.get(override_env, "").strip()
    if override:
        path = Path(override).expanduser()
        if not _version_ok(path, kind):
            raise FFmpegBinError(
                f"{override_env}={override!r} is missing, not executable, or is not FFmpeg "
                f"{PINNED_FFMPEG_TAG}. Re-run scripts/install_learned_metrics.sh or unset the override."
            )
        return str(path.resolve())

    for bin_dir in _candidate_bin_dirs():
        candidate = bin_dir / kind
        if _stamp_ok(bin_dir) and _version_ok(candidate, kind):
            return str(candidate.resolve())

    env_hint = os.environ.get("PIXVQA_LEARNED_ENV_DIR", "<default .venv>")
    raise FFmpegBinError(
        f"pinned {kind} not found beside the learned Python environment ({env_hint}). "
        "Run scripts/install_learned_metrics.sh (or install_pixvqa.sh with learned metrics) "
        f"to provision FFmpeg {PINNED_FFMPEG_TAG} into <venv>/bin. "
        "System PATH is intentionally not searched."
    )


def resolve_ffmpeg_bin() -> str:
    return _resolve_one("ffmpeg", "PIXVQA_FFMPEG_BIN")


def resolve_ffprobe_bin() -> str:
    return _resolve_one("ffprobe", "PIXVQA_FFPROBE_BIN")


def resolve_ffprobe_legacy(env_key: str) -> str:
    """Honor a metric-specific ffprobe override, otherwise use the pinned resolver."""
    legacy = os.environ.get(env_key, "").strip()
    if legacy:
        path = Path(legacy).expanduser()
        if not _version_ok(path, "ffprobe"):
            raise FFmpegBinError(
                f"{env_key}={legacy!r} is missing, not executable, or is not FFmpeg "
                f"{PINNED_FFMPEG_TAG}. Unset {env_key} to use the pinned venv ffprobe, "
                "or re-run scripts/install_learned_metrics.sh."
            )
        return str(path.resolve())
    return resolve_ffprobe_bin()


def validate_pinned_bins(*, learned_python: str | None = None) -> None:
    """Raise FFmpegBinError when pinned ffmpeg/ffprobe are absent beside learned_python."""
    keys = ("PIXVQA_FFMPEG_BIN", "PIXVQA_FFPROBE_BIN", "PIXVQA_LEARNED_ENV_DIR")
    saved = {key: os.environ.get(key) for key in keys}
    prior = os.environ.get("PIXVQA_LEARNED_PYTHON")
    try:
        for key in keys:
            os.environ.pop(key, None)
        if learned_python:
            os.environ["PIXVQA_LEARNED_PYTHON"] = learned_python
        resolve_ffmpeg_bin()
        resolve_ffprobe_bin()
    finally:
        for key in keys:
            if saved[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved[key]
        if prior is None:
            os.environ.pop("PIXVQA_LEARNED_PYTHON", None)
        else:
            os.environ["PIXVQA_LEARNED_PYTHON"] = prior
