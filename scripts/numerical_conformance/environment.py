"""Runtime discovery for numerical conformance."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import CPU_BIN, FULL_BIN


@dataclass
class Environment:
    full_bin: Path
    cpu_bin: Path
    ffmpeg: str | None
    ffprobe: str | None
    numpy_ok: bool
    cuda_available: bool
    pixvqa_version: str


def discover(full_bin: Path | None = None) -> Environment:
    fb = full_bin or FULL_BIN
    ffmpeg = shutil.which("ffmpeg") or None
    ffprobe = shutil.which("ffprobe") or None
    try:
        import numpy  # noqa: F401

        numpy_ok = True
    except ImportError:
        numpy_ok = False
    version = "unknown"
    if fb.is_file():
        proc = subprocess.run([str(fb), "--version"], capture_output=True, text=True, timeout=30)
        version = (proc.stdout + proc.stderr).strip().splitlines()[0] if proc.returncode == 0 else "unknown"
    cuda = False
    if fb.is_file():
        proc = subprocess.run([str(fb), "--help"], capture_output=True, text=True, timeout=30)
        cuda = "cuda" in (proc.stdout + proc.stderr).lower() or "gpu" in (proc.stdout + proc.stderr).lower()
    return Environment(
        full_bin=fb,
        cpu_bin=CPU_BIN,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        numpy_ok=numpy_ok,
        cuda_available=cuda,
        pixvqa_version=version,
    )


def bin_runnable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        proc = subprocess.run([str(path), "--help"], capture_output=True, text=True, timeout=20)
        return proc.returncode in (0, 1)
    except (OSError, subprocess.TimeoutExpired):
        return False
