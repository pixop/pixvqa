"""Shared runner types for the PixVQA QA suite. Python 3.9+, standard library only."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Result:
    status: str
    detail: str = ""
    command: str = ""


@dataclass
class Case:
    id: str
    group: str
    profiles: frozenset
    tags: tuple
    func: Callable
    optional: bool = False


@dataclass
class World:
    root: Path
    tmp: Path
    cpu: Path
    swdec: Path
    full: Path
    ffmpeg: Optional[str]
    vvdec: Optional[str]
    model: Path
    vmaf_cli: Optional[Path]
    notes: list = field(default_factory=list)

    def binary(self, name: str) -> Optional[Path]:
        path = {"cpu": self.cpu, "swdec": self.swdec, "full": self.full}[name]
        return path if path.is_file() else None


def which(names: tuple) -> Optional[str]:
    for name in names:
        if name and Path(name).is_file():
            return name
        found = shutil.which(name) if name and "/" not in name else None
        if found:
            return found
    return None


def find_ffmpeg() -> Optional[str]:
    return which((os.environ.get("FFMPEG") or "", "ffmpeg"))


def find_vvdec() -> Optional[str]:
    return which((os.environ.get("VVDEC") or "", "vvdecapp"))


def find_vmaf_cli(root: Path) -> Optional[Path]:
    env = os.environ.get("VMAF_CLI")
    candidates = [
        Path(env) if env else None,
        root / "deps" / "libvmaf-74f930d" / "bin" / "vmaf",
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def run_cmd(cmd: list, timeout: float = 60, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged,
    )


def scored(text: str) -> bool:
    if re.search(r"\[ (?:PSNR|SSIM|MS-SSIM|XPSNR)\s+y:-?\d", text):
        return True
    if re.search(r"\[ PU21-[A-Z0-9-]+:-?\d", text):
        return True
    if re.search(r"\[ VMAF [a-z]+:-?\d", text):
        return True
    if re.search(r"Total frames:[1-9]", text):
        return True
    return False


def chroma_size(w: int, h: int, chroma: str) -> tuple:
    if chroma == "444":
        return w, h
    if chroma == "422":
        return w // 2, h
    return w // 2, h // 2


def write_yuv(path: Path, w: int, h: int, chroma: str, bits: int, frames: int, bias: int = 0) -> None:
    cw, ch = chroma_size(w, h, chroma)
    mask = (1 << bits) - 1
    with path.open("wb") as out:
        for i in range(frames):
            def plane(width: int, height: int, base: int) -> bytes:
                raw = bytearray()
                for row in range(height):
                    for col in range(width):
                        value = (base + bias + i * 17 + row * 3 + col * 5) & mask
                        if bits > 8:
                            raw += int(value).to_bytes(2, "little")
                        else:
                            raw.append(value)
                return bytes(raw)

            out.write(plane(w, h, 40))
            out.write(plane(cw, ch, 80))
            out.write(plane(cw, ch, 140))


def frame_bytes(w: int, h: int, chroma: str, bits: int) -> int:
    cw, ch = chroma_size(w, h, chroma)
    sample = 2 if bits > 8 else 1
    return (w * h + 2 * cw * ch) * sample
