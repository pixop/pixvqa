#!/usr/bin/env python3
"""Run the bash examples in README.md against generated fixtures."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_yuv(path: Path, w: int, h: int, bits: int, frames: int, bias: int) -> None:
    cw, ch = w // 2, h // 2
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


def ffmpeg() -> str | None:
    env = os.environ.get("FFMPEG")
    if env and Path(env).is_file():
        return env
    return shutil.which("ffmpeg")


def encode(ff: str, src: Path, dst: Path, pix: str, size: str, frames: int, args: list[str]) -> bool:
    done = subprocess.run(
        [ff, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", pix, "-s", size, "-r", "24", "-i", str(src),
         "-frames:v", str(frames), *args, str(dst)],
        capture_output=True, text=True,
    )
    return done.returncode == 0 and dst.is_file()


def main() -> int:
    binary = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "swdec" / "pixvqa"))
    if not binary.is_file():
        binary = ROOT / "build" / "cpu" / "pixvqa"
    if not binary.is_file():
        print("no pixvqa binary", file=sys.stderr)
        return 2
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.S)
    blocks = [b.strip() for b in blocks if "pixvqa " in b and not b.strip().startswith("export") and "bash " not in b]
    if len(blocks) < 8:
        print(f"expected the usage examples, found {len(blocks)}", file=sys.stderr)
        return 2
    ff = ffmpeg()
    fails = 0
    with tempfile.TemporaryDirectory(prefix="pixvqa_readme_") as tmp:
        td = Path(tmp)
        write_yuv(td / "reference.yuv", 64, 64, 8, 2, 0)
        write_yuv(td / "distorted.yuv", 64, 64, 8, 2, 30)
        write_yuv(td / "reference10.yuv", 64, 64, 10, 2, 0)
        write_yuv(td / "distorted10.yuv", 64, 64, 10, 2, 40)
        write_yuv(td / "vmaf_ref.yuv", 320, 240, 8, 2, 0)
        write_yuv(td / "vmaf_dist.yuv", 320, 240, 8, 2, 25)
        write_yuv(td / "clip.av1.yuv", 64, 64, 8, 2, 0)
        write_yuv(td / "clip.vvc.yuv", 64, 64, 10, 2, 0)
        write_yuv(td / "clip.hevc.yuv", 256, 256, 10, 2, 0)
        model = ROOT / "model"
        if model.is_dir():
            (td / "model").mkdir()
            for path in model.glob("*.json"):
                (td / "model" / path.name).write_bytes(path.read_bytes())
        if ff:
            encode(ff, td / "reference.yuv", td / "clip.h264.mp4", "yuv420p", "64x64", 2,
                   ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "0"])
            encode(ff, td / "clip.av1.yuv", td / "clip.av1.mp4", "yuv420p", "64x64", 2,
                   ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-cpu-used", "8"])
            raw = (td / "clip.av1.yuv").read_bytes()
            # The AV1 example compares an independent decode, not the pre-encode pictures.
            dec = subprocess.run(
                [ff, "-y", "-hide_banner", "-loglevel", "error", "-c:v", "libdav1d", "-i", str(td / "clip.av1.mp4"),
                 "-f", "rawvideo", "-pix_fmt", "yuv420p", str(td / "clip.av1.yuv")],
                capture_output=True, text=True,
            )
            if dec.returncode != 0:
                (td / "clip.av1.yuv").write_bytes(raw)
            encode(ff, td / "clip.vvc.yuv", td / "clip.vvc.mp4", "yuv420p10le", "64x64", 2,
                   ["-c:v", "libvvenc", "-preset", "faster", "-qp", "28"])
            vraw = (td / "clip.vvc.yuv").read_bytes()
            vdec = subprocess.run(
                [ff, "-y", "-hide_banner", "-loglevel", "error", "-c:v", "vvc", "-i", str(td / "clip.vvc.mp4"),
                 "-f", "rawvideo", "-pix_fmt", "yuv420p10le", str(td / "clip.vvc.yuv")],
                capture_output=True, text=True,
            )
            if vdec.returncode != 0:
                (td / "clip.vvc.yuv").write_bytes(vraw)
            encode(ff, td / "clip.hevc.yuv", td / "clip.hevc.mp4", "yuv420p10le", "256x256", 2,
                   ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1", "-tag:v", "hvc1"])
            hraw = (td / "clip.hevc.yuv").read_bytes()
            hdec = subprocess.run(
                [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(td / "clip.hevc.mp4"),
                 "-f", "rawvideo", "-pix_fmt", "yuv420p10le", str(td / "clip.hevc.yuv")],
                capture_output=True, text=True,
            )
            if hdec.returncode != 0:
                (td / "clip.hevc.yuv").write_bytes(hraw)
        env = os.environ.copy()
        env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
        full = ROOT / "build" / "full" / "pixvqa"
        for i, block in enumerate(blocks):
            use = env
            if "--decode gpu" in block:
                if os.environ.get("PIXVQA_CHECK_GPU") != "1" or not full.is_file():
                    print(f"SKIP example {i} gpu")
                    continue
                use = env.copy()
                use["PATH"] = str(full.parent) + os.pathsep + env.get("PATH", "")
            if "clip." in block and not ff:
                print(f"SKIP example {i} no ffmpeg")
                continue
            proc = subprocess.run(["bash", "-lc", block], cwd=td, capture_output=True, text=True, env=use, timeout=180)
            if proc.returncode != 0:
                fails += 1
                print(f"FAIL example {i}\n{block}\n{proc.stderr[-400:]}", file=sys.stderr)
            else:
                print(f"OK   example {i}")
    if fails:
        print(f"{fails} examples failed", file=sys.stderr)
        return 1
    print("README examples passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
