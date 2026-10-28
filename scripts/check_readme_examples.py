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
    import array
    cw, ch = w // 2, h // 2
    mask = (1 << bits) - 1

    def plane(width: int, height: int, base: int, frame: int) -> bytes:
        origin = base + bias + frame * 17
        if bits > 8:
            values = array.array(
                "H",
                ((origin + row * 3 + col * 5) & mask
                 for row in range(height) for col in range(width)),
            )
            return values.tobytes()
        return bytes(
            (origin + row * 3 + col * 5) & mask
            for row in range(height) for col in range(width)
        )

    with path.open("wb") as out:
        for frame in range(frames):
            out.write(plane(w, h, 40, frame))
            out.write(plane(cw, ch, 80, frame))
            out.write(plane(cw, ch, 140, frame))


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
    blocks = [
        b.strip() for b in blocks
        if re.match(r"pixvqa\b", b.strip()) and "bash " not in b
    ]
    if len(blocks) < 5:
        print(f"expected the usage examples, found {len(blocks)}", file=sys.stderr)
        return 2
    ff = ffmpeg()
    fails = 0
    with tempfile.TemporaryDirectory(prefix="pixvqa_readme_") as tmp:
        td = Path(tmp)
        # User-facing README examples are 1920x1080. These clips are generated
        # here; they are not the small fixtures used by the QA profiles.
        write_yuv(td / "reference.yuv", 1920, 1080, 8, 2, 0)
        write_yuv(td / "distorted.yuv", 1920, 1080, 8, 2, 30)
        write_yuv(td / "reference10.yuv", 1920, 1080, 10, 2, 0)
        write_yuv(td / "distorted10.yuv", 1920, 1080, 10, 2, 40)
        model = ROOT / "model"
        if model.is_dir():
            for path in model.rglob("*.json"):
                dest = td / "model" / path.relative_to(model)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())
        write_yuv(td / "caller10.yuv", 320, 240, 10, 2, 0)
        write_yuv(td / "caller10b.yuv", 320, 240, 10, 2, 40)
        if ff:
            encode(ff, td / "reference.yuv", td / "reference.mp4", "yuv420p", "1920x1080", 2,
                   ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "18"])
            encode(ff, td / "distorted.yuv", td / "distorted.mp4", "yuv420p", "1920x1080", 2,
                   ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "28"])
            encode(ff, td / "reference10.yuv", td / "reference10.hevc.mp4", "yuv420p10le", "1920x1080", 2,
                   ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1", "-tag:v", "hvc1"])
            sample_dir = td / "samples"
            sample_dir.mkdir()
            encode(ff, td / "caller10.yuv", sample_dir / "ref_hevc.mp4", "yuv420p10le", "320x240", 2,
                   ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1:bframes=0", "-tag:v", "hvc1"])
            encode(ff, td / "caller10b.yuv", sample_dir / "dist_hevc.mp4", "yuv420p10le", "320x240", 2,
                   ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1:bframes=0", "-tag:v", "hvc1"])
            encode(ff, td / "caller10.yuv", sample_dir / "ref_av1.mp4", "yuv420p10le", "320x240", 2,
                   ["-c:v", "libsvtav1", "-pix_fmt", "yuv420p10le", "-preset", "10"])
            encode(ff, td / "caller10b.yuv", sample_dir / "dist_av1.mp4", "yuv420p10le", "320x240", 2,
                   ["-c:v", "libsvtav1", "-pix_fmt", "yuv420p10le", "-preset", "10"])
            encode(ff, td / "caller10.yuv", sample_dir / "ref_vvc.mp4", "yuv420p10le", "320x240", 2,
                   ["-c:v", "libvvenc", "-pix_fmt", "yuv420p10le", "-preset", "faster", "-qp", "32"])
            encode(ff, td / "caller10b.yuv", sample_dir / "dist_vvc.mp4", "yuv420p10le", "320x240", 2,
                   ["-c:v", "libvvenc", "-pix_fmt", "yuv420p10le", "-preset", "faster", "-qp", "32"])
        env = os.environ.copy()
        env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
        full = ROOT / "build" / "full" / "pixvqa"
        for i, block in enumerate(blocks):
            use = env
            if "--decode gpu" in block or ("av1" in block and full.is_file()):
                if "--decode gpu" in block and (os.environ.get("PIXVQA_CHECK_GPU") != "1" or not full.is_file()):
                    print(f"SKIP example {i} gpu")
                    continue
                if not full.is_file():
                    print(f"FAIL example {i} needs build/full/pixvqa", file=sys.stderr)
                    fails += 1
                    continue
                use = env.copy()
                use["PATH"] = str(full.parent) + os.pathsep + env.get("PATH", "")
            if ".mp4" in block and not ff:
                print(f"SKIP example {i} no ffmpeg")
                continue
            proc = subprocess.run(["bash", "-lc", block], cwd=td, capture_output=True, text=True, env=use, timeout=300)
            names = []
            found = re.search(r"--metrics\s+(\S+)", block)
            if found:
                names = found.group(1).split(",")
            labels = {
                "psnr": "[ PSNR",
                "xpsnr": "[ XPSNR",
                "ssim": "[ SSIM",
                "msssim": "[ MS-SSIM",
                "vmaf": "[ VMAF",
                "pu21-psnr": "[ PU21-PSNR",
                "pu21-ssim": "[ PU21-SSIM",
                "pu21-msssim": "[ PU21-MS-SSIM",
            }
            backends = {
                "psnr": "PSNR: cpu",
                "xpsnr": "XPSNR: cpu",
                "ssim": "SSIM: cpu",
                "msssim": "MS-SSIM: cpu",
                "vmaf": "VMAF: cpu",
                "pu21-psnr": "PU21-PSNR: cpu",
                "pu21-ssim": "PU21-SSIM: cpu",
                "pu21-msssim": "PU21-MS-SSIM: cpu",
            }
            missing = [labels[n] for n in names if n in labels and labels[n] not in proc.stdout]
            missing += [backends[n] for n in names if n in backends and backends[n] not in proc.stdout]
            if "--qcs" in block and "QCS value:" not in proc.stdout:
                missing.append("QCS value:")
            if "--decode gpu" in block and "distorted: gpu" not in proc.stdout:
                missing.append("distorted: gpu")
            if "--decode cpu" in block and "distorted: cpu" not in proc.stdout:
                missing.append("distorted: cpu")
            if proc.returncode != 0 or missing:
                fails += 1
                print(f"FAIL example {i} missing={missing}\n{block}\n{proc.stderr[-400:]}\n{proc.stdout[-500:]}", file=sys.stderr)
            else:
                print(f"OK   example {i} {','.join(names)}")
    if fails:
        print(f"{fails} examples failed", file=sys.stderr)
        return 1
    print("README examples passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
