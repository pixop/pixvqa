#!/usr/bin/env python3
"""Run --help, --version, and the published pixvqa examples.

README.md and docs/CLI_USAGE.md are both read. `pixvqa` and a path whose final
component is pixvqa, including `./bin/pixvqa`, are the same command. Installer
scripts and examples that name files this harness does not create are not
counted. At least six self-contained examples must remain.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "build" / "cpu" / "pixvqa"
CUDA = ROOT / "build" / "cuda" / "pixvqa"
SW = ROOT / "build" / "swdec" / "pixvqa"
FULL = ROOT / "build" / "full" / "pixvqa"
DOCS = (ROOT / "README.md", ROOT / "docs" / "CLI_USAGE.md")
# A command token: bare pixvqa, ./bin/pixvqa, or any path ending in /pixvqa.
INVOCATION = re.compile(r"(^|\n)([ \t]*)((?:\.\./|\./|/)?(?:[\w.-]+/)*)pixvqa\b")
OUTPUT_FLAGS = {"--cvvdp-output-json", "--cvqahdr-output-json"}
# Inputs the fixture builder below actually writes. An example that names
# anything else needs a private checkout, checkpoint, or clip.
FIXTURES = {
    "reference.yuv", "distorted.yuv", "reference10.yuv", "distorted10.yuv",
    "reference.mp4", "distorted.mp4", "reference10.hevc.mp4",
    "samples/ref_hevc.mp4", "samples/dist_hevc.mp4",
    "samples/ref_av1.mp4", "samples/dist_av1.mp4",
    "samples/ref_vvc.mp4", "samples/dist_vvc.mp4",
    "model/vmaf_v1.0.16_3d0h.json",
    "model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json",
}
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def run(bin_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(bin_path), *args], capture_output=True, text=True, timeout=60)


def collapsed(text: str) -> str:
    """Join help lines so a wrapped sentence still matches as one phrase."""
    return re.sub(r"\s+", " ", text)


def check_help(bin_path: Path, kind: str) -> None:
    done = run(bin_path, ["--help"])
    text = done.stdout
    if done.returncode != 0:
        fail(f"{kind} help rc {done.returncode}")
        return
    flat = collapsed(text)
    help_lines = len(text.splitlines())
    if not 40 <= help_lines <= 60:
        fail(f"{kind} help length {help_lines}, expected 40-60 lines")
        return
    for needle in (
        "Input", "Metrics", "Execution", "QCS", "HDR", "General",
        "--metrics", "msssim, vmaf", "pu21-psnr, pu21-ssim, pu21-msssim",
        "--qcs-range", "--metric-device", "--decode", "--threads",
        "--model PATH", "--fps", "--frames", "--input",
        "docs/QCS.md", "docs/PU21.md", "--help-hdr",
        "[30,50]", "[70,100]",
    ):
        if needle not in text and needle not in flat:
            fail(f"{kind} help missing {needle}")
            return
    if "NVIDIA" in text or "GeForce" in text:
        fail(f"{kind} help initialized a device")
        return
    short = run(bin_path, ["-h"])
    if short.returncode != 0 or short.stdout != text:
        fail(f"{kind} -h disagrees with --help")
        return
    hdr = run(bin_path, ["--help-hdr"])
    if hdr.returncode != 0 or "--hdr-transfer" not in hdr.stdout:
        fail(f"{kind} --help-hdr missing --hdr-transfer rc={hdr.returncode}")
        return
    hdr_flat = collapsed(hdr.stdout)
    hdr_lines = len(hdr.stdout.splitlines())
    if not 20 <= hdr_lines <= 30:
        fail(f"{kind} --help-hdr length {hdr_lines}, expected 20-30 lines")
        return
    for needle in (
        "docs/PU21.md", "--hdr-primaries", "--hdr-matrix", "--hdr-range",
        "--hdr-peak-nits", "--hdr-black-nits", "--hdr-hlg-gamma",
        "--hdr-chroma-loc", "--hdr-chroma-up",
        "1000 cd/m2", "HLG black", "gamma 1.2", "--ref-hdr-transfer",
        "st2084", "stdb67", "--hlg-peak", "bt601",
        "per-input option, shared option, stream metadata",
    ):
        if needle not in hdr.stdout and needle not in hdr_flat:
            fail(f"{kind} --help-hdr missing {needle}")
            return
    if "1.2 + 0.42" in hdr_flat or "10.1109/PCS50896.2021.9477471" in hdr.stdout:
        fail(f"{kind} --help-hdr still contains the formula or the paper citation")
        return
    print(f"OK   {kind} help")


def check_version(bin_path: Path, kind: str) -> None:
    done = run(bin_path, ["--version"])
    if done.returncode != 0:
        fail(f"{kind} version rc {done.returncode} {done.stderr}")
        return
    if "pixvqa version 0.3.0-dev" not in done.stdout or f"build: {kind}" not in done.stdout:
        fail(f"{kind} version text {done.stdout!r}")
        return
    if "decode: none" not in done.stdout:
        fail(f"{kind} version missing decode capability {done.stdout!r}")
        return
    if "NVIDIA" in done.stdout or "GeForce" in done.stdout:
        fail(f"{kind} version initialized a device")
        return
    print(f"OK   {kind} version")


def write_yuv(path: Path, width: int, height: int, bits: int, frames: int, bias: int) -> None:
    cw, ch = width // 2, height // 2
    mask = (1 << bits) - 1
    dtype = np.uint16 if bits > 8 else np.uint8
    with path.open("wb") as out:
        for frame in range(frames):
            def plane(pw: int, ph: int, base: int) -> None:
                yy, xx = np.mgrid[0:ph, 0:pw]
                values = (base + bias + frame * 17 + yy * 3 + xx * 5) & mask
                values.astype(dtype).tofile(out)
            plane(width, height, 40)
            plane(cw, ch, 80)
            plane(cw, ch, 140)


def executable_for(block: str) -> Path:
    if "--decode gpu" in block:
        return FULL
    # Software AV1 needs libdav1d. The NVDEC binary still detects the picture.
    if "av1" in block and FULL.is_file():
        return FULL
    if "--decode" in block or "--input compressed" in block or ".mp4" in block:
        return SW
    return CPU


def with_executable(block: str, binary: Path) -> str:
    """Replace every pixvqa invocation with an absolute path. Do not rely on PATH."""
    quoted = shlex.quote(str(binary))
    rewritten, count = INVOCATION.subn(lambda m: f"{m.group(1)}{m.group(2)}{quoted}", block)
    if count < 1:
        raise RuntimeError(f"example does not invoke pixvqa:\n{block}")
    return rewritten


def input_paths(text: str) -> list[str]:
    tokens = text.split()
    found = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token in OUTPUT_FLAGS:
            skip = True
            continue
        if re.search(r"\.(?:mp4|yuv|mkv|json|pth|npy)$", token):
            found.append(token)
    return found


def examples() -> list[str]:
    """Self-contained published commands from the README and the CLI guide.

    README.md has three ./bin/pixvqa demonstrations. The command catalog is
    docs/CLI_USAGE.md, which says `pixvqa` means `./bin/pixvqa` or PATH.
    Installer scripts and examples that name external assets stay out: this
    harness cannot create a private model path or a clip it does not generate.
    """
    chosen = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for raw in re.findall(r"```bash\n(.*?)```", text, flags=re.S):
            block = raw.strip()
            if not block or "bash " in block or block.startswith("export") or block.startswith("git "):
                continue
            if INVOCATION.match(block) is None:
                continue
            paths = input_paths(block)
            if not paths or any(item not in FIXTURES for item in paths):
                continue
            chosen.append(block)
    return chosen


def main() -> int:
    if not CPU.exists() or not CUDA.exists():
        print("build both binaries first", file=sys.stderr)
        return 2
    check_help(CPU, "cpu")
    check_help(CUDA, "cuda")
    check_version(CPU, "cpu")
    check_version(CUDA, "cuda")
    bad = run(CPU, ["--not-a-real-option"])
    if bad.returncode == 0 or "PARSE ERROR" not in bad.stderr:
        fail(f"unknown option rc={bad.returncode} {bad.stderr}")
    else:
        print("OK   unknown option")
    cpu_cuda = run(CPU, ["--metric-device", "gpu", "--metrics", "psnr", "-r", "a", "-s", "16x16", "-f", "420p:10",
                         "-t", "b", "-s", "16x16", "-f", "420p:10"])
    if cpu_cuda.returncode == 0:
        fail("CPU build accepted --metric-device gpu")
    else:
        print("OK   cpu build rejects cuda kernel")
    blocks = examples()
    if len(blocks) < 6:
        fail(f"expected the published examples, found {len(blocks)}")
    env = os.environ.copy()
    ffmpeg_bin = os.environ.get("FFMPEG")
    with tempfile.TemporaryDirectory(prefix="cli_help_") as tmp:
        td = Path(tmp)
        write_yuv(td / "reference.yuv", 1920, 1080, 8, 2, 0)
        write_yuv(td / "distorted.yuv", 1920, 1080, 8, 2, 30)
        write_yuv(td / "reference10.yuv", 1920, 1080, 10, 2, 0)
        write_yuv(td / "distorted10.yuv", 1920, 1080, 10, 2, 40)
        model_src = ROOT / "model"
        model_dst = td / "model"
        if model_src.is_dir():
            for p in model_src.rglob("*.json"):
                dest = model_dst / p.relative_to(model_src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
        write_yuv(td / "caller10.yuv", 320, 240, 10, 2, 0)
        write_yuv(td / "caller10b.yuv", 320, 240, 10, 2, 40)
        if ffmpeg_bin and Path(ffmpeg_bin).is_file():
            (td / "samples").mkdir()
            # The README CVVDP example treats reference.mp4 as BT.2020 PQ.
            # FFmpeg 8 keeps transfer and primaries only when the frames carry them.
            for src, dst, pix, size, codec, extra in (
                ("reference.yuv", "reference.mp4", "yuv420p", "1920x1080", "libx264",
                 ["-vf", "setparams=color_trc=smpte2084:color_primaries=bt2020:colorspace=bt2020nc:range=tv",
                  "-preset", "veryfast", "-qp", "18",
                  "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                  "-colorspace", "bt2020nc", "-color_range", "tv"]),
                ("distorted.yuv", "distorted.mp4", "yuv420p", "1920x1080", "libx264",
                 ["-vf", "setparams=color_trc=smpte2084:color_primaries=bt2020:colorspace=bt2020nc:range=tv",
                  "-preset", "veryfast", "-qp", "28",
                  "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                  "-colorspace", "bt2020nc", "-color_range", "tv"]),
                ("reference10.yuv", "reference10.hevc.mp4", "yuv420p10le", "1920x1080", "libx265",
                 ["-x265-params", "lossless=1", "-tag:v", "hvc1"]),
                ("caller10.yuv", "samples/ref_hevc.mp4", "yuv420p10le", "320x240", "libx265",
                 ["-x265-params", "lossless=1:bframes=0", "-tag:v", "hvc1"]),
                ("caller10b.yuv", "samples/dist_hevc.mp4", "yuv420p10le", "320x240", "libx265",
                 ["-x265-params", "lossless=1:bframes=0", "-tag:v", "hvc1"]),
                ("caller10.yuv", "samples/ref_av1.mp4", "yuv420p10le", "320x240", "libsvtav1", ["-preset", "10"]),
                ("caller10b.yuv", "samples/dist_av1.mp4", "yuv420p10le", "320x240", "libsvtav1", ["-preset", "10"]),
                ("caller10.yuv", "samples/ref_vvc.mp4", "yuv420p10le", "320x240", "libvvenc",
                 ["-preset", "faster", "-qp", "32"]),
                ("caller10b.yuv", "samples/dist_vvc.mp4", "yuv420p10le", "320x240", "libvvenc",
                 ["-preset", "faster", "-qp", "32"]),
            ):
                subprocess.run(
                    [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                     "-f", "rawvideo", "-pix_fmt", pix, "-s", size, "-r", "24",
                     "-i", str(td / src), "-frames:v", "2", "-c:v", codec, "-pix_fmt", pix, *extra,
                     str(td / dst)],
                    check=True,
                )
        for i, block in enumerate(blocks):
            binary = executable_for(block)
            if not binary.is_file():
                fail(f"example {i} missing binary {binary}")
                continue
            if ".mp4" in block and not (ffmpeg_bin and Path(ffmpeg_bin).is_file()):
                fail(f"example {i} needs FFMPEG to build the compressed fixture")
                continue
            command = with_executable(block, binary)
            proc = subprocess.run(["bash", "-lc", command], cwd=td, capture_output=True, text=True,
                                  env=env, timeout=180)
            if proc.returncode != 0:
                fail(f"example {i} rc={proc.returncode} bin={binary}\n{command}\n{proc.stderr[-400:]}")
            else:
                print(f"OK   example {i} {binary.name}")
    if FAILS:
        print(f"{FAILS} CLI checks failed", file=sys.stderr)
        return 1
    print("All CLI help checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
