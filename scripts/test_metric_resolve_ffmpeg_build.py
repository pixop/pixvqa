#!/usr/bin/env python3
"""Regression: FFmpeg-enabled metric_resolve in full/swdec binaries.

Verifies that CVVDP can infer PQ transfer from compressed stream metadata
(without --hdr-transfer), and that full builds link metric_resolve compiled
with PIXVQA_HAS_FFMPEG (not the CUDA-only build/cuda/metric_resolve.o).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("PIXVQA_TEST_FFMPEG") or shutil.which("ffmpeg")
FFMPEG_DISABLED = "cannot read transfer from compressed metadata without an FFmpeg-enabled pixvqa build"


def fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}", file=sys.stderr)
    raise SystemExit(1)


def ok(name: str) -> None:
    print(f"OK   {name}")


def write_yuv(path: Path, frames: int = 2, size: int = 64) -> None:
    path.write_bytes(b"\x00\x00" * (size * size * 3 // 2 * frames))


def make_pq_mp4(path: Path) -> None:
    assert FFMPEG
    proc = subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=24",
            "-frames:v",
            "2",
            "-vf",
            "setparams=color_trc=smpte2084:color_primaries=bt2020:colorspace=bt2020nc:range=tv",
            "-c:v",
            "libx264",
            "-qp",
            "28",
            "-pix_fmt",
            "yuv420p10le",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
            "-color_range",
            "tv",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        fail("fixtures", proc.stderr[-300:])


def cvvdp_mock_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PIXVQA_LEARNED_CONFIG"] = "none"
    env["PYTHONPATH"] = str(tmp_path / "mock")
    mock = tmp_path / "mock" / "pixvqa_cvvdp"
    mock.mkdir(parents=True)
    (mock / "__init__.py").write_text("", encoding="utf-8")
    (mock / "__main__.py").write_text(
        "import sys\n"
        "if '--check' in sys.argv:\n"
        "  print('pixvqa-check ok'); raise SystemExit(0)\n"
        "print('{\"jod\":1}\\n')\n",
        encoding="utf-8",
    )
    return env


def run_cvvdp_metadata_inference(pixvqa: Path, label: str) -> None:
    if not pixvqa.is_file():
        print(f"SKIP {label} (missing {pixvqa})")
        return
    if not FFMPEG:
        print(f"SKIP {label} (ffmpeg not found)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = tmp_path / "ref.yuv"
        pq = tmp_path / "test.mp4"
        write_yuv(raw)
        make_pq_mp4(pq)
        env = cvvdp_mock_env(tmp_path)
        base = [
            str(pixvqa),
            "--metrics",
            "cvvdp",
            "-r",
            str(raw),
            "-t",
            str(pq),
            "-s",
            "64x64",
            "-f",
            "420p:10",
            "--fps",
            "24/1",
            "--cvvdp-python",
            sys.executable,
        ]
        proc = subprocess.run(base, capture_output=True, text=True, env=env, timeout=120)
        if FFMPEG_DISABLED in proc.stderr:
            fail(label, f"metric_resolve lacks FFmpeg at link/compile time:\n{proc.stderr[-600:]}")
        if proc.returncode != 0:
            fail(label, f"rc={proc.returncode}\n{proc.stderr}\n{proc.stdout}")
        if "pixvqa-cvvdp: info: interpretation=pq" not in proc.stderr:
            fail(f"{label}-log", proc.stderr[-500:])
        ok(label)


def check_full_object_uses_ffmpeg() -> None:
    obj = ROOT / "build" / "full" / "metric_resolve.o"
    stale = ROOT / "build" / "cuda" / "metric_resolve.o"
    if not obj.is_file():
        print("SKIP full-metric_resolve-object (run make full first)")
        return
    nm = subprocess.run(["nm", str(obj)], capture_output=True, text=True, check=False)
    if nm.returncode != 0:
        print("SKIP full-metric_resolve-object (nm unavailable)")
        return
    if "probe_stream_hdr" not in nm.stdout:
        fail("full-metric_resolve-object", f"{obj} does not reference probe_stream_hdr")
    ok("full-metric_resolve-object-ffmpeg")
    if stale.is_file():
        stale_nm = subprocess.run(["nm", str(stale)], capture_output=True, text=True, check=False)
        if stale_nm.returncode == 0 and "probe_stream_hdr" in stale_nm.stdout:
            fail(
                "cuda-metric_resolve-stale",
                f"{stale} must stay FFmpeg-free for make cuda; rebuild after make cuda",
            )
        ok("cuda-metric_resolve-object-still-ffmpeg-free")


def main() -> int:
    swdec = Path(os.environ.get("PIXVQA_DECODE_BIN", ROOT / "build" / "swdec" / "pixvqa"))
    full = Path(os.environ.get("PIXVQA_FULL_BIN", ROOT / "build" / "full" / "pixvqa"))
    run_cvvdp_metadata_inference(swdec, "swdec-cvvdp-compressed-transfer-inference")
    run_cvvdp_metadata_inference(full, "full-cvvdp-compressed-transfer-inference")
    check_full_object_uses_ffmpeg()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
