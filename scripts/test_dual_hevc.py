#!/usr/bin/env python3
"""Short dual-input HEVC checks. Scores only; no timing."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "swdec" / "pixvqa"))
W, H, N = 64, 64, 4
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def find_ffmpeg() -> str | None:
    env = os.environ.get("FFMPEG")
    if env and Path(env).is_file():
        return env
    for cand in (shutil.which("ffmpeg") or "",):
        if cand and Path(cand).is_file():
            return cand
    return None


def write_yuv(path: Path, bias: int) -> None:
    with path.open("wb") as f:
        for i in range(N):
            y = bytearray()
            for row in range(H):
                for col in range(W):
                    v = (80 + bias + i * 17 + row * 3 + col) & 0x3FF
                    y += v.to_bytes(2, "little")
            u = bytearray()
            vch = bytearray()
            for row in range(H // 2):
                for col in range(W // 2):
                    cu = (400 + bias + i + row) & 0x3FF
                    cv = (600 + i + col) & 0x3FF
                    u += cu.to_bytes(2, "little")
                    vch += cv.to_bytes(2, "little")
            f.write(y)
            f.write(u)
            f.write(vch)


def run_ff(ffmpeg: str, args: list[str]) -> None:
    p = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(p.stderr or p.stdout or "ffmpeg failed")


def encode(ffmpeg: str, src: Path, dst: Path, frames: int = N) -> None:
    ext = dst.suffix.lower()
    common = [
        "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}", "-r", "24",
        "-i", str(src), "-frames:v", str(frames),
        "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-profile:v", "main10",
        "-x265-params", "lossless=1:bframes=0",
    ]
    if ext == ".hevc":
        run_ff(ffmpeg, [*common, "-f", "hevc", str(dst)])
    else:
        run_ff(ffmpeg, [*common, "-tag:v", "hvc1", str(dst)])


def decode_raw(ffmpeg: str, src: Path, dst: Path) -> None:
    run_ff(ffmpeg, ["-i", str(src), "-f", "rawvideo", "-pix_fmt", "yuv420p10le", str(dst)])


def pix(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(BIN), *args], capture_output=True, text=True)


def sides(ref: Path, dist: Path, fmt: str = "420p:10") -> list[str]:
    return ["-r", str(ref), "-s", f"{W}x{H}", "-f", fmt, "-t", str(dist), "-s", f"{W}x{H}", "-f", fmt]


def score_text(out: str) -> str:
    keep = []
    for line in out.splitlines():
        if line.startswith("Summary") or line.startswith("---") or line.startswith("Total"):
            continue
        if any(k in line for k in ("PSNR", "SSIM", "XPSNR", "MS-SSIM")) and "Execution" not in line and not line.startswith("  "):
            keep.append(line.strip())
    return "\n".join(keep)


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN, file=sys.stderr)
        return 2
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("SKIP no ffmpeg encoder")
        return 2
    with tempfile.TemporaryDirectory(prefix="pixvqa_p3_") as td:
        t = Path(td)
        ref = t / "ref.yuv"
        dist = t / "dist.yuv"
        write_yuv(ref, 0)
        write_yuv(dist, 40)
        ref_h, dist_h = t / "ref.hevc", t / "dist.hevc"
        ref_m, dist_m = t / "ref.mp4", t / "dist.mp4"
        ref_k, dist_k = t / "ref.mkv", t / "dist.mkv"
        short_h = t / "short.hevc"
        encode(ffmpeg, ref, ref_h)
        encode(ffmpeg, dist, dist_h)
        encode(ffmpeg, ref, ref_m)
        encode(ffmpeg, dist, dist_m)
        encode(ffmpeg, ref, ref_k)
        encode(ffmpeg, dist, dist_k)
        encode(ffmpeg, ref, short_h, frames=2)
        ref_dec, dist_dec = t / "ref.dec.yuv", t / "dist.dec.yuv"
        decode_raw(ffmpeg, ref_h, ref_dec)
        decode_raw(ffmpeg, dist_h, dist_dec)
        if ref_dec.read_bytes() != ref.read_bytes() or dist_dec.read_bytes() != dist.read_bytes():
            fail("ffmpeg lossless decode does not match the source YUV")
        else:
            print("OK   lossless elementary round-trip")

        base = ["--metrics", "psnr,ssim,msssim", "--threads", "2"]
        raw = pix([*base, *sides(ref_dec, dist_dec)])
        if raw.returncode != 0:
            fail(f"raw/raw rc={raw.returncode} {raw.stderr}")
            return 1
        if "reference: raw" not in raw.stdout or "distorted: raw" not in raw.stdout:
            fail(f"raw summary\n{raw.stdout}")
        else:
            print("OK   raw/raw identified as raw")
        if "decoder threads" in raw.stdout:
            fail("raw run claimed codec threads")
        else:
            print("OK   raw run has no decoder threads")
        expect = score_text(raw.stdout)

        pairs = [
            ("hevc/hevc", ref_h, dist_h, "cpu", "cpu"),
            ("raw/hevc", ref_dec, dist_h, "raw", "cpu"),
            ("hevc/raw", ref_h, dist_dec, "cpu", "raw"),
            ("mp4/mp4", ref_m, dist_m, "cpu", "cpu"),
            ("mkv/mkv", ref_k, dist_k, "cpu", "cpu"),
            ("mp4/hevc", ref_m, dist_h, "cpu", "cpu"),
            ("hevc/mkv", ref_h, dist_k, "cpu", "cpu"),
            ("raw/mkv", ref_dec, dist_k, "raw", "cpu"),
        ]
        for name, a, b, ra, rb in pairs:
            p = pix([*base, "--decode", "cpu", *sides(a, b)])
            if p.returncode != 0:
                fail(f"{name} rc={p.returncode} {p.stderr}")
                continue
            if f"reference: {ra}" not in p.stdout or f"distorted: {rb}" not in p.stdout:
                fail(f"{name} summary\n{p.stdout}")
                continue
            got = score_text(p.stdout)
            if got != expect:
                fail(f"{name} scores differ\n{got}\n!=\n{expect}")
            else:
                print(f"OK   {name}")

        same = pix([*base, "--decode", "auto", *sides(ref_h, ref_h)])
        if same.returncode != 0 or "999.99" not in same.stdout:
            fail(f"identical hevc/hevc {same.returncode} {same.stderr} {same.stdout[-400:]}")
        elif "reference: cpu" not in same.stdout or "8 per input" not in same.stdout:
            fail(f"auto hevc summary\n{same.stdout}")
        else:
            print("OK   identical elementary pair and auto software decode")

        mismatch = pix([*base, "--decode", "cpu", *sides(ref_h, short_h)])
        if mismatch.returncode == 0 or "frame count mismatch" not in mismatch.stderr:
            fail(f"length mismatch rc={mismatch.returncode} {mismatch.stderr}")
        else:
            print("OK   frame-count mismatch")

        bad_size = pix([*base, "--decode", "cpu", "-r", str(ref_h), "-s", "32x32", "-f", "420p:10",
                        "-t", str(dist_h), "-s", "32x32", "-f", "420p:10"])
        if bad_size.returncode == 0 or "-s asks for" not in bad_size.stderr:
            fail(f"size reject rc={bad_size.returncode} {bad_size.stderr}")
        else:
            print("OK   resolution mismatch")

        gpu = pix([*base, "--decode", "gpu", *sides(ref_h, dist_h)])
        if gpu.returncode == 0 or "GPU decoding is not compiled" not in gpu.stderr:
            fail(f"gpu request rc={gpu.returncode} {gpu.stderr}")
        elif "PSNR" in gpu.stdout:
            fail("gpu request produced scores")
        else:
            print("OK   explicit gpu is not replaced by software")

        forced = pix(["--metrics", "psnr", "--decode", "cpu", "--threads", "4", *sides(ref, dist)])
        if forced.returncode != 0:
            fail(f"cpu on raw rc={forced.returncode} {forced.stderr}")
        elif "reference: raw" not in forced.stdout or "distorted: raw" not in forced.stdout:
            fail(f"cpu on raw summary\n{forced.stdout}")
        elif "decoder threads" in forced.stdout:
            fail("cpu on raw claimed decoder threads")
        else:
            print("OK   --decode cpu on raw stays raw")

        mpeg = t / "mpeg4.mp4"
        run_ff(ffmpeg, [
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-i", str(ref),
            "-frames:v", "2", "-c:v", "mpeg4", str(mpeg),
        ])
        other = pix(["--metrics", "psnr", "--decode", "cpu", *sides(mpeg, mpeg)])
        if other.returncode == 0 or "-f asks for" not in other.stderr:
            fail(f"8-bit mpeg4 declared as 10-bit rc={other.returncode} {other.stderr}")
        elif "Summary results" in other.stdout:
            fail("bit-depth mismatch produced a score")
        else:
            print("OK   8-bit stream is not treated as 10-bit")

        bits = pix(["--metrics", "psnr", "--decode", "cpu", *sides(ref_h, dist_h, "420p:8")])
        if bits.returncode == 0 or "-f asks for" not in bits.stderr:
            fail(f"10-bit hevc as 8-bit rc={bits.returncode} {bits.stderr}")
        else:
            print("OK   10-bit stream is not treated as 8-bit")

        threads = pix(["--metrics", "psnr", "--decode", "cpu", "--threads", "4", *sides(ref_h, dist_h)])
        if "threads: 4" not in threads.stdout or "decoder threads: 8 per input (fixed)" not in threads.stdout:
            fail(f"thread summary\n{threads.stdout}")
        else:
            print("OK   metric threads 4 and decoder pools stay 8")

    if FAILS:
        print(f"{FAILS} dual-input checks failed", file=sys.stderr)
        return 1
    print("All dual-input checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
