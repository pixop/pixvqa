#!/usr/bin/env python3
"""Compressed inputs detect resolution, pixel format, and a constant frame rate.

Explicit -s and -f still have to match the decoded picture. Raw YUV still
requires both. Scores from the detected path must match the explicit path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "swdec" / "pixvqa"))
FULL = Path(os.environ.get("PIXVQA_FULL", ROOT / "build" / "full" / "pixvqa"))
W = H = 64
FRAMES = 4
FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def ok(msg: str) -> None:
    print("OK  ", msg)


def ffmpeg_bin() -> str | None:
    env = os.environ.get("FFMPEG") or ""
    if env and Path(env).is_file():
        return env
    found = shutil.which("ffmpeg")
    return found if found else None


def run_ff(ffmpeg: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )


def has_encoder(ffmpeg: str, name: str) -> bool:
    done = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
    return f" {name} " in done.stdout


def write_yuv(path: Path, bits: int, bias: int, frames: int = FRAMES, w: int = W, h: int = H) -> None:
    cw, ch = w // 2, h // 2
    mask = (1 << bits) - 1
    y_samples = w * h
    c_samples = cw * ch
    with path.open("wb") as out:
        for frame in range(frames):
            def samples(n: int, base: int) -> bytes:
                values = bytearray()
                for i in range(n):
                    v = (base + bias + frame * 3 + (i % 17)) & mask
                    if bits > 8:
                        values += int(v).to_bytes(2, "little")
                    else:
                        values.append(v)
                return bytes(values)

            out.write(samples(y_samples, 40))
            out.write(samples(c_samples, 80))
            out.write(samples(c_samples, 140))


def pix(args: list[str], binary: Path = BIN) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary), *args], capture_output=True, text=True)


def score_lines(text: str) -> str:
    keep = []
    for line in text.splitlines():
        if any(token in line for token in ("PSNR", "XPSNR", "SSIM", "MS-SSIM")):
            if line.startswith("pixvqa:") or line.startswith("Execution"):
                continue
            keep.append(line.strip())
    return "\n".join(keep)


def expect_reject(done: subprocess.CompletedProcess[str], needle: str, label: str) -> None:
    text = done.stderr + done.stdout
    if done.returncode == 0 or needle not in text or "Summary:" in done.stdout:
        fail(f"{label}: rc={done.returncode} missing {needle!r}\n{text[-500:]}")
        return
    ok(label)


def expect_scores(auto: list[str], explicit: list[str], label: str, binary: Path = BIN) -> None:
    a = pix(auto, binary)
    b = pix(explicit, binary)
    if a.returncode != 0 or b.returncode != 0:
        fail(f"{label}: rc auto={a.returncode} explicit={b.returncode}\n{a.stderr[-300:]}\n{b.stderr[-300:]}")
        return
    if score_lines(a.stdout) != score_lines(b.stdout) or not score_lines(a.stdout):
        fail(f"{label}: scores differ\nauto:\n{score_lines(a.stdout)}\nexplicit:\n{score_lines(b.stdout)}")
        return
    ok(label)


def encode(ffmpeg: str, src: Path, dst: Path, pix_fmt: str, codec: list[str], extra: list[str] | None = None,
           rate: str = "24", frames: int = FRAMES, size: str = f"{W}x{H}") -> bool:
    done = run_ff(
        ffmpeg,
        [
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", size, "-r", rate,
            "-i", str(src), "-frames:v", str(frames),
            *codec, "-pix_fmt", pix_fmt, *(extra or []), str(dst),
        ],
    )
    if done.returncode != 0 or not dst.is_file():
        print(done.stderr[-300:], file=sys.stderr)
        return False
    return True


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN, file=sys.stderr)
        return 2
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        print("FAIL no ffmpeg", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="pixvqa_detect_") as td:
        t = Path(td)
        raw8 = t / "raw8.yuv"
        raw8b = t / "raw8b.yuv"
        raw10 = t / "raw10.yuv"
        raw10b = t / "raw10b.yuv"
        write_yuv(raw8, 8, 0)
        write_yuv(raw8b, 8, 20)
        write_yuv(raw10, 10, 0)
        write_yuv(raw10b, 10, 30)
        write_yuv(t / "small.yuv", 8, 0, w=48, h=48)

        expect_reject(
            pix(["--metrics", "psnr", "-v", "0", "-r", str(raw8), "-t", str(raw8b)]),
            "requires -s and -f",
            "raw + raw requires -s and -f",
        )

        hevc = t / "ref.mp4"
        hevc_d = t / "dist.mp4"
        annex = t / "annex.hevc"
        hevc_ok = has_encoder(ffmpeg, "libx265") and encode(
            ffmpeg, raw10, hevc, "yuv420p10le",
            ["-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", "lossless=1:bframes=0"],
        ) and encode(
            ffmpeg, raw10b, hevc_d, "yuv420p10le",
            ["-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", "lossless=1:bframes=0"],
        )
        if not hevc_ok:
            fail("HEVC encoder did not produce a stream")
        else:
            ok("encoded HEVC")
            common = ["--metrics", "psnr,ssim", "-v", "0", "--decode", "cpu"]
            expect_scores(
                [*common, "-r", str(hevc), "-t", str(hevc_d)],
                [*common, "-r", str(hevc), "-s", f"{W}x{H}", "-f", "420p:10",
                 "-t", str(hevc_d), "-s", f"{W}x{H}", "-f", "420p:10"],
                "HEVC compressed + compressed matches explicit metadata",
            )
            expect_scores(
                ["--metrics", "psnr", "-v", "0", "--decode", "cpu",
                 "-r", str(raw10), "--input", "raw", "-s", f"{W}x{H}", "-f", "420p:10",
                 "-t", str(hevc)],
                ["--metrics", "psnr", "-v", "0", "--decode", "cpu",
                 "-r", str(raw10), "--input", "raw", "-s", f"{W}x{H}", "-f", "420p:10",
                 "-t", str(hevc), "-s", f"{W}x{H}", "-f", "420p:10"],
                "raw + HEVC matches explicit metadata",
            )
            expect_scores(
                ["--metrics", "psnr", "-v", "0", "--decode", "cpu",
                 "-r", str(hevc), "-t", str(raw10), "--input", "raw", "-s", f"{W}x{H}", "-f", "420p:10"],
                ["--metrics", "psnr", "-v", "0", "--decode", "cpu",
                 "-r", str(hevc), "-s", f"{W}x{H}", "-f", "420p:10",
                 "-t", str(raw10), "--input", "raw", "-s", f"{W}x{H}", "-f", "420p:10"],
                "HEVC + raw matches explicit metadata",
            )
            expect_reject(
                pix(["--metrics", "psnr", "-v", "0", "-r", str(raw10), "-t", str(hevc)]),
                "requires -s and -f",
                "raw + compressed still requires raw -s and -f",
            )
            expect_reject(
                pix(["--metrics", "psnr", "-v", "0", "-r", str(hevc), "-t", str(raw10)]),
                "requires -s and -f",
                "compressed + raw still requires raw -s and -f",
            )
            expect_reject(
                pix(["--metrics", "psnr", "-v", "0", "-r", str(hevc), "-s", "32x32", "-f", "420p:10",
                     "-t", str(hevc_d)]),
                "-s asks for",
                "explicit wrong resolution is rejected",
            )
            expect_reject(
                pix(["--metrics", "psnr", "-v", "0", "-r", str(hevc), "-s", f"{W}x{H}", "-f", "420p:8",
                     "-t", str(hevc_d)]),
                "-f asks for",
                "explicit wrong pixel format is rejected",
            )
            small = t / "small.mp4"
            if encode(ffmpeg, t / "small.yuv", small, "yuv420p", ["-c:v", "libx264", "-preset", "veryfast", "-qp", "20"],
                      size="48x48"):
                expect_reject(
                    pix(["--metrics", "psnr", "-v", "0", "-r", str(hevc), "-t", str(small)]),
                    "disagree",
                    "different resolutions are rejected",
                )
            else:
                fail("could not encode the smaller HEVC/AVC stream")
            eight = t / "eight.mp4"
            if encode(ffmpeg, raw8, eight, "yuv420p", ["-c:v", "libx264", "-preset", "veryfast", "-qp", "18"]):
                expect_reject(
                    pix(["--metrics", "psnr", "-v", "0", "-r", str(hevc), "-t", str(eight)]),
                    "disagree",
                    "different pixel formats are rejected",
                )
            else:
                fail("could not encode the 8-bit stream")
            expect_scores(
                ["--metrics", "xpsnr", "-v", "0", "--decode", "cpu", "-r", str(hevc), "-t", str(hevc_d)],
                ["--metrics", "xpsnr", "--fps", "24", "-v", "0", "--decode", "cpu",
                 "-r", str(hevc), "-t", str(hevc_d)],
                "HEVC XPSNR uses the constant 24 fps stream rate",
            )
            if encode(ffmpeg, raw10, annex, "yuv420p10le",
                      ["-c:v", "libx265", "-x265-params", "lossless=1:bframes=0", "-f", "hevc"]):
                expect_reject(
                    pix(["--metrics", "xpsnr", "-v", "0", "--decode", "cpu", "-r", str(annex), "-t", str(annex)]),
                    "xpsnr requires --fps",
                    "annex-B HEVC without a constant rate requires --fps",
                )
            else:
                fail("could not encode annex-B HEVC")
            rate30 = t / "rate30.mp4"
            if encode(
                ffmpeg, raw10, rate30, "yuv420p10le",
                ["-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", "lossless=1:bframes=0"],
                rate="30",
            ):
                expect_reject(
                    pix(["--metrics", "xpsnr", "-v", "0", "--decode", "cpu", "-r", str(hevc), "-t", str(rate30)]),
                    "xpsnr requires --fps",
                    "different constant frame rates require --fps",
                )
            else:
                fail("could not encode the 30 fps HEVC stream")

            pq = t / "pq.mp4"
            hlg = t / "hlg.mp4"
            bt2020 = t / "bt2020.mp4"
            vui_pq = "lossless=1:bframes=0:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited"
            vui_hlg = "lossless=1:bframes=0:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited"
            vui_2020 = "lossless=1:bframes=0:colorprim=bt2020:transfer=bt709:colormatrix=bt2020nc:range=limited"
            color_pq = ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "tv"]
            color_hlg = ["-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-color_range", "tv"]
            color_2020 = ["-color_primaries", "bt2020", "-color_trc", "bt709", "-colorspace", "bt2020nc", "-color_range", "tv"]
            if encode(ffmpeg, raw10, pq, "yuv420p10le",
                      ["-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", vui_pq], color_pq) and encode(
                    ffmpeg, raw10, hlg, "yuv420p10le",
                    ["-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", vui_hlg], color_hlg):
                pq_run = pix(["--metrics", "psnr", "-v", "0", "--decode", "cpu", "-r", str(pq), "-t", str(pq)])
                hlg_run = pix(["--metrics", "psnr", "-v", "0", "--decode", "cpu", "-r", str(hlg), "-t", str(hlg)])
                if pq_run.returncode != 0 or "transfer=pq (stream)" not in pq_run.stdout:
                    fail(f"PQ metadata was not propagated\n{pq_run.stdout[-300:]}\n{pq_run.stderr[-300:]}")
                else:
                    ok("PQ transfer comes from the stream")
                if hlg_run.returncode != 0 or "transfer=hlg (stream)" not in hlg_run.stdout:
                    fail(f"HLG metadata was not propagated\n{hlg_run.stdout[-300:]}\n{hlg_run.stderr[-300:]}")
                else:
                    ok("HLG transfer comes from the stream")
            else:
                fail("could not encode PQ or HLG HEVC")
            if encode(ffmpeg, raw10, bt2020, "yuv420p10le",
                      ["-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", vui_2020], color_2020):
                tagged = pix(["--metrics", "psnr", "-v", "0", "--decode", "cpu", "-r", str(bt2020), "-t", str(bt2020)])
                if tagged.returncode != 0 or "transfer=pq" in tagged.stdout:
                    fail(f"BT.2020 primaries were treated as PQ\n{tagged.stdout[-300:]}")
                else:
                    ok("BT.2020 primaries do not imply PQ")
            else:
                fail("could not encode the BT.2020 HEVC stream")

        if has_encoder(ffmpeg, "libaom-av1"):
            av1 = t / "av1.mp4"
            av1d = t / "av1d.mp4"
            if encode(ffmpeg, raw8, av1, "yuv420p", ["-c:v", "libaom-av1", "-cpu-used", "8", "-b:v", "0"]) and encode(
                    ffmpeg, raw8b, av1d, "yuv420p", ["-c:v", "libaom-av1", "-cpu-used", "8", "-b:v", "0"]):
                probe = pix(["--metrics", "psnr", "-v", "0", "--decode", "cpu", "-n", "1", "-r", str(av1), "-t", str(av1)])
                if "requires libdav1d" in probe.stderr or "requires libdav1d" in probe.stdout:
                    print("SKIP AV1 software decoder is not linked")
                else:
                    expect_scores(
                        ["--metrics", "psnr", "-v", "0", "--decode", "cpu", "-r", str(av1), "-t", str(av1d)],
                        ["--metrics", "psnr", "-v", "0", "--decode", "cpu",
                         "-r", str(av1), "-s", f"{W}x{H}", "-f", "420p:8",
                         "-t", str(av1d), "-s", f"{W}x{H}", "-f", "420p:8"],
                        "AV1 compressed + compressed matches explicit metadata",
                    )
            else:
                fail("AV1 encoder did not produce a stream")
        else:
            print("SKIP AV1 encoder is unavailable")

        if has_encoder(ffmpeg, "libvvenc"):
            vvc = t / "vvc.mp4"
            if encode(ffmpeg, raw10, vvc, "yuv420p10le", ["-c:v", "libvvenc", "-preset", "faster", "-qp", "32"]):
                expect_scores(
                    ["--metrics", "psnr", "-v", "0", "--decode", "cpu", "-r", str(vvc), "-t", str(vvc)],
                    ["--metrics", "psnr", "-v", "0", "--decode", "cpu",
                     "-r", str(vvc), "-s", f"{W}x{H}", "-f", "420p:10",
                     "-t", str(vvc), "-s", f"{W}x{H}", "-f", "420p:10"],
                    "VVC compressed + compressed matches explicit metadata",
                )
            else:
                print("SKIP VVC encoder failed")
        else:
            print("SKIP VVC encoder is unavailable")

        odd = t / "rgb.mkv"
        rgb = t / "rgb.raw"
        rgb.write_bytes(bytes((i * 17) & 255 for i in range(16 * 16 * 3 * 2)))
        rgb_enc = run_ff(
            ffmpeg,
            ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "16x16", "-r", "24", "-i", str(rgb),
             "-frames:v", "2", "-c:v", "ffv1", "-pix_fmt", "rgb24", str(odd)],
        )
        if rgb_enc.returncode == 0:
            expect_reject(
                pix(["--metrics", "psnr", "-v", "0", "--decode", "cpu", "-r", str(odd), "-t", str(odd)]),
                "pixel format is",
                "unsupported decoded layout is rejected",
            )
        else:
            fail("could not encode an unsupported rgb24 stream")

        if FULL.is_file() and shutil.which("nvidia-smi"):
            nv_w, nv_h = 256, 144
            nv_raw = t / "nv.yuv"
            nv = t / "nv.mov"
            write_yuv(nv_raw, 10, 0, w=nv_w, h=nv_h)
            if encode(
                ffmpeg, nv_raw, nv, "yuv420p10le",
                ["-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", "lossless=1:bframes=0"],
                size=f"{nv_w}x{nv_h}",
            ):
                expect_scores(
                    ["--metrics", "psnr", "-v", "0", "--decode", "gpu", "-r", str(nv), "-t", str(nv)],
                    ["--metrics", "psnr", "-v", "0", "--decode", "gpu",
                     "-r", str(nv), "-s", f"{nv_w}x{nv_h}", "-f", "420p:10",
                     "-t", str(nv), "-s", f"{nv_w}x{nv_h}", "-f", "420p:10"],
                    "NVDEC detection matches explicit metadata",
                    FULL,
                )
            else:
                fail("could not prepare the NVDEC HEVC stream")
        else:
            print("SKIP NVDEC binary or NVIDIA driver is unavailable")

    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
