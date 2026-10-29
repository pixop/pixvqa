#!/usr/bin/env python3
"""P5 software-decoder checks against an independent FFmpeg decode.

Lossy encodes are compared with FFmpeg's decode of the same bitstream.
They are not compared with the pre-encode pictures.
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
W, H, N = 64, 64, 4
FAILS = 0
NOTES: list[str] = []


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print("FAIL", msg, file=sys.stderr)


def note(msg: str) -> None:
    NOTES.append(msg)
    print("NOTE", msg)


def find_ffmpeg() -> str | None:
    env = os.environ.get("FFMPEG")
    if env and Path(env).is_file():
        return env
    for cand in (shutil.which("ffmpeg") or "",):
        if cand and Path(cand).is_file():
            return cand
    return None


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def ffmpeg_has(ffmpeg: str, kind: str, name: str) -> bool:
    done = run([ffmpeg, "-hide_banner", f"-{kind}"])
    return f" {name} " in done.stdout or done.stdout.startswith(name) or f"\n{name} " in ("\n" + done.stdout)


def write_raw(path: Path, pix: str, frames: int = N) -> None:
    ten = pix.endswith("10le") or pix.endswith("p10le")
    chroma_w = W if "444" in pix else W // 2
    chroma_h = H if ("444" in pix or "422" in pix) else H // 2
    with path.open("wb") as f:
        for i in range(frames):
            def plane(width: int, height: int, bias: int, bits: int) -> bytes:
                mask = (1 << bits) - 1
                out = bytearray()
                for row in range(height):
                    for col in range(width):
                        value = (bias + i * 13 + row * 3 + col * 5) & mask
                        if bits > 8:
                            out += int(value).to_bytes(2, "little")
                        else:
                            out.append(value)
                return bytes(out)
            bits = 10 if ten else 8
            f.write(plane(W, H, 40, bits))
            f.write(plane(chroma_w, chroma_h, 80, bits))
            f.write(plane(chroma_w, chroma_h, 140, bits))


def encode(ffmpeg: str, src: Path, dst: Path, pix: str, codec_args: list[str], extra: list[str] | None = None) -> bool:
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{W}x{H}", "-r", "24",
        "-i", str(src), "-frames:v", str(N), *codec_args, *(extra or []), str(dst),
    ]
    done = run(cmd)
    if done.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        return False
    return True


def decode_ref(ffmpeg: str, src: Path, dst: Path, pix: str) -> bool:
    done = run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-f", "rawvideo", "-pix_fmt", pix, str(dst),
    ])
    return done.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def pix(args: list[str]) -> subprocess.CompletedProcess[str]:
    return run([str(BIN), *args])


def expect_identical(label: str, reference: Path, compressed: Path, fmt: str) -> None:
    done = pix([
        "--metrics", "psnr", "--decode", "cpu", "-n", str(N), "-v", "0",
        "-r", str(reference), "-s", f"{W}x{H}", "-f", fmt,
        "-t", str(compressed), "-s", f"{W}x{H}", "-f", fmt,
    ])
    if done.returncode != 0:
        fail(f"{label} rc={done.returncode} {done.stderr[-400:]}")
        return
    if "999.99" not in done.stdout:
        fail(f"{label} decoder mismatch\n{done.stdout[-500:]}")
        return
    if "reference: raw" not in done.stdout or "distorted: cpu" not in done.stdout:
        fail(f"{label} pairing summary\n{done.stdout}")
        return
    print(f"OK   {label}")


def try_case(ffmpeg: str, td: Path, label: str, pix_fmt: str, fmt: str, codec_args: list[str],
             container: str, extra: list[str] | None = None) -> None:
    src = td / f"{label}.src.yuv"
    enc = td / f"{label}.{container}"
    dec = td / f"{label}.dec.yuv"
    write_raw(src, pix_fmt)
    if not encode(ffmpeg, src, enc, pix_fmt, codec_args, extra):
        note(f"{label} could not be encoded with this FFmpeg build")
        return
    if not decode_ref(ffmpeg, enc, dec, pix_fmt):
        note(f"{label} was muxed, but this FFmpeg build could not decode it")
        return
    expect_identical(label, dec, enc, fmt)


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN, file=sys.stderr)
        return 2
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("SKIP no ffmpeg", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="pixvqa_p5_") as tmp:
        td = Path(tmp)
        try_case(ffmpeg, td, "h264_mp4", "yuv420p", "420p:8",
                 ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "0", "-bf", "0"], "mp4")
        try_case(ffmpeg, td, "h264_bframes", "yuv420p", "420p:8",
                 ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-qp", "20", "-bf", "2", "-g", "4"], "mp4")
        try_case(ffmpeg, td, "hevc_ts", "yuv420p10le", "420p:10",
                 ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1:bframes=0", "-tag:v", "hvc1"],
                 "ts", ["-f", "mpegts"])
        try_case(ffmpeg, td, "hevc_mov", "yuv420p10le", "420p:10",
                 ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1:bframes=0", "-tag:v", "hvc1"], "mov")
        try_case(ffmpeg, td, "hevc_avi", "yuv420p10le", "420p:10",
                 ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1:bframes=0"], "avi")
        try_case(ffmpeg, td, "ffv1_mkv", "yuv422p10le", "422p:10",
                 ["-c:v", "ffv1", "-pix_fmt", "yuv422p10le", "-level", "3"], "mkv")
        try_case(ffmpeg, td, "ffv1_444", "yuv444p", "444p:8",
                 ["-c:v", "ffv1", "-pix_fmt", "yuv444p", "-level", "3"], "mkv")
        try_case(ffmpeg, td, "mpeg2_ts", "yuv420p", "420p:8",
                 ["-c:v", "mpeg2video", "-pix_fmt", "yuv420p", "-q:v", "2"], "ts", ["-f", "mpegts"])
        try_case(ffmpeg, td, "mjpeg_avi", "yuvj420p", "420p:8",
                 ["-c:v", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", "3"], "avi")
        if ffmpeg_has(ffmpeg, "encoders", "libvpx-vp9"):
            try_case(ffmpeg, td, "vp9_webm", "yuv420p", "420p:8",
                     ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-lossless", "1", "-cpu-used", "8"], "webm")
        else:
            note("vp9 encoder libvpx-vp9 is not in this FFmpeg build")
        av1 = None
        for name in ("libaom-av1", "libsvtav1"):
            if ffmpeg_has(ffmpeg, "encoders", name):
                av1 = name
                break
        if av1:
            src = td / "av1.src.yuv"
            enc = td / "av1.mp4"
            dec = td / "av1.dec.yuv"
            write_raw(src, "yuv420p")
            codec_args = (["-c:v", av1, "-pix_fmt", "yuv420p", "-cpu-used", "8"] if av1 == "libaom-av1"
                          else ["-c:v", av1, "-pix_fmt", "yuv420p"])
            if not encode(ffmpeg, src, enc, "yuv420p", codec_args):
                note("av1_mp4 could not be encoded with this FFmpeg build")
            elif not decode_ref(ffmpeg, enc, dec, "yuv420p"):
                note("av1_mp4 was muxed, but this FFmpeg CLI could not decode it")
            else:
                done = pix([
                    "--metrics", "psnr", "--decode", "cpu", "-n", str(N), "-v", "0",
                    "-r", str(dec), "-s", f"{W}x{H}", "-f", "420p:8",
                    "-t", str(enc), "-s", f"{W}x{H}", "-f", "420p:8",
                ])
                if done.returncode != 0 or "999.99" not in done.stdout or "decoder=libdav1d" not in done.stderr:
                    fail(f"av1_mp4 rc={done.returncode} {done.stderr[-400:]} {done.stdout[-200:]}")
                else:
                    print("OK   av1_mp4")
        else:
            note("no AV1 encoder in this FFmpeg build")
        if ffmpeg_has(ffmpeg, "encoders", "prores_ks"):
            try_case(ffmpeg, td, "prores_mov", "yuv422p10le", "422p:10",
                     ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], "mov")
        else:
            note("prores_ks is not in this FFmpeg build")

        src8 = td / "mix8.yuv"
        write_raw(src8, "yuv420p")
        h264 = td / "mix.h264.mp4"
        ffv = td / "mix.ffv1.mkv"
        encode(ffmpeg, src8, h264, "yuv420p",
               ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "0", "-bf", "0", "-preset", "veryfast"])
        encode(ffmpeg, src8, ffv, "yuv420p", ["-c:v", "ffv1", "-pix_fmt", "yuv420p", "-level", "3"])
        mixed = pix([
            "--metrics", "psnr", "--decode", "cpu", "-n", str(N), "-v", "0",
            "-r", str(h264), "-s", f"{W}x{H}", "-f", "420p:8",
            "-t", str(ffv), "-s", f"{W}x{H}", "-f", "420p:8",
        ])
        if mixed.returncode != 0 or "999.99" not in mixed.stdout:
            fail(f"h264 vs ffv1 rc={mixed.returncode} {mixed.stderr[-300:]} {mixed.stdout[-300:]}")
        elif "reference: cpu" not in mixed.stdout or "distorted: cpu" not in mixed.stdout:
            fail(f"mixed summary\n{mixed.stdout}")
        else:
            print("OK   h264 mp4 versus ffv1 mkv")

        # Compressed reference versus the independent raw decode of that bitstream.
        h264_dec = td / "h264_mp4.dec.yuv"
        h264_mp4 = td / "h264_mp4.mp4"
        if h264_dec.is_file() and h264_mp4.is_file():
            swapped = pix([
                "--metrics", "psnr", "--decode", "cpu", "-n", str(N), "-v", "0",
                "-r", str(h264_mp4), "-s", f"{W}x{H}", "-f", "420p:8",
                "-t", str(h264_dec), "-s", f"{W}x{H}", "-f", "420p:8",
            ])
            if swapped.returncode != 0 or "999.99" not in swapped.stdout:
                fail(f"compressed vs raw rc={swapped.returncode} {swapped.stderr[-300:]}")
            elif "reference: cpu" not in swapped.stdout or "distorted: raw" not in swapped.stdout:
                fail(f"compressed vs raw summary\n{swapped.stdout}")
            else:
                print("OK   h264 mp4 versus its raw decode")

        hevc_ts = td / "hevc_ts.ts"
        hevc_mov = td / "hevc_mov.mov"
        if hevc_ts.is_file() and hevc_mov.is_file():
            containers = pix([
                "--metrics", "psnr", "--decode", "cpu", "-n", str(N), "-v", "0",
                "-r", str(hevc_ts), "-s", f"{W}x{H}", "-f", "420p:10",
                "-t", str(hevc_mov), "-s", f"{W}x{H}", "-f", "420p:10",
            ])
            if containers.returncode != 0 or "999.99" not in containers.stdout:
                fail(f"hevc ts vs mov rc={containers.returncode} {containers.stderr[-300:]}")
            else:
                print("OK   hevc mpeg-ts versus mov")

        if h264_mp4.is_file():
            short = td / "two.yuv"
            write_raw(short, "yuv420p", frames=2)
            mismatch = pix([
                "--metrics", "psnr", "--decode", "cpu", "-v", "0",
                "-r", str(h264_mp4), "-s", f"{W}x{H}", "-f", "420p:8",
                "-t", str(short), "-s", f"{W}x{H}", "-f", "420p:8",
            ])
            if mismatch.returncode == 0 or "frame count mismatch" not in mismatch.stderr:
                fail(f"length mismatch was not rejected: {mismatch.stderr[-300:]}")
            elif "Summary:" in mismatch.stdout:
                fail("length mismatch still printed a summary")
            else:
                print("OK   frame-count mismatch rejected")

            chroma = pix([
                "--metrics", "psnr", "--decode", "cpu",
                "-r", str(h264_mp4), "-s", f"{W}x{H}", "-f", "420p:8",
                "-t", str(h264_dec), "-s", f"{W}x{H}", "-f", "444p:8",
            ])
            if chroma.returncode == 0 or "disagree" not in chroma.stderr or "Summary:" in chroma.stdout:
                fail(f"chroma mismatch was accepted: {chroma.stderr[-300:]}")
            else:
                print("OK   4:2:0 versus 4:4:4 rejected")

            forced = pix([
                "--metrics", "psnr", "--decode", "cpu", "--input", "compressed",
                "-r", str(h264_dec), "-s", f"{W}x{H}", "-f", "420p:8",
                "-t", str(h264_dec), "-s", f"{W}x{H}", "-f", "420p:8",
            ])
            if forced.returncode == 0 or "no video stream" not in forced.stderr or "Summary:" in forced.stdout:
                fail(f"raw file was decoded as compressed: {forced.stderr[-300:]}")
            else:
                print("OK   --input compressed rejects a raw file")

        nv = td / "semi.mkv"
        raw_nv = td / "semi.yuv"
        write_raw(raw_nv, "yuv420p")
        if encode(ffmpeg, raw_nv, nv, "nv12", ["-c:v", "rawvideo", "-pix_fmt", "nv12"]):
            bad = pix([
                "--metrics", "psnr", "--decode", "cpu",
                "-r", str(nv), "-s", f"{W}x{H}", "-f", "420p:8",
                "-t", str(nv), "-s", f"{W}x{H}", "-f", "420p:8",
            ])
            if bad.returncode == 0 or "nv12" not in bad.stderr or "Summary:" in bad.stdout:
                fail(f"nv12 was accepted: {bad.stderr[-300:]}")
            else:
                print("OK   nv12 rejected by name")
        else:
            note("could not mux raw nv12")

        deep = td / "twelve.mkv"
        raw12 = td / "twelve.yuv"
        # 12-bit samples are outside the raw writer. Ask FFmpeg to make a tiny ffv1.
        made = run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=1", "-frames:v", "2",
            "-c:v", "ffv1", "-pix_fmt", "yuv420p12le", str(deep),
        ])
        if made.returncode != 0:
            note("12-bit ffv1 fixture was not produced")
        else:
            bad = pix([
                "--metrics", "psnr", "--decode", "cpu",
                "-r", str(deep), "-s", f"{W}x{H}", "-f", "420p:10",
                "-t", str(deep), "-s", f"{W}x{H}", "-f", "420p:10",
            ])
            if bad.returncode == 0 or "yuv420p12le" not in bad.stderr or "Summary:" in bad.stdout:
                fail(f"12-bit was accepted: {bad.stderr[-400:]}")
            else:
                print("OK   12-bit rejected without conversion")

        rgb = td / "rgb.nut"
        made_rgb = run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=red:s={W}x{H}:r=1", "-frames:v", "2",
            "-c:v", "rawvideo", "-pix_fmt", "rgb24", "-f", "nut", str(rgb),
        ])
        if made_rgb.returncode != 0:
            note("rgb24 rawvideo fixture was not produced: " + (made_rgb.stderr or "").strip()[-200:])
        else:
            bad = pix([
                "--metrics", "psnr", "--decode", "cpu",
                "-r", str(rgb), "-s", f"{W}x{H}", "-f", "444p:8",
                "-t", str(rgb), "-s", f"{W}x{H}", "-f", "444p:8",
            ])
            if bad.returncode == 0 or "rgb24" not in bad.stderr:
                fail(f"rgb24 was accepted: {bad.stderr[-300:]}")
            else:
                print("OK   rgb24 rejected by name")

        # HDR tags survive a container that is not the old HEVC-only path.
        tagged = td / "tagged.mp4"
        write_raw(td / "tag.yuv", "yuv420p10le")
        if encode(ffmpeg, td / "tag.yuv", tagged, "yuv420p10le", [
            "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
            "-x265-params", "lossless=1:bframes=0:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "tv",
        ]):
            hdr = pix([
                "--metrics", "psnr", "--decode", "cpu", "-n", "1", "-v", "0",
                "-r", str(tagged), "-s", f"{W}x{H}", "-f", "420p:10",
                "-t", str(tagged), "-s", f"{W}x{H}", "-f", "420p:10",
            ])
            if hdr.returncode != 0:
                fail(f"tagged hevc rc={hdr.returncode} {hdr.stderr[-300:]}")
            elif "transfer=pq (stream)" not in hdr.stdout or "primaries=bt2020 (stream)" not in hdr.stdout:
                fail(f"HDR tags were not kept:\n{hdr.stdout}")
            else:
                print("OK   PQ metadata kept from the HEVC stream")

        bad_gpu = pix([
            "--metrics", "psnr", "--decode", "gpu",
            "-r", str(h264), "-s", f"{W}x{H}", "-f", "420p:8",
            "-t", str(h264), "-s", f"{W}x{H}", "-f", "420p:8",
        ])
        text = bad_gpu.stderr
        if bad_gpu.returncode == 0 or "Summary:" in bad_gpu.stdout:
            fail("gpu request scored a non-HEVC file")
        elif "GPU decoding is not compiled" in text or "supports HEVC only" in text or "420p:10" in text:
            print("OK   explicit gpu does not fall back for H.264")
        else:
            fail(f"gpu h264 diagnostic: {text[-400:]}")

    if NOTES:
        print("Notes:")
        for item in NOTES:
            print(" -", item)
    if FAILS:
        print(FAILS, "failed", file=sys.stderr)
        return 1
    print("All FFmpeg input checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
