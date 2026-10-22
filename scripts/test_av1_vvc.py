#!/usr/bin/env python3
"""AV1 libdav1d, AV1 NVDEC, VVC software, and a fresh HEVC NVDEC check.

Film grain stays at libdav1d's default (apply it only when the bitstream
carries it). These encodes do not write a film-grain header. The operating
point stays the single default point. Pixel equality is PSNR 999.99 against
an independent decode of the same bitstream.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "swdec" / "pixvqa"))
FULL = Path(os.environ.get("PIXVQA_FULL", ROOT / "build" / "full" / "pixvqa"))
W, H, N = 256, 256, 4
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


def write_raw(path: Path, pix: str, frames: int = N, bias: int = 40) -> None:
    ten = "10" in pix
    chroma_w = W // 2
    chroma_h = H // 2
    with path.open("wb") as f:
        for i in range(frames):
            def plane(width: int, height: int, base: int, bits: int) -> bytes:
                mask = (1 << bits) - 1
                out = bytearray()
                for row in range(height):
                    for col in range(width):
                        value = (base + i * 17 + row * 3 + col * 5) & mask
                        if bits > 8:
                            out += int(value).to_bytes(2, "little")
                        else:
                            out.append(value)
                return bytes(out)
            bits = 10 if ten else 8
            f.write(plane(W, H, bias, bits))
            f.write(plane(chroma_w, chroma_h, bias + 40, bits))
            f.write(plane(chroma_w, chroma_h, bias + 90, bits))


def encode(ffmpeg: str, src: Path, dst: Path, pix: str, args: list[str], frames: int = N) -> bool:
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{W}x{H}", "-r", "24",
        "-i", str(src), "-frames:v", str(frames), *args, str(dst),
    ]
    done = run(cmd)
    return done.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def decode_ref(ffmpeg: str, src: Path, dst: Path, pix: str, decoder: list[str] | None = None) -> bool:
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *(decoder or []),
           "-i", str(src), "-f", "rawvideo", "-pix_fmt", pix, str(dst)]
    done = run(cmd)
    return done.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def pix(binary: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return run([str(binary), *args])


def expect_match(label: str, binary: Path, reference: Path, compressed: Path, fmt: str,
                 decoder: str, backend: str, frames: int = N) -> subprocess.CompletedProcess[str] | None:
    done = pix(binary, [
        "--metrics", "psnr", "--decode", "cpu" if backend == "software" else "gpu",
        "--metric-device", "cpu", "-v", "0",
        "-r", str(reference), "-s", f"{W}x{H}", "-f", fmt,
        "-t", str(compressed), "-s", f"{W}x{H}", "-f", fmt,
    ])
    text = done.stdout + done.stderr
    if done.returncode != 0 or "999.99" not in done.stdout:
        fail(f"{label} rc={done.returncode} {done.stderr[-500:]} {done.stdout[-300:]}")
        return None
    if f"decoder={decoder}" not in done.stderr or f"backend={backend}" not in done.stderr:
        fail(f"{label} decoder log\n{done.stderr}")
        return None
    if f"Total frames:{frames}" not in done.stdout:
        fail(f"{label} frame count\n{done.stdout}")
        return None
    print(f"OK   {label}")
    return done


def pu21_value(stdout: str) -> float | None:
    match = re.search(r"\[ PU21-PSNR:([0-9.]+) \]", stdout)
    if not match:
        return None
    return float(match.group(1))


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN, file=sys.stderr)
        return 2
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("SKIP no ffmpeg", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="pixvqa_av1_vvc_") as tmp:
        td = Path(tmp)
        src8 = td / "src8.yuv"
        src10 = td / "src10.yuv"
        write_raw(src8, "yuv420p")
        write_raw(src10, "yuv420p10le")

        av1_8 = td / "av1_8.mp4"
        if encode(ffmpeg, src8, av1_8, "yuv420p", ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-cpu-used", "8", "-lag-in-frames", "2"]):
            dec = td / "av1_8.yuv"
            if not decode_ref(ffmpeg, av1_8, dec, "yuv420p", ["-c:v", "libdav1d"]):
                fail("independent libdav1d could not decode 8-bit AV1")
            else:
                expect_match("av1 8-bit mp4 libdav1d", BIN, dec, av1_8, "420p:8", "libdav1d", "software")
                mkv = td / "av1_8.mkv"
                remux = run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(av1_8), "-c", "copy", str(mkv)])
                if remux.returncode != 0:
                    fail("could not remux AV1 MP4 to MKV: " + (remux.stderr or "")[-200:])
                else:
                    both = pix(BIN, [
                        "--metrics", "psnr", "--decode", "cpu", "-v", "0",
                        "-r", str(av1_8), "-s", f"{W}x{H}", "-f", "420p:8",
                        "-t", str(mkv), "-s", f"{W}x{H}", "-f", "420p:8",
                    ])
                    if both.returncode != 0 or "999.99" not in both.stdout:
                        fail(f"av1 mp4 vs mkv rc={both.returncode} {both.stderr[-400:]} {both.stdout[-200:]}")
                    else:
                        print("OK   av1 mp4 versus mkv")
                raw_side = pix(BIN, [
                    "--metrics", "psnr", "--decode", "cpu", "-v", "0",
                    "-r", str(src8), "-s", f"{W}x{H}", "-f", "420p:8",
                    "-t", str(av1_8), "-s", f"{W}x{H}", "-f", "420p:8",
                ])
                # Lossy AV1 is not required to match the source. The raw side must still be raw.
                if raw_side.returncode != 0 or "reference: raw" not in raw_side.stdout or "distorted: cpu" not in raw_side.stdout:
                    fail(f"raw versus av1 summary {raw_side.stderr[-300:]}")
                else:
                    print("OK   raw versus av1")
        else:
            fail("could not encode 8-bit AV1")

        av1_10 = td / "av1_10.mp4"
        if encode(ffmpeg, src10, av1_10, "yuv420p10le", [
            "-c:v", "libaom-av1", "-pix_fmt", "yuv420p10le", "-cpu-used", "8",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "tv",
            "-aom-params", "color-primaries=9:transfer-characteristics=16:matrix-coefficients=9",
        ]):
            dec = td / "av1_10.yuv"
            if not decode_ref(ffmpeg, av1_10, dec, "yuv420p10le", ["-c:v", "libdav1d"]):
                fail("independent libdav1d could not decode 10-bit AV1")
            else:
                hdr = expect_match("av1 10-bit mp4 libdav1d", BIN, dec, av1_10, "420p:10", "libdav1d", "software")
                if hdr and "transfer=pq" not in hdr.stdout:
                    fail(f"AV1 PQ transfer was not kept:\n{hdr.stdout}")
                other = td / "src10b.yuv"
                write_raw(other, "yuv420p10le", bias=70)
                compressed_score = pix(BIN, [
                    "--metrics", "pu21-psnr", "--decode", "cpu", "--hdr-transfer", "pq", "-v", "0",
                    "-r", str(av1_10), "-s", f"{W}x{H}", "-f", "420p:10",
                    "-t", str(other), "-s", f"{W}x{H}", "-f", "420p:10",
                ])
                raw_score = pix(BIN, [
                    "--metrics", "pu21-psnr", "--decode", "cpu", "--hdr-transfer", "pq", "-v", "0",
                    "-r", str(dec), "-s", f"{W}x{H}", "-f", "420p:10",
                    "-t", str(other), "-s", f"{W}x{H}", "-f", "420p:10",
                ])
                a = pu21_value(compressed_score.stdout)
                b = pu21_value(raw_score.stdout)
                if compressed_score.returncode != 0 or raw_score.returncode != 0 or a is None or b is None:
                    fail(f"pu21 av1 rc {compressed_score.returncode}/{raw_score.returncode}")
                elif abs(a - b) > 1e-4:
                    fail(f"pu21 av1 {a} versus raw {b}")
                else:
                    print(f"OK   pu21-psnr av1 matches raw decode ({a:.6f})")
        else:
            fail("could not encode 10-bit AV1")

        broken = td / "broken.av1"
        if av1_8.is_file():
            broken.write_bytes(av1_8.read_bytes()[:32])
            bad = pix(BIN, [
                "--metrics", "psnr", "--decode", "cpu", "--input", "compressed",
                "-r", str(broken), "-s", f"{W}x{H}", "-f", "420p:8",
                "-t", str(broken), "-s", f"{W}x{H}", "-f", "420p:8",
            ])
            if bad.returncode == 0 or "Summary results" in bad.stdout:
                fail("truncated AV1 was scored")
            else:
                print("OK   truncated AV1 is not scored")

        vvc = td / "clip.vvc.mp4"
        if encode(ffmpeg, src10, vvc, "yuv420p10le", ["-c:v", "libvvenc", "-preset", "faster", "-qp", "32", "-period", "1"]):
            dec = td / "vvc.yuv"
            if not decode_ref(ffmpeg, vvc, dec, "yuv420p10le", ["-c:v", "vvc"]):
                fail("independent FFmpeg vvc decoder could not decode the clip")
            else:
                got = expect_match("vvc 10-bit software", BIN, dec, vvc, "420p:10", "vvc", "software")
                if got is None:
                    pass
                vvdec = shutil.which("vvdecapp") or os.environ.get("VVDEC", "")
                if vvdec and Path(vvdec).is_file():
                    extracted = td / "clip.266"
                    ext = run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(vvc), "-c", "copy", "-f", "vvc", str(extracted)])
                    out_yuv = td / "vvdec.yuv"
                    app = run([vvdec, "-b", str(extracted), "-o", str(out_yuv)]) if ext.returncode == 0 else ext
                    if app.returncode != 0 or not out_yuv.is_file():
                        note("VVdeC did not decode this VVC clip: " + (app.stderr or "")[-200:])
                    else:
                        same = pix(BIN, [
                            "--metrics", "psnr", "--decode", "cpu", "-v", "0",
                            "-r", str(out_yuv), "-s", f"{W}x{H}", "-f", "420p:10",
                            "-t", str(vvc), "-s", f"{W}x{H}", "-f", "420p:10",
                        ])
                        if same.returncode != 0 or "999.99" not in same.stdout:
                            fail(f"vvc versus VVdeC {same.stderr[-300:]}")
                        else:
                            print("OK   vvc matches VVdeC")
                else:
                    note("VVdeC was not available; the VVC pixel oracle is a separate FFmpeg vvc process")
            refused = pix(BIN, [
                "--metrics", "psnr", "--decode", "gpu",
                "-r", str(vvc), "-s", f"{W}x{H}", "-f", "420p:10",
                "-t", str(vvc), "-s", f"{W}x{H}", "-f", "420p:10",
            ])
            if refused.returncode == 0 or "Summary results" in refused.stdout or "backend=software" in refused.stderr:
                fail(f"gpu vvc was not rejected: {refused.stderr[-300:]}")
            elif "not compiled" in refused.stderr or "vvc" in refused.stderr.lower():
                print("OK   gpu rejects VVC")
            else:
                fail(f"gpu vvc diagnostic: {refused.stderr[-300:]}")
        else:
            fail("could not encode VVC")

        if os.environ.get("PIXVQA_RUN_GPU") == "1" and FULL.is_file():
            if av1_8.is_file():
                dec = td / "av1_8.yuv"
                if dec.is_file():
                    expect_match("av1 8-bit nvdec", FULL, dec, av1_8, "420p:8", "av1_cuvid", "nvdec")
            if av1_10.is_file():
                dec = td / "av1_10.yuv"
                if dec.is_file():
                    expect_match("av1 10-bit nvdec", FULL, dec, av1_10, "420p:10", "av1_cuvid", "nvdec")
            hevc = td / "hevc10.mp4"
            if encode(ffmpeg, src10, hevc, "yuv420p10le", ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "lossless=1:bframes=0"]):
                dec = td / "hevc10.yuv"
                if decode_ref(ffmpeg, hevc, dec, "yuv420p10le"):
                    expect_match("hevc 10-bit nvdec", FULL, dec, hevc, "420p:10", "hevc", "nvdec")
            else:
                fail("could not encode HEVC for NVDEC")
            if vvc.is_file():
                refused = pix(FULL, [
                    "--metrics", "psnr", "--decode", "gpu",
                    "-r", str(vvc), "-s", f"{W}x{H}", "-f", "420p:10",
                    "-t", str(vvc), "-s", f"{W}x{H}", "-f", "420p:10",
                ])
                if refused.returncode == 0 or "Summary results" in refused.stdout or "backend=software" in refused.stderr:
                    fail(f"full binary gpu vvc fallback: {refused.stderr[-300:]}")
                else:
                    print("OK   full binary gpu rejects VVC")
        elif not FULL.is_file():
            note("build/full/pixvqa is absent; AV1 and HEVC NVDEC were not run")

    if NOTES:
        print("Notes:")
        for item in NOTES:
            print(" -", item)
    if FAILS:
        print(FAILS, "failed", file=sys.stderr)
        return 1
    print("All AV1 and VVC checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
