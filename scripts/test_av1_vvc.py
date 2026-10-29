#!/usr/bin/env python3
"""AV1 libdav1d, AV1 NVDEC, VVC software, and a fresh HEVC NVDEC check.

Film grain stays at libdav1d's default (apply it only when the bitstream
carries it). These encodes do not write a film-grain header. The operating
point stays the single default point. Pixel equality is PSNR 999.99 against
an independent decode of the same bitstream.
"""
from __future__ import annotations

import hashlib
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
VVC_W, VVC_H = 480, 360
VVC_SHA256 = "eb1191c46b50cb8b2b631194f74a7a9152cced29e0f7d1278d7723420e83d44d"
VVC_FIXTURE = ROOT / "tests" / "fixtures" / "vvc" / "bbb_360p.vvc"
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
    if done.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 0:
        print(done.stderr[-500:], file=sys.stderr)
        return False
    return True


def decode_ref(ffmpeg: str, src: Path, dst: Path, pix: str, decoder: list[str] | None = None) -> bool:
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *(decoder or []),
           "-i", str(src), "-f", "rawvideo", "-pix_fmt", pix, str(dst)]
    done = run(cmd)
    return done.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def pix(binary: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return run([str(binary), *args])


def expect_match(label: str, binary: Path, reference: Path, compressed: Path, fmt: str,
                 decoder: str, backend: str, frames: int = N, width: int = W, height: int = H) -> subprocess.CompletedProcess[str] | None:
    done = pix(binary, [
        "--metrics", "psnr", "--decode", "cpu" if backend == "software" else "gpu",
        "--metric-device", "cpu", "-v", "0",
        "-r", str(reference), "-s", f"{width}x{height}", "-f", fmt,
        "-t", str(compressed), "--input", "compressed", "-s", f"{width}x{height}", "-f", fmt,
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


def vvc_reference_decoder(fallback: str) -> list[str]:
    """Find an FFmpeg CLI with native VVC decode; encoding is not required."""
    candidates = [
        os.environ.get("PIXVQA_VVC_FFMPEG", ""),
        os.environ.get("FFMPEG", ""),
        fallback,
        str(Path.home() / "codeccomp/encoders/nvenc/bin/ffmpeg"),
        str(Path.home() / "codeccomp/metrics/vmaf1.0/bin/ffmpeg"),
    ]
    valid: list[str] = []
    for candidate in dict.fromkeys(candidates):
        if not candidate or not Path(candidate).is_file():
            continue
        result = run([candidate, "-hide_banner", "-decoders"])
        if result.returncode == 0 and re.search(r"^\s*V[^\n]*\bvvc\b", result.stdout, re.M):
            valid.append(candidate)
    return valid


def yuv10_frames(reference: Path, width: int, height: int) -> int:
    # yuv420p10le: 2 bytes per sample, 1.5 samples per pixel.
    frame_bytes = width * height * 3
    if width <= 0 or height <= 0 or reference.stat().st_size == 0 or reference.stat().st_size % frame_bytes:
        return 0
    return reference.stat().st_size // frame_bytes


def fixture_frames(reference: Path) -> int:
    return yuv10_frames(reference, VVC_W, VVC_H)


def selected_vvc() -> tuple[Path, int, int, bool]:
    """Default fixture is the unredistributed BBB stream. A synthetic override is explicit."""
    override = os.environ.get("PIXVQA_VVC_FIXTURE", "").strip()
    if not override:
        return VVC_FIXTURE, VVC_W, VVC_H, False
    size = os.environ.get("PIXVQA_VVC_SIZE", "").lower().strip()
    if "x" not in size:
        fail("PIXVQA_VVC_FIXTURE requires PIXVQA_VVC_SIZE=WxH")
        return Path(override), 0, 0, True
    width_s, height_s = size.split("x", 1)
    try:
        width, height = int(width_s), int(height_s)
    except ValueError:
        fail(f"PIXVQA_VVC_SIZE is not WxH: {size}")
        return Path(override), 0, 0, True
    return Path(override), width, height, True


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN, file=sys.stderr)
        return 2
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("FAIL no ffmpeg", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="pixvqa_av1_vvc_") as tmp:
        td = Path(tmp)
        src8 = td / "src8.yuv"
        src10 = td / "src10.yuv"
        write_raw(src8, "yuv420p")
        write_raw(src10, "yuv420p10le")
        codec = {"av1_8": "FAIL", "av1_10": "FAIL", "vvc": "FAIL"}

        section = FAILS
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
        codec["av1_8"] = "PASS" if FAILS == section else "FAIL"

        section = FAILS
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
        codec["av1_10"] = "PASS" if FAILS == section else "FAIL"

        truncated_before = FAILS
        broken = td / "broken.av1"
        if av1_8.is_file():
            broken.write_bytes(av1_8.read_bytes()[:32])
            bad = pix(BIN, [
                "--metrics", "psnr", "--decode", "cpu", "--input", "compressed",
                "-r", str(broken), "-s", f"{W}x{H}", "-f", "420p:8",
                "-t", str(broken), "-s", f"{W}x{H}", "-f", "420p:8",
            ])
            if bad.returncode == 0 or "Summary:" in bad.stdout:
                fail("truncated AV1 was scored")
            else:
                print("OK   truncated AV1 is not scored")
        if FAILS != truncated_before:
            codec["av1_8"] = "FAIL"

        section = FAILS
        vvc, vvc_w, vvc_h, synthetic = selected_vvc()
        vvc_trusted = False
        reference = os.environ.get("PIXVQA_VVC_REFERENCE", "").strip()
        if synthetic and (vvc_w <= 0 or vvc_h <= 0):
            pass
        elif not vvc.is_file():
            if synthetic:
                fail(f"synthetic VVC fixture missing: {vvc}")
            else:
                fail(
                    f"VVC fixture missing: {vvc}. "
                    "bbb_360p.vvc is not in the repository because redistribution is unverified. "
                    "This is a failure, not a skip. "
                    "A CI job can set PIXVQA_VVC_FIXTURE, PIXVQA_VVC_SIZE, and PIXVQA_VVC_REFERENCE."
                )
        elif not synthetic and hashlib.sha256(vvc.read_bytes()).hexdigest() != VVC_SHA256:
            fail("VVC fixture SHA-256 mismatch; refusing an unverified bitstream")
        else:
            vvc_trusted = True
            oracle = td / "vvc-reference.yuv"
            if synthetic:
                if not reference or not Path(reference).is_file():
                    fail("synthetic VVC requires PIXVQA_VVC_REFERENCE from an independent vvdec decode")
                else:
                    oracle.write_bytes(Path(reference).read_bytes())
                    frames = yuv10_frames(oracle, vvc_w, vvc_h)
                    if frames <= 0:
                        fail(f"synthetic VVC reference is not yuv420p10le {vvc_w}x{vvc_h}")
                    else:
                        note(f"VVC independent pixel oracle: vvdec reference {reference}; {frames} frames")
                        expect_match("vvc 10-bit software", BIN, oracle, vvc, "420p:10",
                                     "vvc", "software", frames=frames, width=vvc_w, height=vvc_h)
            else:
                oracle_ffmpeg = None
                oracle_diagnostics = []
                for candidate in vvc_reference_decoder(ffmpeg):
                    attempt = run([
                        candidate, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "vvc", "-i", str(vvc), "-f", "rawvideo",
                        "-pix_fmt", "yuv420p10le", str(oracle),
                    ])
                    if attempt.returncode == 0 and oracle.is_file() and fixture_frames(oracle) > 0:
                        oracle_ffmpeg = candidate
                        break
                    oracle_diagnostics.append(f"{candidate}: {attempt.stderr[-350:]}")
                if oracle_ffmpeg is None:
                    fail("independent FFmpeg VVC decode unavailable. Set PIXVQA_VVC_FFMPEG "
                         "to a modern FFmpeg with native VVC decoding. " + " | ".join(oracle_diagnostics))
                else:
                    frames = fixture_frames(oracle)
                    note(f"VVC independent pixel oracle: {oracle_ffmpeg}; {frames} frames")
                    expect_match("vvc 10-bit software", BIN, oracle, vvc, "420p:10",
                                 "vvc", "software", frames=frames, width=VVC_W, height=VVC_H)
            refused = pix(BIN, [
                "--metrics", "psnr", "--decode", "gpu", "--input", "compressed",
                "-r", str(vvc), "-s", f"{vvc_w}x{vvc_h}", "-f", "420p:10",
                "-t", str(vvc), "-s", f"{vvc_w}x{vvc_h}", "-f", "420p:10",
            ])
            if refused.returncode == 0 or "Summary:" in refused.stdout or "backend=software" in refused.stderr:
                fail(f"gpu vvc was not rejected: {refused.stderr[-300:]}")
            elif "not compiled" in refused.stderr or "vvc" in refused.stderr.lower():
                print("OK   gpu rejects VVC")
            else:
                fail(f"gpu vvc diagnostic: {refused.stderr[-300:]}")
        codec["vvc"] = "PASS" if FAILS == section else "FAIL"

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
            if vvc_trusted:
                refused = pix(FULL, [
                    "--metrics", "psnr", "--decode", "gpu",
                    "-r", str(vvc), "--input", "compressed", "-s", f"{vvc_w}x{vvc_h}", "-f", "420p:10",
                    "-t", str(vvc), "--input", "compressed", "-s", f"{vvc_w}x{vvc_h}", "-f", "420p:10",
                ])
                if refused.returncode == 0 or "Summary:" in refused.stdout or "backend=software" in refused.stderr:
                    fail(f"full binary gpu vvc fallback: {refused.stderr[-300:]}")
                else:
                    print("OK   full binary gpu rejects VVC")
        elif not FULL.is_file():
            note("build/full/pixvqa is absent; AV1 and HEVC NVDEC were not run")

        line = "CODEC_STATUS " + " ".join(f"{name}={state}" for name, state in codec.items())
        print(line)
        report = ROOT / "reports" / "qa" / "codec-status.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(line + "\n", encoding="utf-8")

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
