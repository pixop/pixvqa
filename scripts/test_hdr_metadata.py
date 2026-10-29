#!/usr/bin/env python3
"""HDR metadata: missing, conflicting, overridden, and documented defaults."""
from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "swdec" / "pixvqa"))
W = H = 64
N = 2
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


def write_yuv(path: Path, y0: int) -> None:
    with path.open("wb") as f:
        for i in range(N):
            y = bytearray()
            for row in range(H):
                for col in range(W):
                    v = (y0 + i * 3 + row + col) & 0x3FF
                    y += int(v).to_bytes(2, "little")
            chroma = bytearray()
            for _ in range((H // 2) * (W // 2) * 2):
                chroma += (512).to_bytes(2, "little")
            f.write(y)
            f.write(chroma)


def run_ff(ffmpeg: str, args: list[str]) -> None:
    p = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(p.stderr or p.stdout or "ffmpeg failed")


def encode(ffmpeg: str, src: Path, dst: Path, x265: str, color: list[str]) -> None:
    run_ff(
        ffmpeg,
        [
            "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}", "-r", "24",
            "-i", str(src), "-frames:v", str(N),
            "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-profile:v", "main10",
            "-x265-params", "lossless=1:bframes=0:" + x265,
            *color,
            "-f", "hevc", str(dst),
        ],
    )


def probe(ffmpeg: str, src: Path) -> dict[str, str]:
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    if not Path(ffprobe).is_file():
        found = shutil.which("ffprobe")
        if not found:
            raise SystemExit("ffprobe not found")
        ffprobe = found
    p = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=color_transfer,color_primaries,color_space,color_range",
            "-of", "default=nw=1", str(src),
        ],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise SystemExit(p.stderr or "ffprobe failed")
    out = {}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def pix(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(BIN), *args], capture_output=True, text=True)


def sides(ref: Path, dist: Path) -> list[str]:
    return [
        "-r", str(ref), "-s", f"{W}x{H}", "-f", "420p:10",
        "-t", str(dist), "-s", f"{W}x{H}", "-f", "420p:10",
        "--metrics", "psnr", "-n", str(N), "-v", "0",
    ]


def score_text(out: str) -> str:
    keep = []
    for line in out.splitlines():
        if "PSNR" in line and not line.startswith(" ") and "Execution" not in line and not line.startswith("HDR"):
            keep.append(line.strip())
    return "\n".join(keep)


def hdr_lines(out: str) -> str:
    lines = []
    take = False
    for line in out.splitlines():
        if line.startswith("HDR:"):
            take = True
        elif take and line.startswith("  "):
            lines.append(line.strip())
        elif take:
            break
    return "\n".join(lines)


def expect_reject(done: subprocess.CompletedProcess[str], needle: str, label: str) -> None:
    if done.returncode == 0:
        fail(f"{label}: expected rejection, rc=0")
        return
    text = done.stderr + done.stdout
    if needle not in text:
        fail(f"{label}: missing {needle!r} in {text[-500:]!r}")
        return
    if "Summary:" in done.stdout or "[ PSNR" in done.stdout:
        fail(f"{label}: printed a score")
        return
    print("OK  ", label)


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN, file=sys.stderr)
        return 2
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("SKIP no ffmpeg")
        return 2

    with tempfile.TemporaryDirectory(prefix="pixvqa_hdr_") as td:
        t = Path(td)
        ref = t / "ref.yuv"
        dist = t / "dist.yuv"
        write_yuv(ref, 200)
        write_yuv(dist, 240)

        plain = pix(sides(ref, dist))
        if plain.returncode != 0:
            fail(f"raw missing rc={plain.returncode} {plain.stderr[-300:]}")
        else:
            block = hdr_lines(plain.stdout)
            if block.count("transfer=unspecified (absent)") != 2:
                fail(f"raw missing block: {block}")
            elif "transfer=pq" in block or "bt2020" in block:
                fail(f"raw missing assumed HDR: {block}")
            elif "Summary:" not in plain.stdout:
                fail("raw missing did not score")
            else:
                print("OK   raw_missing_not_pq")

        overridden = pix(sides(ref, dist) + ["--tf", "st2084", "--cs", "bt2020l"])
        if overridden.returncode != 0:
            fail(f"raw override rc={overridden.returncode} {overridden.stderr[-400:]}")
        else:
            block = hdr_lines(overridden.stdout)
            if "transfer=pq (override)" not in block or block.count("primaries=bt2020 (override)") != 2:
                fail(f"raw override block: {block}")
            elif "peak=10000" not in block or "st2084 eotf" not in block:
                fail(f"raw override display: {block}")
            elif score_text(overridden.stdout) != score_text(plain.stdout):
                fail("override changed PSNR text")
            else:
                print("OK   raw_override_pq_scores_unchanged")

        incomplete = pix(sides(ref, dist) + ["--tf", "st2084"])
        if incomplete.returncode != 0:
            fail(f"raw pq defaults rc={incomplete.returncode} {incomplete.stderr[-400:]}")
        else:
            block = hdr_lines(incomplete.stdout)
            if "primaries assumed bt2020" not in block or "matrix assumed bt2020nc" not in block:
                fail(f"raw pq defaults missing assumption: {block}")
            elif "range assumed limited" not in block or "transfer=pq (override)" not in block:
                fail(f"raw pq defaults block: {block}")
            elif score_text(incomplete.stdout) != score_text(plain.stdout):
                fail("pq defaults changed PSNR")
            else:
                print("OK   raw_pq_defaults")

        no_peak = pix(sides(ref, dist) + ["--tf", "stdb67", "--cs", "bt2020l"])
        if no_peak.returncode != 0:
            fail(f"hlg default peak rc={no_peak.returncode} {no_peak.stderr[-400:]}")
        else:
            block = hdr_lines(no_peak.stdout)
            if "peak=1000" not in block or "(assumption)" not in block or "gamma=1.200000000" not in block:
                fail(f"hlg default peak block: {block}")
            else:
                print("OK   hlg_default_peak_1000")

        hlg = pix(sides(ref, dist) + ["--tf", "stdb67", "--cs", "bt2020l", "--hlg-peak", "1000"])
        if hlg.returncode != 0:
            fail(f"hlg rc={hlg.returncode} {hlg.stderr[-400:]}")
        else:
            block = hdr_lines(hlg.stdout)
            if "transfer=hlg (override)" not in block or "gamma=1.200000000" not in block or "peak=1000" not in block:
                fail(f"hlg display: {block}")
            elif "st2084" in block:
                fail("hlg used the PQ transform")
            else:
                print("OK   hlg_peak_1000")

        hlg400 = pix(sides(ref, dist) + ["--tf", "stdb67", "--cs", "bt709f", "--hlg-peak", "400"])
        if hlg400.returncode != 0:
            fail(f"hlg400 rc={hlg400.returncode} {hlg400.stderr[-300:]}")
        else:
            m = re.search(r"gamma=([0-9.]+)", hdr_lines(hlg400.stdout))
            expect = 1.2 + 0.42 * math.log10(0.4)
            if not m or abs(float(m.group(1)) - expect) > 1e-9:
                fail(f"hlg400 gamma {m.group(1) if m else None} != {expect}")
            elif "primaries=bt709" not in hdr_lines(hlg400.stdout) or "range=full" not in hdr_lines(hlg400.stdout):
                fail(f"hlg400 color: {hdr_lines(hlg400.stdout)}")
            else:
                print("OK   hlg_peak_400_bt709")

        expect_reject(pix(sides(ref, dist) + ["--tf", "st2084", "--cs", "bt2020l", "--hlg-peak", "1000"]),
                      "hlg-peak", "pq_rejects_hlg_peak")
        expect_reject(pix(sides(ref, dist) + ["--tf", "st2084", "--cs", "bt601l"]),
                      "not a supported", "bt601_pq_rejected")
        both = pix(sides(ref, dist) + ["--tf", "sdr", "--ref-tf", "st2084", "--cs", "bt2020l"])
        if both.returncode != 0 or "Summary:" not in both.stdout:
            fail(f"global_and_ref_tf rc={both.returncode} {both.stderr[-400:]}")
        else:
            block = hdr_lines(both.stdout)
            if "reference: transfer=pq (override)" not in block or "distorted: transfer=sdr (override)" not in block:
                fail(f"global_and_ref_tf block: {block}")
            elif "conventional:" not in block:
                fail("global_and_ref_tf omitted the sample-code notice")
            else:
                print("OK   global_and_ref_tf")
        ranged = pix(sides(ref, dist) + ["--tf", "st2084", "--cs", "bt2020l", "--ref-hdr-range", "full",
                                          "--metrics", "pu21-psnr"])
        if ranged.returncode != 0:
            fail(f"cs_plus_ref_range rc={ranged.returncode} {ranged.stderr[-400:]}")
        else:
            block = hdr_lines(ranged.stdout)
            if "reference: transfer=pq (override)" not in block or "reference:" not in block:
                fail(f"cs_plus_ref_range block: {block}")
            ref_line = next(line for line in block.splitlines() if line.startswith("reference:"))
            dist_line = next(line for line in block.splitlines() if line.startswith("distorted:"))
            if "range=full (override)" not in ref_line or "range=limited (override)" not in dist_line:
                fail(f"cs_plus_ref_range lines:\n{ref_line}\n{dist_line}")
            elif "PU21-PSNR" not in ranged.stdout:
                fail("cs_plus_ref_range produced no PU21 score")
            else:
                print("OK   cs_plus_ref_range")
        peaked = pix(sides(ref, dist) + ["--tf", "stdb67", "--cs", "bt2020l", "--hlg-peak", "1000",
                                          "--ref-hdr-peak-nits", "400", "--metrics", "pu21-psnr"])
        if peaked.returncode != 0:
            fail(f"shared_peak_plus_ref_peak rc={peaked.returncode} {peaked.stderr[-400:]}")
        else:
            block = hdr_lines(peaked.stdout)
            ref_disp = next((line for line in block.splitlines() if line.startswith("display reference:")), "")
            dist_disp = next((line for line in block.splitlines() if line.startswith("display distorted:")), "")
            if "peak=400" not in ref_disp or "peak=1000" not in dist_disp:
                fail(f"shared_peak_plus_ref_peak:\n{ref_disp}\n{dist_disp}")
            elif "PU21-PSNR" not in peaked.stdout:
                fail("shared_peak_plus_ref_peak produced no PU21 score")
            else:
                print("OK   shared_peak_plus_ref_peak")
        expect_reject(pix(sides(ref, dist) + ["--tf", "st2084", "--hdr-transfer", "pq", "--cs", "bt2020l"]),
                      "--tf and --hdr-transfer both set", "two_shared_transfers")
        expect_reject(pix(sides(ref, dist) + ["--tf", "stdb67", "--cs", "bt2020l", "--hlg-peak", "1000",
                                               "--hdr-peak-nits", "1000"]),
                      "--hlg-peak and --hdr-peak-nits both set", "two_shared_peaks")

        per_side = pix(sides(ref, dist) + ["--ref-tf", "st2084", "--ref-cs", "bt2020l",
                                            "--dist-tf", "stdb67", "--dist-cs", "bt2020l", "--hlg-peak", "1000"])
        if per_side.returncode != 0:
            fail(f"per-side rc={per_side.returncode} {per_side.stderr[-400:]}")
        else:
            block = hdr_lines(per_side.stdout)
            if "reference: transfer=pq (override)" not in block or "distorted: transfer=hlg (override)" not in block:
                fail(f"per-side block: {block}")
            elif "Summary:" not in per_side.stdout:
                fail("per-side did not score")
            else:
                print("OK   per_side_pq_and_hlg")

        ref_h = t / "ref_none.hevc"
        dist_h = t / "dist_none.hevc"
        encode(ffmpeg, ref, ref_h, "colorprim=undef:transfer=undef:colormatrix=undef",
               ["-color_primaries", "unknown", "-color_trc", "unknown", "-colorspace", "unknown"])
        encode(ffmpeg, dist, dist_h, "colorprim=undef:transfer=undef:colormatrix=undef",
               ["-color_primaries", "unknown", "-color_trc", "unknown", "-colorspace", "unknown"])
        none = pix(sides(ref_h, dist_h))
        probed = probe(ffmpeg, ref_h)
        if none.returncode != 0:
            fail(f"hevc missing rc={none.returncode} {none.stderr[-400:]} probe={probed}")
        else:
            block = hdr_lines(none.stdout)
            if "transfer=pq" in block or "transfer=hlg" in block:
                fail(f"hevc missing assumed HDR: {block} probe={probed}")
            elif "transfer=unspecified (stream)" not in block and "transfer=unspecified (absent)" not in block:
                fail(f"hevc missing block: {block} probe={probed}")
            elif score_text(none.stdout) != score_text(plain.stdout):
                fail("hevc missing changed PSNR versus raw")
            else:
                print("OK   hevc_missing_not_pq", probed.get("color_transfer"))

        pq_h = t / "pq.hevc"
        hlg_h = t / "hlg.hevc"
        pq709 = t / "pq709.hevc"
        pq_full = t / "pqfull.hevc"
        encode(ffmpeg, ref, pq_h, "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
               ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "tv"])
        encode(ffmpeg, dist, hlg_h, "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited",
               ["-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-color_range", "tv"])
        encode(ffmpeg, dist, pq709, "colorprim=bt709:transfer=smpte2084:colormatrix=bt709:range=limited",
               ["-color_primaries", "bt709", "-color_trc", "smpte2084", "-colorspace", "bt709", "-color_range", "tv"])
        encode(ffmpeg, dist, pq_full, "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=full",
               ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "pc"])

        pq_probe = probe(ffmpeg, pq_h)
        if pq_probe.get("color_transfer") not in ("smpte2084", "smpte2084"):
            fail(f"encoder did not tag PQ: {pq_probe}")
        else:
            streamed = pix(sides(pq_h, pq_h))
            block = hdr_lines(streamed.stdout)
            if streamed.returncode != 0:
                fail(f"stream pq rc={streamed.returncode} {streamed.stderr[-400:]}")
            elif "transfer=pq (stream)" not in block or "primaries=bt2020 (stream)" not in block:
                fail(f"stream pq block: {block} probe={pq_probe}")
            elif "range=limited (stream)" not in block or "matrix=bt2020nc (stream)" not in block:
                fail(f"stream pq matrix/range: {block}")
            elif "peak=10000" not in block:
                fail(f"stream pq display: {block}")
            elif score_text(streamed.stdout) != score_text(pix(sides(ref, ref)).stdout):
                fail("stream metadata changed a same-file PSNR")
            else:
                print("OK   hevc_pq_from_stream")

        conflict = pix(sides(pq_h, hlg_h))
        if conflict.returncode != 0:
            fail(f"independent transfers rc={conflict.returncode} {conflict.stderr[-400:]}")
        else:
            block = hdr_lines(conflict.stdout)
            if "reference: transfer=pq (stream)" not in block or "distorted: transfer=hlg (stream)" not in block:
                fail(f"independent transfers block: {block}")
            elif score_text(conflict.stdout) != score_text(plain.stdout):
                fail("independent transfers changed PSNR")
            else:
                print("OK   hevc_pq_and_hlg_independent")

        forced = pix(sides(pq_h, hlg_h) + ["--tf", "st2084", "--cs", "bt2020l"])
        if forced.returncode != 0:
            fail(f"forced rc={forced.returncode} {forced.stderr[-400:]}")
        else:
            block = hdr_lines(forced.stdout)
            if block.count("transfer=pq (override)") != 2 or "peak=10000" not in block:
                fail(f"forced block: {block}")
            elif score_text(forced.stdout) != score_text(plain.stdout):
                fail("forced override changed PSNR versus raw of the same pixels")
            else:
                print("OK   override_resolves_pq_hlg_conflict")

        prim = pix(sides(pq_h, pq709))
        if prim.returncode != 0:
            fail(f"independent primaries rc={prim.returncode} {prim.stderr[-400:]}")
        else:
            block = hdr_lines(prim.stdout)
            if "primaries=bt2020 (stream)" not in block or "primaries=bt709 (stream)" not in block:
                fail(f"independent primaries block: {block}")
            else:
                print("OK   hevc_independent_primaries")
        ranges = pix(sides(pq_h, pq_full))
        if ranges.returncode != 0:
            fail(f"independent range rc={ranges.returncode} {ranges.stderr[-400:]}")
        else:
            block = hdr_lines(ranges.stdout)
            if "range=limited (stream)" not in block or "range=full (stream)" not in block:
                fail(f"independent range block: {block}")
            else:
                print("OK   hevc_independent_range")

        one_side = pix(sides(pq_h, hlg_h) + ["--dist-tf", "st2084", "--dist-cs", "bt2020l"])
        if one_side.returncode != 0:
            fail(f"one-side override rc={one_side.returncode} {one_side.stderr[-400:]}")
        else:
            block = hdr_lines(one_side.stdout)
            if "reference: transfer=pq (stream)" not in block or "distorted: transfer=pq (override)" not in block:
                fail(f"one-side block: {block}")
            else:
                print("OK   one_side_override_keeps_the_other_stream")

        # VMAF's model rejects luma below 320x240 before a picture is read.
        vw, vh = 320, 240
        vref = t / "vref.yuv"
        def write_big(path: Path, y0: int) -> None:
            with path.open("wb") as f:
                for i in range(N):
                    y = bytearray()
                    for row in range(vh):
                        for col in range(vw):
                            y += ((y0 + i + (row + col) % 64) & 0x3FF).to_bytes(2, "little")
                    chroma = bytearray()
                    for _ in range((vh // 2) * (vw // 2) * 2):
                        chroma += (512).to_bytes(2, "little")
                    f.write(y)
                    f.write(chroma)
        write_big(vref, 180)
        vpq = t / "vpq.hevc"
        run_ff(ffmpeg, [
            "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{vw}x{vh}", "-r", "24",
            "-i", str(vref), "-frames:v", str(N),
            "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-profile:v", "main10",
            "-x265-params", "lossless=1:bframes=0:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "tv",
            "-f", "hevc", str(vpq),
        ])
        vmaf = pix([
            "-r", str(vpq), "-s", f"{vw}x{vh}", "-f", "420p:10",
            "-t", str(vpq), "-s", f"{vw}x{vh}", "-f", "420p:10",
            "--metrics", "psnr,vmaf", "--model", "model/vmaf_v1.0.16_3d0h.json", "-n", "1", "-v", "0",
        ])
        if "no libvmaf" in vmaf.stderr:
            if vmaf.returncode == 0 or "Summary:" in vmaf.stdout or "[ PSNR" in vmaf.stdout:
                fail("vmaf_unavailable_still_scored")
            else:
                print("OK   vmaf_unavailable_no_score")
        else:
            expect_reject(vmaf, "vmaf rejects HDR transfer pq", "vmaf_rejects_stream_pq")
            if "validated HDR" in vmaf.stderr or "HDR VMAF" in vmaf.stdout:
                fail("ordinary vmaf claimed an HDR model")
            vmaf_sdr = pix([
                "-r", str(vpq), "-s", f"{vw}x{vh}", "-f", "420p:10",
                "-t", str(vpq), "-s", f"{vw}x{vh}", "-f", "420p:10",
                "--tf", "sdr", "--metrics", "vmaf", "--model", "model/vmaf_v1.0.16_3d0h.json",
                "-n", "1", "-v", "0",
            ])
            if vmaf_sdr.returncode != 0 or "Summary:" not in vmaf_sdr.stdout:
                fail(f"vmaf_explicit_sdr rc={vmaf_sdr.returncode} {vmaf_sdr.stderr[-400:]}")
            elif "transfer=sdr (override)" not in vmaf_sdr.stdout:
                fail("vmaf_explicit_sdr did not record the override")
            elif "VMAF" not in vmaf_sdr.stdout:
                fail("vmaf_explicit_sdr produced no VMAF score")
            else:
                print("OK   vmaf_explicit_sdr_on_pq_stream")

    if FAILS:
        print(FAILS, "failed", file=sys.stderr)
        return 1
    print("All HDR metadata checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
