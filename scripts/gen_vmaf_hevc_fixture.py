#!/usr/bin/env python3
"""Generate a small SDR HEVC Main10 fixture + oracle raw YUV for Gate B.

Outputs under testdata/vmaf_gate_b/. The directory is gitignored.
Proves B-picture presence via ffprobe pict_type inspection (not encode flags alone).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("VMAF_GATE_B_DIR", ROOT / "testdata" / "vmaf_gate_b"))


def tool_path(explicit: str, sibling_of: str, name: str) -> str:
    if explicit:
        return explicit
    if sibling_of:
        sibling = Path(sibling_of).with_name(name)
        if sibling.is_file():
            return str(sibling)
    found = shutil.which(name)
    return found or name


_ffmpeg = os.environ.get("FFMPEG_BIN") or os.environ.get("FFMPEG") or ""
FFMPEG = tool_path(_ffmpeg, "", "ffmpeg")
FFPROBE = tool_path(os.environ.get("FFPROBE_BIN") or os.environ.get("FFPROBE") or "", FFMPEG, "ffprobe")
W = int(os.environ.get("VMAF_GATE_B_W", "640"))
H = int(os.environ.get("VMAF_GATE_B_H", "360"))
N = int(os.environ.get("VMAF_GATE_B_N", "16"))
FPS = 30


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_bytes() -> int:
    # 420p:10 planar: Y + U + V, 2 bytes/sample
    return W * H * 2 + 2 * ((W // 2) * (H // 2) * 2)


def write_yuv10(path: Path) -> None:
    """Temporal variation + mid-sequence scene cut; mild distortion twin written separately."""
    with path.open("wb") as f:
        for fi in range(N):
            scene = 1 if fi >= N // 2 else 0
            for row in range(H):
                for col in range(W):
                    base = (220 + scene * 350 + (row * 3 + col * 5 + fi * 19) % 450) & 0x3FF
                    f.write(struct.pack("<H", base))
            for seed in (512, 528):
                for row in range(H // 2):
                    for col in range(W // 2):
                        u = (seed + scene * 30 + (row + col + fi) % 40) & 0x3FF
                        f.write(struct.pack("<H", u))


def write_distorted(ref: Path, dist: Path) -> None:
    """Compression-like spatial noise on a copy of the reference sequence."""
    data = bytearray(ref.read_bytes())
    fb = frame_bytes()
    for fi in range(N):
        base = fi * fb
        for i in range(0, W * H * 2, 2):
            off = base + i
            v = data[off] | (data[off + 1] << 8)
            dlt = ((i // 2 + fi * 13) % 61) - 30
            nv = max(0, min(1023, v + dlt))
            data[off] = nv & 0xFF
            data[off + 1] = (nv >> 8) & 0xFF
    dist.write_bytes(data)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr + p.stdout)
        raise SystemExit(p.returncode)


def probe_hevc(hevc: Path) -> dict:
    """Inspect stream metadata and per-frame pict_type; require ≥1 B-picture."""
    if not Path(FFPROBE).is_file():
        raise SystemExit(f"missing ffprobe: {FFPROBE}")
    stream = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,nb_frames",
            "-of",
            "json",
            str(hevc),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(stream.stdout)
    st = meta["streams"][0]
    frames = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=pict_type,pts,coded_picture_number,display_picture_number",
            "-of",
            "json",
            str(hevc),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    flist = json.loads(frames.stdout).get("frames", [])
    types = [f.get("pict_type", "?") for f in flist]
    counts = dict(Counter(types))
    evidence = {
        "ffprobe": FFPROBE,
        "codec_name": st.get("codec_name"),
        "profile": st.get("profile"),
        "pix_fmt": st.get("pix_fmt"),
        "width": int(st.get("width", 0)),
        "height": int(st.get("height", 0)),
        "nb_frames_stream": int(st.get("nb_frames") or 0),
        "pict_types_in_display_order": types,
        "pict_type_counts": counts,
        "b_picture_count": int(counts.get("B", 0)),
        "has_b_pictures": int(counts.get("B", 0)) >= 1,
    }
    errors = []
    if evidence["codec_name"] != "hevc":
        errors.append(f"codec_name={evidence['codec_name']}")
    if evidence["profile"] != "Main 10":
        errors.append(f"profile={evidence['profile']}")
    if evidence["pix_fmt"] != "yuv420p10le":
        errors.append(f"pix_fmt={evidence['pix_fmt']}")
    if evidence["width"] != W or evidence["height"] != H:
        errors.append(f"size={evidence['width']}x{evidence['height']}")
    if evidence["nb_frames_stream"] != N and len(types) != N:
        errors.append(f"frames stream={evidence['nb_frames_stream']} listed={len(types)}")
    if len(types) != N:
        errors.append(f"ffprobe listed {len(types)} frames, expected {N}")
    if not evidence["has_b_pictures"]:
        errors.append("no B-pictures present despite bframes=3 encode request")
    if errors:
        raise SystemExit("HEVC fixture probe failed: " + "; ".join(errors))
    return evidence


def main() -> int:
    if not Path(FFMPEG).is_file():
        print(f"missing ffmpeg: {FFMPEG}", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    ref = OUT / "ref.yuv"
    dist_yuv_src = OUT / "dist_src.yuv"
    hevc = OUT / "dist.hevc.mp4"
    oracle = OUT / "dist_oracle.yuv"

    write_yuv10(ref)
    write_distorted(ref, dist_yuv_src)

    run(
        [
            FFMPEG,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p10le",
            "-s:v",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            str(dist_yuv_src),
            "-an",
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-profile:v",
            "main10",
            "-preset",
            "medium",
            "-x265-params",
            "bframes=3:keyint=8:min-keyint=4:pools=4",
            "-frames:v",
            str(N),
            str(hevc),
        ]
    )

    hevc_probe = probe_hevc(hevc)

    run(
        [
            FFMPEG,
            "-y",
            "-hwaccel",
            "none",
            "-c:v",
            "hevc",
            "-i",
            str(hevc),
            "-frames:v",
            str(N),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p10le",
            str(oracle),
        ]
    )

    expected_size = N * frame_bytes()
    if oracle.stat().st_size != expected_size:
        raise SystemExit(
            f"oracle size {oracle.stat().st_size} != expected {expected_size} "
            f"({N} frames × {frame_bytes()} bytes)"
        )

    manifest = {
        "width": W,
        "height": H,
        "frames": N,
        "fps": FPS,
        "format": "420p:10",
        "frame_bytes": frame_bytes(),
        "expected_yuv_bytes": expected_size,
        "ffmpeg": FFMPEG,
        "hevc_stream": hevc_probe,
        "files": {
            "ref.yuv": sha256_file(ref),
            "dist_src.yuv": sha256_file(dist_yuv_src),
            "dist.hevc.mp4": sha256_file(hevc),
            "dist_oracle.yuv": sha256_file(oracle),
        },
        "notes": (
            "dist_oracle.yuv is the software-decoded reconstruction of dist.hevc.mp4 "
            "(display order). Path A uses ref.yuv + dist_oracle.yuv. Path B/C use "
            "ref.yuv + dist.hevc.mp4. hevc_stream.pict_types_in_display_order is from ffprobe."
        ),
    }
    man_path = OUT / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
