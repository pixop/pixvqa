#!/usr/bin/env python3
"""Prepare deterministic 48-frame UHD fixtures for CUDA XPSNR regression tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import hashlib
import json
from pathlib import Path

CACHE = Path("/tmp/pixvqa_p4")
WIDTH = 3840
HEIGHT = 2160
FRAMES = 48
EXPECTED_BYTES = WIDTH * HEIGHT * 3 * FRAMES
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg")

FIXTURES = {
    "job0_ref_48.yuv": "",
    "job0_48.yuv": "noise=alls=4:allf=t+u",
    "job1_ref_48.yuv": "hue=h=35",
    "job1_48.yuv": "hue=h=35,noise=alls=8:allf=t+u",
}

MANIFEST = CACHE / "manifest.json"

def generator_signature() -> str:
    configuration = {
        "width": WIDTH,
        "height": HEIGHT,
        "frames": FRAMES,
        "pixel_format": "yuv420p10le",
        "source": "testsrc2",
        "rate": 60,
        "filters": FIXTURES,
    }

    payload = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_manifest() -> dict:
    if not MANIFEST.is_file():
        return {}

    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    if (
        data.get("version") != 1
        or data.get("width") != WIDTH
        or data.get("height") != HEIGHT
        or data.get("frames") != FRAMES
        or data.get("pixel_format") != "yuv420p10le"
        or data.get("generator_signature") != generator_signature()
        or not isinstance(data.get("files"), dict)
    ):
        return {}

    return data

def cache_valid(path: Path, manifest: dict) -> bool:
    if not path.is_file():
        return False

    record = manifest.get("files", {}).get(path.name)

    if not isinstance(record, dict):
        return False

    if (
        path.stat().st_size != EXPECTED_BYTES
        or record.get("size") != EXPECTED_BYTES
    ):
        return False

    expected_hash = record.get("sha256")

    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return False

    return sha256_file(path) == expected_hash

def main() -> int:
    if not FFMPEG:
        print("ERROR: ffmpeg not found", file=sys.stderr)
        return 1

    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()

    if not manifest:
        manifest = {
            "version": 1,
            "width": WIDTH,
            "height": HEIGHT,
            "frames": FRAMES,
            "generator_signature": generator_signature(),
            "pixel_format": "yuv420p10le",
            "files": {},
        }

    manifest_changed = False

    for filename, filters in FIXTURES.items():
        target = CACHE / filename

        if cache_valid(target, manifest):
            print(f"REUSE {filename}")
            continue

        temporary = target.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)

        command = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi",
            "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate=60",
        ]

        if filters:
            command += ["-vf", filters]

        command += [
            "-frames:v", str(FRAMES),
            "-pix_fmt", "yuv420p10le",
            "-f", "rawvideo",
            str(temporary),
        ]

        print(f"GENERATE {filename}", flush=True)
        result = subprocess.run(command, check=False)

        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            print(f"ERROR: generation failed for {filename}", file=sys.stderr)
            return 1

        actual_size = temporary.stat().st_size

        if actual_size != EXPECTED_BYTES:
            temporary.unlink(missing_ok=True)
            print(
                f"ERROR: {filename} size {actual_size}, expected {EXPECTED_BYTES}",
                file=sys.stderr,
            )
            return 1

        temporary.replace(target)

        manifest["files"][filename] = {
            "size": EXPECTED_BYTES,
            "sha256": sha256_file(target),
        }
        manifest_changed = True

        print(f"OK   {filename}")

    if manifest_changed:
        temporary_manifest = MANIFEST.with_suffix(".json.tmp")

        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        temporary_manifest.replace(MANIFEST)
        print("Manifest updated")

    print("All UHD XPSNR fixtures prepared")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
