#!/usr/bin/env python3
"""Fixture cache compatibility, streaming equivalence, and memory bounds."""
from __future__ import annotations

import json
import sys
import tempfile
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.config import FRAMES  # noqa: E402
from numerical_conformance.fixtures import (  # noqa: E402
    CORE_KEYS,
    MANIFEST_VERSION,
    FamilyFixtures,
    _merge_manifest_files,
    _paths_exist,
)
from numerical_conformance.scene import all_scene_frames  # noqa: E402
from numerical_conformance.yuv_io import (  # noqa: E402
    files_identical,
    rgb_frames_to_yuv,
    stream_reference_yuv,
    write_yuv420p10,
)


def _dummy_fx(root: Path) -> FamilyFixtures:
    return FamilyFixtures(
        family="sdr",
        root=root,
        reference_yuv=root / "reference.yuv",
        distorted_yuv=root / "distorted.yuv",
        reference_lossless=root / "reference_lossless.mkv",
        distorted_lossy=root / "distorted_lossy.mp4",
        reference_yuv_60=root / "reference_60f.yuv",
        reference_lossless_60=root / "reference_lossless_60f.mkv",
        distorted_lossy_60=root / "distorted_lossy_60f.mp4",
    )


def test_legacy_manifest_merge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fx = _dummy_fx(root)
        legacy = {
            "verified": True,
            "files": {
                "reference_yuv": str(root / "reference.yuv"),
                "distorted_yuv": str(root / "distorted.yuv"),
                "reference_lossless": str(root / "reference_lossless.mkv"),
                "distorted_lossy": str(root / "distorted_lossy.mp4"),
            },
        }
        merged = _merge_manifest_files(legacy, fx)
        for k in CORE_KEYS:
            assert k in merged
        assert "reference_yuv_60" in merged
        assert not _paths_exist(merged, ("reference_yuv_60",))


def test_streaming_yuv_equivalence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for family in ("sdr", "pq", "hlg"):
            stream_path = root / f"{family}_stream.yuv"
            list_path = root / f"{family}_list.yuv"
            stream_reference_yuv(stream_path, family=family, frame_count=FRAMES)
            write_yuv420p10(list_path, rgb_frames_to_yuv(all_scene_frames(FRAMES), family))
            if not files_identical(stream_path, list_path):
                raise AssertionError(f"streaming != list encoder for {family}")


def test_fixtures_use_streaming_not_batch() -> None:
    import inspect

    import numerical_conformance.fixtures as mod

    src = inspect.getsource(mod)
    if "all_scene_frames" in src:
        raise AssertionError("fixtures must not batch all_scene_frames")
    if "stream_reference_yuv" not in src:
        raise AssertionError("fixtures must stream_reference_yuv")


def test_single_frame_memory_peak() -> None:
    """Sanity check: one frame stays far below multi-frame batch RSS (~7 GiB on 60f)."""
    from numerical_conformance.yuv_io import encode_frame_yuv420p10_planes

    tracemalloc.start()
    encode_frame_yuv420p10_planes(0, FRAMES, "pq")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # NumPy/tracemalloc over-count vs RSS; bound rejects 60-frame batch retention (~7 GiB RSS).
    if peak > 2_500_000_000:
        raise AssertionError(f"single-frame tracemalloc peak too high: {peak}")
    print(f"INFO single-frame UHD encode tracemalloc peak: {peak / 1e6:.1f} MB")


def test_manifest_version_constant() -> None:
    assert MANIFEST_VERSION >= 2


def main() -> int:
    test_legacy_manifest_merge()
    print("OK   legacy-manifest-merge")
    test_streaming_yuv_equivalence()
    print("OK   streaming-yuv-equivalence")
    test_fixtures_use_streaming_not_batch()
    print("OK   fixtures-streaming-only")
    test_single_frame_memory_peak()
    print("OK   single-frame-memory-peak")
    test_manifest_version_constant()
    print("OK   manifest-version")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
