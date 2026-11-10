#!/usr/bin/env python3
"""Focused tests for learned readiness FFmpeg stamp validation."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pixvqa_learned_install.validate import (  # noqa: E402
    probe_ffmpeg_bins,
    validate_readiness,
)
from pixvqa_video.ffmpeg_bins import (  # noqa: E402
    FFmpegBinError,
    PINNED_FFMPEG_COMMIT,
    STAMP_FILENAME,
    resolve_ffprobe_legacy,
)


def _write_stub(bin_dir: Path, name: str) -> Path:
    path = bin_dir / name
    path.write_text(
        "#!/bin/sh\nprintf 'ffmpeg version n8.1.3\\n'",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_probe_passes_with_stamp() -> None:
    with tempfile.TemporaryDirectory() as root:
        env = Path(root)
        bin_dir = env / "bin"
        bin_dir.mkdir()
        py = bin_dir / "python"
        py.write_text("#!/bin/sh\n", encoding="utf-8")
        py.chmod(py.stat().st_mode | stat.S_IXUSR)
        _write_stub(bin_dir, "ffmpeg")
        _write_stub(bin_dir, "ffprobe")
        (env / STAMP_FILENAME).write_text(PINNED_FFMPEG_COMMIT + "\n", encoding="utf-8")
        assert probe_ffmpeg_bins(str(py)) is None
    print("PASS: probe_ffmpeg_bins accepts stamped venv bins")


def test_probe_fails_without_stamp() -> None:
    with tempfile.TemporaryDirectory() as root:
        env = Path(root)
        bin_dir = env / "bin"
        bin_dir.mkdir()
        py = bin_dir / "python"
        py.write_text("#!/bin/sh\n", encoding="utf-8")
        py.chmod(py.stat().st_mode | stat.S_IXUSR)
        _write_stub(bin_dir, "ffmpeg")
        _write_stub(bin_dir, "ffprobe")
        err = probe_ffmpeg_bins(str(py))
        assert err and "System PATH is intentionally not searched" in err
    print("PASS: probe_ffmpeg_bins fails closed without stamp")


def test_validate_skips_ffmpeg_when_disabled() -> None:
    ready, issues = validate_readiness(
        ROOT,
        learned_python="/nonexistent/python",
        feature_python="",
        inference_python="",
        skip_load=True,
        check_ffmpeg=False,
    )
    assert not any("ffmpeg" in i.lower() or "ffprobe" in i.lower() for i in issues)
    print("PASS: check_ffmpeg=False skips ffmpeg readiness")


def test_legacy_ffprobe_override_requires_pinned_version() -> None:
    with tempfile.TemporaryDirectory() as root:
        bad = Path(root) / "ffprobe"
        bad.write_text("#!/bin/sh\nprintf 'ffmpeg version 4.0\\n'", encoding="utf-8")
        bad.chmod(bad.stat().st_mode | stat.S_IXUSR)
        os.environ["PIXVQA_CVVDP_FFPROBE"] = str(bad)
        try:
            resolve_ffprobe_legacy("PIXVQA_CVVDP_FFPROBE")
            raise AssertionError("expected FFmpegBinError for wrong ffprobe version")
        except FFmpegBinError as exc:
            assert "n8.1.3" in str(exc)
        finally:
            os.environ.pop("PIXVQA_CVVDP_FFPROBE", None)
    print("PASS: legacy ffprobe override validates pinned FFmpeg version")


def main() -> int:
    test_probe_passes_with_stamp()
    test_probe_fails_without_stamp()
    test_validate_skips_ffmpeg_when_disabled()
    test_legacy_ffprobe_override_requires_pinned_version()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
