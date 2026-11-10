#!/usr/bin/env python3
"""Unit tests for strict learned FFmpeg binary resolution."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pixvqa_video.ffmpeg_bins import (  # noqa: E402
    FFmpegBinError,
    PINNED_FFMPEG_COMMIT,
    STAMP_FILENAME,
    resolve_ffmpeg_bin,
    resolve_ffprobe_bin,
)


def _write_stub(bin_dir: Path, name: str) -> Path:
    path = bin_dir / name
    path.write_text(
        "#!/bin/sh\n"
        'printf "ffmpeg version %s\\n"\n' % "n8.1.3",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_resolve_from_venv_stamp() -> None:
    with tempfile.TemporaryDirectory() as root:
        env = Path(root)
        bin_dir = env / "bin"
        bin_dir.mkdir()
        _write_stub(bin_dir, "ffmpeg")
        _write_stub(bin_dir, "ffprobe")
        (env / STAMP_FILENAME).write_text(PINNED_FFMPEG_COMMIT + "\n", encoding="utf-8")
        os.environ["PIXVQA_LEARNED_ENV_DIR"] = str(env)
        os.environ.pop("PIXVQA_FFMPEG_BIN", None)
        os.environ.pop("PIXVQA_FFPROBE_BIN", None)
        ffmpeg = resolve_ffmpeg_bin()
        ffprobe = resolve_ffprobe_bin()
        assert ffmpeg == str((bin_dir / "ffmpeg").resolve())
        assert ffprobe == str((bin_dir / "ffprobe").resolve())
        print("PASS: resolve from stamped venv bin")


def test_resolve_via_symlinked_interpreter() -> None:
    """Regression: sys.executable may symlink outside the venv; bins stay in venv/bin."""
    with tempfile.TemporaryDirectory() as root:
        env = Path(root) / ".venv"
        bin_dir = env / "bin"
        bin_dir.mkdir(parents=True)
        external = Path(root) / "external" / "bin"
        external.mkdir(parents=True)
        external_py = external / "python3"
        external_py.write_text("#!/bin/sh\n", encoding="utf-8")
        external_py.chmod(external_py.stat().st_mode | stat.S_IXUSR)
        (bin_dir / "python").symlink_to(external_py)
        # Wrappers behave like real learned installers: shell script, self-contained version.
        for name in ("ffmpeg", "ffprobe"):
            wrapper = bin_dir / name
            wrapper.write_text(
                f"#!/bin/sh\nprintf 'ffmpeg version n8.1.3\\n'",
                encoding="utf-8",
            )
            wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        (env / STAMP_FILENAME).write_text(PINNED_FFMPEG_COMMIT + "\n", encoding="utf-8")

        os.environ.pop("PIXVQA_LEARNED_ENV_DIR", None)
        os.environ.pop("VIRTUAL_ENV", None)
        os.environ.pop("PIXVQA_LEARNED_PYTHON", None)
        os.environ.pop("PIXVQA_FFMPEG_BIN", None)
        os.environ.pop("PIXVQA_FFPROBE_BIN", None)

        old_executable = sys.executable
        try:
            sys.executable = str(bin_dir / "python")
            ffmpeg = resolve_ffmpeg_bin()
            ffprobe = resolve_ffprobe_bin()
        finally:
            sys.executable = old_executable

        assert ffmpeg == str((bin_dir / "ffmpeg").resolve())
        assert ffprobe == str((bin_dir / "ffprobe").resolve())
        assert Path(ffmpeg).parent == bin_dir.resolve()
        print("PASS: resolve via symlinked interpreter without env hints")


def test_fail_closed_without_stamp() -> None:
    with tempfile.TemporaryDirectory() as root:
        env = Path(root)
        bin_dir = env / "bin"
        bin_dir.mkdir()
        _write_stub(bin_dir, "ffmpeg")
        os.environ["PIXVQA_LEARNED_ENV_DIR"] = str(env)
        os.environ.pop("PIXVQA_FFMPEG_BIN", None)
        try:
            resolve_ffmpeg_bin()
        except FFmpegBinError as exc:
            assert "System PATH is intentionally not searched" in str(exc)
            print("PASS: fail closed without stamp")
            return
        raise AssertionError("expected FFmpegBinError")


def main() -> int:
    test_resolve_from_venv_stamp()
    test_resolve_via_symlinked_interpreter()
    test_fail_closed_without_stamp()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
