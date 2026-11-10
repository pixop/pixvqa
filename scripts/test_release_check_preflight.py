#!/usr/bin/env python3
"""Offline unit tests for release_check.py preflight (no builds or GPU)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "release_check.py"
spec = importlib.util.spec_from_file_location("release_check", MODULE_PATH)
release_check = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(release_check)


class ReleaseCheckPreflightTests(unittest.TestCase):
    def _numpy_ok(self) -> None:
        return mock.patch.object(
            release_check.subprocess,
            "run",
            return_value=mock.Mock(returncode=0),
        )

    def test_preflight_sets_ffmpeg_from_path(self) -> None:
        env: dict[str, str] = {}
        with self._numpy_ok(), mock.patch.object(
            release_check.shutil, "which", return_value="/usr/bin/ffmpeg"
        ):
            notes = release_check.preflight_environment(env, sys.executable)
        self.assertEqual(env["PIXVQA_TEST_FFMPEG"], "/usr/bin/ffmpeg")
        self.assertTrue(any("PIXVQA_TEST_FFMPEG" in note for note in notes))

    def test_preflight_prefers_existing_ffmpeg_env(self) -> None:
        env = {"PIXVQA_TEST_FFMPEG": "/custom/ffmpeg", "FFMPEG": "/other/ffmpeg"}
        with self._numpy_ok():
            notes = release_check.preflight_environment(env, sys.executable)
        self.assertEqual(env["PIXVQA_TEST_FFMPEG"], "/custom/ffmpeg")
        self.assertEqual(notes, [])

    def test_preflight_warns_when_numpy_missing(self) -> None:
        env: dict[str, str] = {}

        def _run(cmd: list[str], **_kwargs: object) -> mock.Mock:
            if cmd[-1] == "import numpy":
                raise release_check.subprocess.CalledProcessError(1, "python")
            return mock.Mock(returncode=0)

        with mock.patch.object(release_check.subprocess, "run", side_effect=_run), mock.patch.object(
            release_check.shutil, "which", return_value=None
        ):
            notes = release_check.preflight_environment(env, sys.executable)
        self.assertEqual(len(notes), 1)
        self.assertIn("cannot import numpy", notes[0])


if __name__ == "__main__":
    raise SystemExit(unittest.main())
