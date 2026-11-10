#!/usr/bin/env python3
"""Unit tests for learned-env pin verification (offline, no real torch install)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pixvqa_learned_install import env_pins  # noqa: E402


class LearnedEnvPinsTests(unittest.TestCase):
    def test_parse_lock_pins_ignores_comments(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("# comment\n\ntorch==2.13.0+cu130\nnumpy==2.5.2\n")
            path = Path(handle.name)
        try:
            pins = env_pins.parse_lock_pins(path)
            self.assertEqual(pins["torch"], "2.13.0+cu130")
            self.assertEqual(pins["numpy"], "2.5.2")
            self.assertNotIn("comment", pins)
        finally:
            path.unlink(missing_ok=True)

    def test_verify_pins_success(self) -> None:
        pins = {"torch": "2.13.0+cu130", "numpy": "2.5.2", "pillow": "13.0.0"}
        with mock.patch.object(env_pins.metadata, "version", side_effect=lambda d: {
            "torch": "2.13.0+cu130",
            "numpy": "2.5.2",
            "pillow": "13.0.0",
        }[d]):
            self.assertEqual(env_pins.verify_pins(pins), [])

    def test_verify_pins_mismatch_torch(self) -> None:
        pins = {"torch": "2.13.0+cu130", "torchvision": "0.28.0+cu130"}
        with mock.patch.object(
            env_pins.metadata,
            "version",
            side_effect=lambda d: "2.11.0+cu128" if d == "torch" else "0.28.0+cu130",
        ):
            errors = env_pins.verify_pins(pins)
        self.assertEqual(len(errors), 1)
        self.assertIn("torch", errors[0])
        self.assertIn("2.11.0+cu128", errors[0])

    def test_verify_pins_missing_distribution(self) -> None:
        pins = {"pytorchvideo": "0.1.5"}

        def _raise(dist: str) -> str:
            raise env_pins.metadata.PackageNotFoundError

        with mock.patch.object(env_pins.metadata, "version", side_effect=_raise):
            errors = env_pins.verify_pins(pins)
        self.assertEqual(len(errors), 1)
        self.assertIn("pytorchvideo", errors[0])

    def test_installed_version_scans_distributions(self) -> None:
        dist = mock.Mock()
        dist.metadata = {"Name": "Pillow"}
        dist.version = "13.0.0"
        with mock.patch.object(env_pins.metadata, "version", side_effect=env_pins.metadata.PackageNotFoundError):
            with mock.patch.object(env_pins.metadata, "distributions", return_value=[dist]):
                self.assertEqual(env_pins.installed_version("pillow"), "13.0.0")

    def test_verify_pins_skips_bundled_cuda_stack_when_torch_differs(self) -> None:
        pins = {
            "torch": "2.13.0+cu130",
            "torchvision": "0.28.0+cu130",
            "nvidia-cublas": "13.1.1.3",
            "numpy": "2.5.2",
        }

        def _version(dist: str) -> str:
            return {
                "torch": "2.11.0+cu128",
                "torchvision": "0.26.0+cu128",
                "numpy": "2.5.2",
            }[dist]

        with mock.patch.object(env_pins, "installed_version", side_effect=_version):
            errors = env_pins.verify_pins(pins)
        reported = {entry.split(":")[0] for entry in errors}
        self.assertIn("torch", reported)
        self.assertIn("torchvision", reported)
        self.assertNotIn("nvidia-cublas", reported)

    def test_verify_pins_requires_cuda_stack_when_torch_matches(self) -> None:
        pins = {
            "torch": "2.13.0+cu130",
            "nvidia-cublas": "13.1.1.3",
        }

        def _version(dist: str) -> str | None:
            if dist == "torch":
                return "2.13.0+cu130"
            return None

        with mock.patch.object(env_pins, "installed_version", side_effect=_version):
            errors = env_pins.verify_pins(pins)
        self.assertEqual(len(errors), 1)
        self.assertIn("nvidia-cublas", errors[0])

    def test_main_rejects_wrong_python(self) -> None:
        with mock.patch.object(env_pins.sys, "version_info", (3, 11, 0, "final", 0)):
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
                handle.write("numpy==2.5.2\n")
                path = Path(handle.name)
            try:
                rc = env_pins.main([str(path)])
            finally:
                path.unlink(missing_ok=True)
        self.assertEqual(rc, 2)

    def test_main_accepts_matching_pins(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("numpy==2.5.2\npillow==13.0.0\n")
            path = Path(handle.name)
        try:
            with mock.patch.object(env_pins.metadata, "version", side_effect=lambda d: {
                "numpy": "2.5.2",
                "pillow": "13.0.0",
            }[d]):
                with mock.patch.object(env_pins.sys, "version_info", (3, 12, 0, "final", 0)):
                    rc = env_pins.main([str(path)])
            self.assertEqual(rc, 0)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
