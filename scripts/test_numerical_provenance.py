#!/usr/bin/env python3
"""Regression tests for numerical baseline provenance collection."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.baselines import validate_provenance  # noqa: E402
from numerical_conformance.provenance import (  # noqa: E402
    _python_package_version,
    collect_learned_provenance,
    cvvdp_has_package_identity,
)


def test_cvvdp_pypi_distribution_version() -> None:
    def fake_version(_python: str | None, name: str) -> str | None:
        if name == "cvvdp":
            return "0.5.7"
        return None

    with patch("numerical_conformance.provenance._python_package_version", side_effect=fake_version):
        block = collect_learned_provenance(
            learned_python="/env/bin/python",
            learned_config=None,
        )
    assert block["cvvdp"]["cvvdp_version"] == "0.5.7"
    assert "package_version_missing" not in block["cvvdp"]
    assert block["python_packages"]["cvvdp"] == "0.5.7"


def test_cvvdp_alternative_colorvideovdp_still_works() -> None:
    def fake_version(_python: str | None, name: str) -> str | None:
        if name == "colorvideovdp":
            return "1.2.3"
        return None

    with patch("numerical_conformance.provenance._python_package_version", side_effect=fake_version):
        block = collect_learned_provenance(learned_python="/env/bin/python", learned_config=None)
    assert block["cvvdp"]["colorvideovdp_version"] == "1.2.3"
    assert cvvdp_has_package_identity(block["cvvdp"])


def test_cvvdp_missing_when_no_distribution() -> None:
    with patch("numerical_conformance.provenance._python_package_version", return_value=None):
        block = collect_learned_provenance(learned_python="/env/bin/python", learned_config=None)
    assert "package_version_missing" in block["cvvdp"]
    assert not cvvdp_has_package_identity(block["cvvdp"])


def test_validate_provenance_accepts_cvvdp_version() -> None:
    meta = {
        "pixvqa_commit": "abc",
        "pixvqa_version": "1",
        "tolerance_profile": "same_backend_repro_v1",
        "tolerances": {},
        "fixtures": {},
        "generated_at": "2026-01-01T00:00:00+00:00",
        "learned_config": "/cfg.toml",
        "learned_provenance": {
            "cvqa_hdr_nr": {
                "checkpoint_sha256": "a" * 64,
                "profile_sha256": "b" * 64,
            },
            "cvvdp": {"cvvdp_version": "0.5.7"},
        },
    }
    assert not validate_provenance(meta)


def test_snapshot_includes_torchvision_and_open_clip() -> None:
    versions = {
        "torchvision": "0.26.0+cu128",
        "open_clip_torch": "3.1.0",
    }

    def fake_version(_python: str | None, name: str) -> str | None:
        return versions.get(name)

    with patch("numerical_conformance.provenance._python_package_version", side_effect=fake_version):
        block = collect_learned_provenance(learned_python="/env/bin/python", learned_config=None)
    assert block["python_packages"]["torchvision"] == "0.26.0+cu128"
    assert block["python_packages"]["open_clip_torch"] == "3.1.0"


def test_python_package_version_uses_learned_interpreter_subprocess() -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        class R:
            returncode = 0
            stdout = "2.11.0+cu128\n"

        return R()

    with patch("numerical_conformance.provenance.subprocess.run", side_effect=fake_run):
        got = _python_package_version("/learned/bin/python", "torch")
    assert got == "2.11.0+cu128"
    assert calls[0][0] == "/learned/bin/python"
    assert calls[0][1:3] == ["-c", "import importlib.metadata as m\nprint(m.version('torch'))\n"]


def test_python_package_version_never_falls_back_to_harness() -> None:
    harness_version = "HARNESS-WRONG"

    def fake_run(cmd, **kwargs):
        class R:
            returncode = 1
            stdout = ""

        return R()

    with patch("numerical_conformance.provenance.subprocess.run", side_effect=fake_run):
        with patch(
            "importlib.metadata.version",
            return_value=harness_version,
            create=True,
        ):
            got = _python_package_version(sys.executable, "numpy")
    assert got is None


def test_python_package_version_isolated_learned_vs_harness_mock() -> None:
    """Two interpreters return different versions; harness must not leak into learned path."""
    learned = sys.executable

    def fake_run(cmd, **kwargs):
        code = cmd[2]
        class R:
            returncode = 0

        if "torch" in code:
            R.stdout = "2.11.0+cu128\n"
        elif "numpy" in code:
            R.stdout = "2.5.2\n"
        else:
            R.returncode = 1
            R.stdout = ""
        return R()

    with patch("numerical_conformance.provenance.subprocess.run", side_effect=fake_run):
        block = collect_learned_provenance(learned_python=learned, learned_config=None)
    assert block["python_packages"]["torch"] == "2.11.0+cu128"
    assert block["python_packages"]["numpy"] == "2.5.2"


def test_pixvqa_cvvdp_distribution_detected() -> None:
    def fake_version(_python: str | None, name: str) -> str | None:
        if name == "pixvqa-cvvdp":
            return "0.9.0"
        return None

    with patch("numerical_conformance.provenance._python_package_version", side_effect=fake_version):
        block = collect_learned_provenance(learned_python="/env/bin/python", learned_config=None)
    assert block["cvvdp"]["pixvqa_cvvdp_version"] == "0.9.0"


def main() -> int:
    tests = [
        test_cvvdp_pypi_distribution_version,
        test_cvvdp_alternative_colorvideovdp_still_works,
        test_cvvdp_missing_when_no_distribution,
        test_validate_provenance_accepts_cvvdp_version,
        test_snapshot_includes_torchvision_and_open_clip,
        test_python_package_version_uses_learned_interpreter_subprocess,
        test_python_package_version_never_falls_back_to_harness,
        test_python_package_version_isolated_learned_vs_harness_mock,
        test_pixvqa_cvvdp_distribution_detected,
    ]
    for fn in tests:
        fn()
        print(f"OK   {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
