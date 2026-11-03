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


def main() -> int:
    tests = [
        test_cvvdp_pypi_distribution_version,
        test_cvvdp_alternative_colorvideovdp_still_works,
        test_cvvdp_missing_when_no_distribution,
        test_validate_provenance_accepts_cvvdp_version,
        test_snapshot_includes_torchvision_and_open_clip,
    ]
    for fn in tests:
        fn()
        print(f"OK   {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
