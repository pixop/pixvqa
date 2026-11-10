#!/usr/bin/env python3
"""Regression: migrate legacy [learned].path ~/bin to managed venv bin."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pixvqa_learned_install.config_sync import merge_config  # noqa: E402


def _merge(dest_body: str, managed_bin: str) -> dict:
    example = ROOT / "config" / "learned_metrics.example.toml"
    dest = Path(tempfile.mkdtemp()) / "learned_metrics.local.toml"
    try:
        dest.write_text(dest_body, encoding="utf-8")
        merge_config(
            example,
            dest,
            {
                "learned.path": managed_bin,
                "learned.python": f"{managed_bin}/python",
            },
        )
        return tomllib.loads(dest.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(dest.parent)


def test_legacy_tilde_bin_migrates() -> None:
    managed = "/managed/learned-venv/bin"
    for legacy in ("~/bin", " ~/bin ", '~/bin/'):
        parsed = _merge(
            f"[learned]\npath = {json.dumps(legacy)}\npython = \"../.venv/bin/python\"\n",
            managed,
        )
        assert parsed["learned"]["path"] == managed, (legacy, parsed)
    print("PASS: legacy ~/bin migrated to managed venv bin")


def test_custom_path_preserved() -> None:
    managed = "/managed/learned-venv/bin"
    custom = "/opt/custom/ffmpeg/bin"
    parsed = _merge(
        f"[learned]\npath = {json.dumps(custom)}\npython = \"../.venv/bin/python\"\n",
        managed,
    )
    assert parsed["learned"]["path"] == custom
    print("PASS: deliberate custom learned.path preserved")


def test_empty_and_fresh_install() -> None:
    managed = "/managed/learned-venv/bin"
    work = Path(tempfile.mkdtemp())
    try:
        example = ROOT / "config" / "learned_metrics.example.toml"
        dest = work / "learned_metrics.local.toml"
        dest.write_text(
            '[learned]\npath = ""\npython = "../.venv/bin/python"\n',
            encoding="utf-8",
        )
        merge_config(example, dest, {"learned.path": managed, "learned.python": f"{managed}/python"})
        assert tomllib.loads(dest.read_text())["learned"]["path"] == managed

        fresh = work / "fresh.toml"
        merge_config(example, fresh, {"learned.path": managed, "learned.python": f"{managed}/python"})
        assert tomllib.loads(fresh.read_text())["learned"]["path"] == managed
    finally:
        shutil.rmtree(work)
    print("PASS: empty and fresh installs receive managed learned.path")


def main() -> int:
    test_legacy_tilde_bin_migrates()
    test_custom_path_preserved()
    test_empty_and_fresh_install()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
