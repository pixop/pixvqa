#!/usr/bin/env python3
"""Regression tests for metric CLI fixture helpers (no pixvqa binary required)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metric_test_fixtures import install_cvvdp_mock, verify_cvvdp_mock_repeatable  # noqa: E402


def main() -> int:
    verify_cvvdp_mock_repeatable(iterations=10)
    with tempfile.TemporaryDirectory(prefix="pixvqa_mock_pkg_") as tmp:
        root = Path(tmp)
        install_cvvdp_mock(root)
        pkg = root / "mock" / "pixvqa_cvvdp"
        if not (pkg / "__main__.py").is_file():
            print("FAIL missing __main__.py after install", file=sys.stderr)
            return 1
        text = (pkg / "__main__.py").read_text(encoding="utf-8")
        if "pixvqa-protocol" not in text:
            print("FAIL mock missing line protocol branch", file=sys.stderr)
            return 1
    print("OK   metric-fixture-harness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
