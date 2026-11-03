#!/usr/bin/env python3
"""Run CLI conformance (phase 1) and numerical conformance (phase 2)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="PixVQA validation: CLI + numerical")
    parser.add_argument("--skip-cli", action="store_true")
    parser.add_argument("--skip-numerical", action="store_true")
    parser.add_argument("--profile", choices=("smoke", "comprehensive"), default="smoke")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    strict = args.strict or os.environ.get("PIXVQA_VALIDATION_STRICT", "") in ("1", "true", "yes")
    rc = 0
    if not args.skip_cli:
        cli = subprocess.run(
            [py, str(ROOT / "scripts" / "test_metric_parameters.py")],
            env={**os.environ, "PIXVQA_CLI_CONFORMANCE_STRICT": "1" if strict else "0"},
        )
        if cli.returncode != 0:
            rc = 1
    if not args.skip_numerical:
        num = subprocess.run(
            [
                py,
                str(ROOT / "scripts" / "test_metric_numerical_conformance.py"),
                "--profile",
                args.profile,
            ]
            + (["--strict"] if strict else []),
        )
        if num.returncode != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
