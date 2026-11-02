#!/usr/bin/env python3
"""Entry point for metric CLI conformance (also invoked from test_metric_parameters.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metric_cli_conformance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
