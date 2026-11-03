"""Shared constants for UHD numerical conformance fixtures."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WIDTH = 3840
HEIGHT = 2160
FRAMES = 5
FPS_NUM = 60
FPS_DEN = 1
FPS = f"{FPS_NUM}/{FPS_DEN}"
CHROMA = "420p:10"
SIZE = f"{WIDTH}x{HEIGHT}"

# Cached generated media (gitignored via .gitignore patterns on *.yuv / *.mkv under cache).
CACHE_ROOT = Path(
    os.environ.get(
        "PIXVQA_NUMERICAL_FIXTURE_CACHE",
        ROOT / ".cache" / "pixvqa-numerical-fixtures",
    )
)

BASELINE_ROOT = Path(
    os.environ.get(
        "PIXVQA_NUMERICAL_BASELINE_DIR",
        ROOT / "baselines" / "numerical_conformance",
    )
)

FULL_BIN = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "full" / "pixvqa"))
CPU_BIN = Path(os.environ.get("PIXVQA_CPU_BIN", ROOT / "build" / "cpu" / "pixvqa"))
SWDEC_BIN = Path(os.environ.get("PIXVQA_DECODE_BIN", ROOT / "build" / "swdec" / "pixvqa"))

BERLIN_CVVDP_JOD = {
    5: 8.788558959960938,
    60: 9.17333984375,
}
