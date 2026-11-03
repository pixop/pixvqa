"""Optional BerlinSeqs CVVDP regression (local asset, not portable fixtures)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import FULL_BIN, SIZE
from .parse_metrics import parse_run
from .tolerances import close

BERLIN_5F = 8.788558959960938
BERLIN_60F = 9.17333984375

BERLIN_PROVENANCE_NOTE = (
    "Pinned historical JOD constants in berlin.py; not independently verified by this harness."
)


def run_berlin_optional(env_full_bin: Path | None = None) -> dict | None:
    ref = os.environ.get("PIXVQA_BERLIN_REF")
    test = os.environ.get("PIXVQA_BERLIN_TEST")
    if not ref or not test:
        return None
    bin_path = env_full_bin or FULL_BIN
    rows = []
    for n, expected in ((5, BERLIN_5F), (60, BERLIN_60F)):
        cmd = [
            str(bin_path),
            "--metrics",
            "cvvdp",
            "-r",
            ref,
            "-t",
            test,
            "-s",
            SIZE,
            "-f",
            "420",
            "-n",
            str(n),
            "-v",
            "1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        parsed = parse_run(proc.stdout, proc.stderr, proc.returncode)
        ok = proc.returncode == 0 and parsed.cvvdp_jod is not None and close(
            "cvvdp_jod", expected, parsed.cvvdp_jod, mode="reference_parity"
        )
        rows.append(
            {
                "frames": n,
                "expected_jod": expected,
                "actual_jod": parsed.cvvdp_jod,
                "status": "PASS" if ok else "FAIL",
                "metric": "cvvdp",
                "field": "cvvdp_jod",
            }
        )
    return {
        "berlin_cvvdp": rows,
        "reference_scope": "cvvdp_jod",
        "metrics": ["cvvdp"],
        "provenance_note": BERLIN_PROVENANCE_NOTE,
        "validation_level": "external_reference_reproducibility",
    }
