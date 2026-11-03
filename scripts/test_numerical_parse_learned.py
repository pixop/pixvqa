#!/usr/bin/env python3
"""Parse contract tests for learned-metric numerical outputs."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.parse_metrics import parse_run  # noqa: E402


def test_cvvdp_json_without_total_frames() -> None:
    stdout = '{"metric":"cvvdp","jod":8.5,"interpretation":"pq"}\n'
    parsed = parse_run(stdout, "", 0)
    assert parsed.total_frames is None
    assert parsed.cvvdp_jod == 8.5


def test_native_total_frames() -> None:
    stdout = "Total frames:5\nPSNR y:40 u:40 v:40 avg:40\n"
    parsed = parse_run(stdout, "", 0)
    assert parsed.total_frames == 5
    assert parsed.psnr_avg == 40.0


def test_cvqa_json_quality() -> None:
    stdout = (
        '{"metric":"cvqa-hdr-nr","quality_score":7.41,'
        '"raw_model_score":0.12,"sampled_frames":2}\n'
    )
    parsed = parse_run(stdout, "", 0)
    assert parsed.cvqa_quality == 7.41
    assert math.isfinite(parsed.cvqa_quality)


def test_cvqa_mixed_summary_line() -> None:
    stdout = (
        "Total frames:60\n"
        "PSNR y:40 u:40 v:40 avg:40\n"
        "[ CVQA-HDR-NR quality_score:8.244694007429416 "
        "raw_model_score:0.2540421187877655 samples:1 frames:60 density:1.000/s device:cuda ]\n"
    )
    parsed = parse_run(stdout, "", 0)
    assert parsed.cvqa_quality == 8.244694007429416
    assert parsed.psnr_avg == 40.0


def test_cvqa_mixed_summary_scientific() -> None:
    stdout = "[ CVQA-HDR-NR quality_score:1.25e+1 raw_model_score:0.1 samples:1 ]\n"
    parsed = parse_run(stdout, "", 0)
    assert parsed.cvqa_quality == 12.5


def test_cvqa_ignores_other_metric_json() -> None:
    stdout = '{"metric":"other","quality_score":9.99,"raw_model_score":0.1}\n'
    parsed = parse_run(stdout, "", 0)
    assert parsed.cvqa_quality is None


def test_cvqa_rejects_non_finite_mixed() -> None:
    stdout = "[ CVQA-HDR-NR quality_score:nan raw_model_score:0.1 ]\n"
    parsed = parse_run(stdout, "", 0)
    assert parsed.cvqa_quality is None


def main() -> int:
    test_cvvdp_json_without_total_frames()
    print("OK   cvvdp-json-no-total-frames")
    test_native_total_frames()
    print("OK   native-total-frames")
    test_cvqa_json_quality()
    print("OK   cvqa-json-quality")
    test_cvqa_mixed_summary_line()
    print("OK   cvqa-mixed-summary-line")
    test_cvqa_mixed_summary_scientific()
    print("OK   cvqa-mixed-summary-scientific")
    test_cvqa_ignores_other_metric_json()
    print("OK   cvqa-ignores-other-metric-json")
    test_cvqa_rejects_non_finite_mixed()
    print("OK   cvqa-rejects-non-finite-mixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
