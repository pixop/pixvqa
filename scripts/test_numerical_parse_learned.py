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


def test_xpsnr_inf_chroma_planes_rtx_summary() -> None:
    stdout = (
        "1:[ XPSNR y:50.0 u:50.0 v:50.0 avg:50.0 min:50.0 ]\n"
        "\nSummary:\n\n"
        "[ XPSNR y:52.2602 u:inf v:inf avg:54.0211 min:52.2602 ]\n"
        "Total frames:5\n"
    )
    parsed = parse_run(stdout, "", 0)
    assert parsed.xpsnr_avg == 54.0211
    assert parsed.total_frames == 5


def test_xpsnr_all_finite_planes() -> None:
    stdout = "[ XPSNR y:52.2602 u:52.2602 v:52.2602 avg:54.0211 min:52.2602 ]\n"
    parsed = parse_run(stdout, "", 0)
    assert parsed.xpsnr_avg == 54.0211


def test_xpsnr_scientific_avg() -> None:
    stdout = "[ XPSNR y:1.0 u:inf v:inf avg:1.25e+1 min:1.0 ]\n"
    parsed = parse_run(stdout, "", 0)
    assert parsed.xpsnr_avg == 12.5


def test_xpsnr_rejects_non_finite_avg() -> None:
    assert parse_run("[ XPSNR y:1 u:inf v:inf avg:inf min:1 ]\n", "", 0).xpsnr_avg is None
    assert parse_run("[ XPSNR y:1 u:inf v:inf avg:nan min:1 ]\n", "", 0).xpsnr_avg is None


def test_xpsnr_malformed_no_match() -> None:
    parsed = parse_run("[ XPSNR broken line ]\n", "", 0)
    assert parsed.xpsnr_avg is None


def test_xpsnr_only_no_psnr_contamination() -> None:
    stdout = """[ XPSNR y:52.2602 u:inf v:inf avg:54.0211 min:52.2602 ]
Total frames:5
"""
    parsed = parse_run(stdout, "", 0)
    assert parsed.psnr_avg is None
    assert parsed.xpsnr_avg == 54.0211


def test_metric_name_collision_matrix() -> None:
    stdout = (
        "[ PU21-PSNR     r:24.0 g:24.0 b:24.0 avg:24.0 ]\n"
        "[ PU21-SSIM     y:0.80 u:n/a v:n/a avg:0.80 ]\n"
        "[ PU21-MS-SSIM  y:0.88 u:n/a v:n/a avg:0.88 ]\n"
        "[ MS-SSIM  y:0.98 u:0.98 v:0.98 avg:0.98 ]\n"
        "[ SSIM  y:0.99 u:inf v:inf avg:0.99 ]\n"
        "[ PSNR  y:40.0 u:40.0 v:40.0 avg:40.0 ] [ MSE  y:0 u:0 v:0 avg:0 ]\n"
        "[ XPSNR y:52.2602 u:inf v:inf avg:54.0211 min:52.2602 ]\n"
    )
    parsed = parse_run(stdout, "", 0)
    assert parsed.psnr_avg == 40.0
    assert parsed.xpsnr_avg == 54.0211
    assert parsed.ssim_avg == 0.99
    assert parsed.msssim_avg == 0.98
    assert parsed.pu21_psnr_avg == 24.0
    assert parsed.pu21_ssim_avg == 0.80
    assert parsed.pu21_msssim_avg == 0.88

    xonly = parse_run("[ XPSNR y:1 u:inf v:inf avg:2 min:1 ]\n", "", 0)
    assert xonly.psnr_avg is None
    assert xonly.ssim_avg is None
    assert xonly.pu21_psnr_avg is None


def test_ssim_inf_chroma_finite_avg() -> None:
    stdout = "[ SSIM  y:0.990000 u:inf v:inf avg:0.990000 ]\n"
    parsed = parse_run(stdout, "", 0)
    assert parsed.ssim_avg == 0.99


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
    test_xpsnr_inf_chroma_planes_rtx_summary()
    print("OK   xpsnr-inf-chroma-planes-rtx-summary")
    test_xpsnr_all_finite_planes()
    print("OK   xpsnr-all-finite-planes")
    test_xpsnr_scientific_avg()
    print("OK   xpsnr-scientific-avg")
    test_xpsnr_rejects_non_finite_avg()
    print("OK   xpsnr-rejects-non-finite-avg")
    test_xpsnr_malformed_no_match()
    print("OK   xpsnr-malformed-no-match")
    test_xpsnr_only_no_psnr_contamination()
    print("OK   xpsnr-only-no-psnr-contamination")
    test_metric_name_collision_matrix()
    print("OK   metric-name-collision-matrix")
    test_ssim_inf_chroma_finite_avg()
    print("OK   ssim-inf-chroma-finite-avg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
