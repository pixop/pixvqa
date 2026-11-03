#!/usr/bin/env python3
"""Regression tests for numerical baseline comparison and tolerances."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.baselines import (  # noqa: E402
    compare_to_baseline,
    validate_provenance,
)
from numerical_conformance.parse_metrics import parse_run  # noqa: E402
from numerical_conformance.registry import not_applicable_reason  # noqa: E402
from numerical_conformance.tolerances import close, within_tolerance  # noqa: E402


def _base_row(**kwargs):
    row = {
        "id": "pq.raw_comp.psnr",
        "family": "pq",
        "metric": "psnr",
        "status": "PASS",
        "psnr_avg": 40.0,
    }
    row.update(kwargs)
    return row


def test_no_approved_baseline_not_evaluated() -> None:
    row = _base_row()
    assert compare_to_baseline(row, {"approved": False, "cases": {}}) is None
    assert row["numerical_regression"] == "NOT_EVALUATED"


def test_missing_baseline_case_fails() -> None:
    baseline = {
        "approved": True,
        "meta": _valid_meta(),
        "cases": {},
    }
    out = compare_to_baseline(_base_row(), baseline)
    assert out is not None
    assert out["status"] == "FAIL"
    assert "missing baseline case" in out["detail"]


def test_missing_baseline_field_fails() -> None:
    baseline = {
        "approved": True,
        "meta": _valid_meta(),
        "cases": {"pq.raw_comp.psnr": {"status": "PASS"}},
    }
    out = compare_to_baseline(_base_row(), baseline)
    assert out is not None
    assert "baseline missing required field psnr_avg" in out["detail"]


def test_missing_observed_field_fails() -> None:
    baseline = {
        "approved": True,
        "meta": _valid_meta(),
        "cases": {"pq.raw_comp.psnr": {"status": "PASS", "psnr_avg": 40.0}},
    }
    row = _base_row(psnr_avg=None)
    del row["psnr_avg"]
    out = compare_to_baseline(row, baseline)
    assert out is not None
    assert "missing observed" in out["detail"]


def test_non_finite_rejected_in_tolerance() -> None:
    ok, detail = within_tolerance("psnr_avg", float("nan"), 40.0)
    assert not ok
    assert detail["reason"] == "non-finite"


def test_tolerance_boundary_accept() -> None:
    ok, _ = within_tolerance("psnr_avg", 40.019, 40.0)
    assert ok
    ok2, _ = within_tolerance("psnr_avg", 40.021, 40.0)
    assert not ok2


def test_scientific_notation_parse() -> None:
    stdout = "[ VMAF mean:1.25e+1 harmonic:12.0 min:10.0 max:15.0 frames:5 ]\n"
    parsed = parse_run(stdout, "", 0)
    assert parsed.vmaf_mean == 12.5


def test_mixed_native_learned_parse() -> None:
    stdout = (
        "Total frames:60\n"
        "[ PSNR  y:40.0 u:40.0 v:40.0 avg:40.0 ] [ MSE  y:0 u:0 v:0 avg:0 ]\n"
        "[ SSIM  y:0.990000 u:0.990000 v:0.990000 avg:0.990000 ]\n"
        "[ CVQA-HDR-NR quality_score:8.244694007429416 raw_model_score:0.25 samples:1 ]\n"
    )
    parsed = parse_run(stdout, "", 0)
    assert parsed.psnr_avg == 40.0
    assert parsed.ssim_avg == 0.99
    assert parsed.cvqa_quality == 8.244694007429416


def test_na_vs_fail_classification() -> None:
    na = not_applicable_reason("cvvdp", "sdr", "raw_raw")
    assert na is not None
    assert na.startswith("N/A:")
    assert not_applicable_reason("cvvdp", "pq", "raw_raw") is None


def test_provenance_validation() -> None:
    assert validate_provenance(None)
    assert not validate_provenance(_valid_meta())


def _valid_meta() -> dict:
    return {
        "pixvqa_commit": "abc",
        "pixvqa_version": "1",
        "tolerance_profile": "same_backend_repro_v1",
        "tolerances": {"psnr_avg": {"abs_tol": 0.02}},
        "fixtures": {"sdr": "/tmp/sdr"},
        "generated_at": "2026-01-01T00:00:00+00:00",
    }


def test_unknown_tolerance_raises() -> None:
    try:
        within_tolerance("unknown_metric_field", 1.0, 1.0)
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown metric field")


def test_passing_baseline_comparison() -> None:
    baseline = {
        "approved": True,
        "meta": _valid_meta(),
        "cases": {
            "pq.raw_comp.group_pq": {
                "status": "PASS",
                "psnr_avg": 40.0,
                "xpsnr_avg": 39.0,
                "ssim_avg": 0.99,
                "msssim_avg": 0.98,
                "vmaf_mean": 90.0,
                "pu21_psnr_avg": 24.0,
                "pu21_ssim_avg": 0.8,
                "pu21_msssim_avg": 0.88,
                "cvvdp_jod": 8.0,
                "cvqa_quality": 7.5,
            }
        },
    }
    row = {
        "id": "pq.raw_comp.group_pq",
        "family": "pq",
        "metric": "group:pq",
        "status": "PASS",
        **baseline["cases"]["pq.raw_comp.group_pq"],
    }
    assert compare_to_baseline(row, baseline) is None
    row["cvqa_quality"] = 7.5 + 1.0
    out = compare_to_baseline(row, baseline)
    assert out is not None
    assert out["numerical_regression"] == "FAIL"


def main() -> int:
    tests = [
        test_no_approved_baseline_not_evaluated,
        test_missing_baseline_case_fails,
        test_missing_baseline_field_fails,
        test_missing_observed_field_fails,
        test_non_finite_rejected_in_tolerance,
        test_tolerance_boundary_accept,
        test_scientific_notation_parse,
        test_mixed_native_learned_parse,
        test_na_vs_fail_classification,
        test_provenance_validation,
        test_unknown_tolerance_raises,
        test_passing_baseline_comparison,
    ]
    for fn in tests:
        fn()
        print(f"OK   {fn.__name__}")
    assert close("ssim_avg", 0.99, 0.99 + 1e-6)
    assert not close("ssim_avg", 0.99, 0.99 + 1e-3)
    assert math.isfinite(8.244694007429416)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
