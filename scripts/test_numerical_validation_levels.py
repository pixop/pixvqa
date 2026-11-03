#!/usr/bin/env python3
"""Regression tests for validation level reporting separation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.validation_levels import (  # noqa: E402
    INTERNAL_NUMERICAL_CONSISTENCY,
    overall_verdict_includes_internal_fail,
    overall_verdict_includes_reference_fail,
    summarize_independent_reference_parity,
    summarize_internal_consistency,
)


def test_independent_not_evaluated_without_berlin() -> None:
    summary = summarize_independent_reference_parity(None)
    assert summary["verdict"] == "NOT_EVALUATED"
    assert summary["berlin_ran"] is False
    assert summary["metrics"] == []
    assert not overall_verdict_includes_reference_fail(summary)


def test_internal_passing_comparisons() -> None:
    results = [
        {"id": "pq.hdr.inferred_vs_explicit", "status": "PASS", "validation_level": INTERNAL_NUMERICAL_CONSISTENCY},
        {"id": "pq.backend.psnr_cpu_vs_gpu", "status": "PASS", "validation_level": INTERNAL_NUMERICAL_CONSISTENCY},
    ]
    summary = summarize_internal_consistency(results)
    assert summary["verdict"] == "PASS"
    assert summary["checks"] == 2
    assert not overall_verdict_includes_internal_fail(summary)


def test_internal_failing_comparison() -> None:
    results = [
        {"id": "sdr.consistency.psnr_raw_vs_comp", "status": "FAIL", "validation_level": INTERNAL_NUMERICAL_CONSISTENCY},
    ]
    summary = summarize_internal_consistency(results)
    assert summary["verdict"] == "FAIL"
    assert overall_verdict_includes_internal_fail(summary)


def test_internal_not_run_when_no_rows() -> None:
    summary = summarize_internal_consistency([])
    assert summary["verdict"] == "NOT_RUN"


def test_berlin_pass_scoped_to_cvvdp_only() -> None:
    block = {
        "berlin_cvvdp": [
            {"frames": 5, "status": "PASS", "metric": "cvvdp"},
            {"frames": 60, "status": "PASS", "metric": "cvvdp"},
        ],
        "reference_scope": "cvvdp_jod",
        "metrics": ["cvvdp"],
        "provenance_note": "Pinned historical JOD constants in berlin.py; not independently verified by this harness.",
    }
    summary = summarize_independent_reference_parity(block)
    assert summary["verdict"] == "PASS"
    assert summary["berlin_ran"] is True
    assert summary["metrics"] == ["cvvdp"]
    assert summary["scope"] == "cvvdp_jod"
    assert "not independently verified" in (summary.get("provenance_note") or "")


def test_berlin_partial_failure() -> None:
    block = {
        "berlin_cvvdp": [
            {"frames": 5, "status": "PASS"},
            {"frames": 60, "status": "FAIL"},
        ],
        "reference_scope": "cvvdp_jod",
        "metrics": ["cvvdp"],
    }
    summary = summarize_independent_reference_parity(block)
    assert summary["verdict"] == "FAIL"
    assert overall_verdict_includes_reference_fail(summary)


def main() -> int:
    tests = [
        test_independent_not_evaluated_without_berlin,
        test_internal_passing_comparisons,
        test_internal_failing_comparison,
        test_internal_not_run_when_no_rows,
        test_berlin_pass_scoped_to_cvvdp_only,
        test_berlin_partial_failure,
    ]
    for fn in tests:
        fn()
        print(f"OK   {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
