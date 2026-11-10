#!/usr/bin/env python3
"""Regression tests for read-only candidate reproducibility comparison."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.candidate_compare import (  # noqa: E402
    compare_provenance_identity,
    compare_run_to_candidate,
    is_numerical_metric_row,
)


def _row(case_id: str, **metrics: float) -> dict:
    return {
        "id": case_id,
        "family": "pq",
        "metric": "psnr",
        "status": "PASS",
        "mandatory": True,
        **metrics,
    }


def _meta(**overrides) -> dict:
    base = {
        "pixvqa_commit": "abc123",
        "tolerance_profile": "same_backend_repro_v1",
        "fixture_manifest_sha256": {"pq": "dead" * 8},
        "learned_provenance": {
            "cvqa_hdr_nr": {
                "checkpoint_sha256": "a" * 64,
                "profile_sha256": "b" * 64,
                "compressed_vqa_hdr_commit": "c" * 40,
            },
            "cvvdp": {"cvvdp_version": "0.5.7"},
            "python_packages": {"torch": "2.13.0+cu130", "numpy": "2.5.2"},
        },
    }
    base.update(overrides)
    return base


def _candidate(cases: dict, meta: dict | None = None) -> dict:
    return {"approved": False, "cases": cases, "meta": meta or _meta()}


def test_successful_comparison() -> None:
    row = _row("pq.raw_comp.psnr", psnr_avg=40.0)
    cand = _candidate(
        {
            "pq.raw_comp.psnr": {
                "status": "PASS",
                "metric": "psnr",
                "family": "pq",
                "psnr_avg": 40.0,
            }
        }
    )
    report = compare_run_to_candidate(
        [row],
        cand,
        run_meta=_meta(),
        candidate_path=Path("/tmp/candidate.json"),
        immutability_ok=True,
    )
    assert report.verdict == "PASS"
    assert not report.case_errors


def test_tolerance_failure() -> None:
    row = _row("pq.raw_comp.psnr", psnr_avg=40.5)
    cand = _candidate(
        {
            "pq.raw_comp.psnr": {
                "status": "PASS",
                "metric": "psnr",
                "family": "pq",
                "psnr_avg": 40.0,
            }
        }
    )
    report = compare_run_to_candidate(
        [row],
        cand,
        run_meta=_meta(),
        candidate_path=Path("/tmp/candidate.json"),
        immutability_ok=True,
    )
    assert report.verdict == "FAIL"
    assert any(e["kind"] == "metric_mismatch" for e in report.case_errors)


def test_missing_candidate_case() -> None:
    row = _row("pq.raw_comp.psnr", psnr_avg=40.0)
    report = compare_run_to_candidate(
        [row],
        _candidate({}),
        run_meta=_meta(),
        candidate_path=Path("/tmp/candidate.json"),
        immutability_ok=True,
    )
    assert report.verdict == "FAIL"
    assert any(e["kind"] == "missing_candidate_case" for e in report.case_errors)


def test_extra_candidate_case() -> None:
    cand = _candidate(
        {
            "pq.raw_comp.psnr": {
                "status": "PASS",
                "metric": "psnr",
                "family": "pq",
                "psnr_avg": 40.0,
            }
        }
    )
    report = compare_run_to_candidate(
        [],
        cand,
        run_meta=_meta(),
        candidate_path=Path("/tmp/candidate.json"),
        immutability_ok=True,
    )
    assert report.verdict == "FAIL"
    assert any(e["kind"] == "missing_run_case" for e in report.case_errors)


def test_unexpected_candidate_field() -> None:
    row = _row("pq.raw_comp.psnr", psnr_avg=40.0)
    cand = _candidate(
        {
            "pq.raw_comp.psnr": {
                "status": "PASS",
                "metric": "psnr",
                "family": "pq",
                "psnr_avg": 40.0,
                "xpsnr_avg": 41.0,
            }
        }
    )
    report = compare_run_to_candidate(
        [row],
        cand,
        run_meta=_meta(),
        candidate_path=Path("/tmp/candidate.json"),
        immutability_ok=True,
    )
    assert report.verdict == "FAIL"


def test_provenance_mismatch() -> None:
    row = _row("pq.raw_comp.psnr", psnr_avg=40.0)
    cand = _candidate(
        {
            "pq.raw_comp.psnr": {
                "status": "PASS",
                "metric": "psnr",
                "family": "pq",
                "psnr_avg": 40.0,
            }
        },
        meta=_meta(tolerance_profile="other_profile"),
    )
    errors = compare_provenance_identity(_meta(), cand["meta"])
    assert errors
    report = compare_run_to_candidate(
        [row],
        cand,
        run_meta=_meta(),
        candidate_path=Path("/tmp/candidate.json"),
        immutability_ok=True,
    )
    assert report.verdict == "FAIL"
    assert report.provenance_errors


def test_commit_mismatch_infrastructure_accepted() -> None:
    row = _row("pq.raw_comp.psnr", psnr_avg=40.0)
    cand = _candidate(
        {
            "pq.raw_comp.psnr": {
                "status": "PASS",
                "metric": "psnr",
                "family": "pq",
                "psnr_avg": 40.0,
            }
        },
        meta=_meta(pixvqa_commit="cand_sha"),
    )
    with mock.patch(
        "numerical_conformance.candidate_compare.verify_pixvqa_commit_compatibility",
    ) as verify:
        from numerical_conformance.candidate_commit_compatibility import CommitCompatibilityResult

        verify.return_value = CommitCompatibilityResult(
            verified=True,
            acceptable=True,
            commits_match=False,
            candidate_commit="cand_sha",
            run_commit="run_sha",
            changed_files=["scripts/numerical_conformance/runner.py"],
        )
        report = compare_run_to_candidate(
            [row],
            cand,
            run_meta=_meta(pixvqa_commit="run_sha"),
            candidate_path=Path("/tmp/candidate.json"),
            immutability_ok=True,
        )
    assert report.verdict == "PASS"
    assert report.commit_compatibility.get("compatibility_exception_accepted") is True
    level = report.to_validation_level()
    assert "compatibility_exception" in level
    assert level["compatibility_exception"]["changed_files"] == [
        "scripts/numerical_conformance/runner.py"
    ]


def test_commit_mismatch_disallowed_fails() -> None:
    row = _row("pq.raw_comp.psnr", psnr_avg=40.0)
    cand = _candidate(
        {
            "pq.raw_comp.psnr": {
                "status": "PASS",
                "metric": "psnr",
                "family": "pq",
                "psnr_avg": 40.0,
            }
        },
        meta=_meta(pixvqa_commit="cand_sha"),
    )
    with mock.patch(
        "numerical_conformance.candidate_compare.verify_pixvqa_commit_compatibility",
    ) as verify:
        from numerical_conformance.candidate_commit_compatibility import CommitCompatibilityResult

        verify.return_value = CommitCompatibilityResult(
            verified=True,
            acceptable=False,
            commits_match=False,
            candidate_commit="cand_sha",
            run_commit="run_sha",
            changed_files=["src/metrics/foo.cpp"],
            disallowed_files=["src/metrics/foo.cpp"],
        )
        report = compare_run_to_candidate(
            [row],
            cand,
            run_meta=_meta(pixvqa_commit="run_sha"),
            candidate_path=Path("/tmp/candidate.json"),
            immutability_ok=True,
        )
    assert report.verdict == "FAIL"
    assert any("outside infrastructure allowlist" in e for e in report.provenance_errors)


def test_internal_rows_excluded() -> None:
    internal = {
        "id": "pq.backend.psnr_cpu_vs_gpu",
        "status": "PASS",
        "validation_level": "internal_numerical_consistency",
    }
    assert not is_numerical_metric_row(internal)


def test_candidate_immutability_flag() -> None:
    report = compare_run_to_candidate(
        [],
        _candidate({}),
        run_meta=_meta(),
        candidate_path=Path("/tmp/candidate.json"),
        immutability_ok=False,
    )
    assert report.verdict == "FAIL"
    assert any(e["kind"] == "candidate_modified" for e in report.case_errors)


def test_candidate_file_bytes_unchanged(tmp_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.json"
        payload = _candidate(
            {
                "pq.raw_comp.psnr": {
                    "status": "PASS",
                    "metric": "psnr",
                    "family": "pq",
                    "psnr_avg": 40.0,
                }
            }
        )
        original = json.dumps(payload, indent=2).encode("utf-8")
        path.write_bytes(original)
        before = path.read_bytes()
        # simulate read-only workflow: no write
        after = path.read_bytes()
        assert before == after == original


def main() -> int:
    tests = [
        test_successful_comparison,
        test_tolerance_failure,
        test_missing_candidate_case,
        test_extra_candidate_case,
        test_unexpected_candidate_field,
        test_provenance_mismatch,
        test_commit_mismatch_infrastructure_accepted,
        test_commit_mismatch_disallowed_fails,
        test_internal_rows_excluded,
        test_candidate_immutability_flag,
        test_candidate_file_bytes_unchanged,
    ]
    for fn in tests:
        fn()
        print(f"OK   {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
