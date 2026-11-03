"""Aggregate numerical suite validation level verdicts (reporting only)."""
from __future__ import annotations

from typing import Any

INTERNAL_NUMERICAL_CONSISTENCY = "internal_numerical_consistency"
EXTERNAL_REFERENCE_REPRO = "external_reference_reproducibility"


def internal_consistency_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("validation_level") == INTERNAL_NUMERICAL_CONSISTENCY]


def summarize_internal_consistency(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = internal_consistency_rows(results)
    if not rows:
        return {
            "verdict": "NOT_RUN",
            "checks": 0,
            "note": (
                "PixVQA cross-path consistency (HDR inferred vs explicit, raw vs compressed PSNR, "
                "CPU vs GPU, software vs NVDEC) not executed in this profile."
            ),
        }
    failed = [r["id"] for r in rows if r.get("status") == "FAIL"]
    verdict = "FAIL" if failed else "PASS"
    return {
        "verdict": verdict,
        "checks": len(rows),
        "failed_ids": failed,
        "note": (
            "Same-build PixVQA consistency across paths and backends; not independent implementation parity."
        ),
    }


def summarize_independent_reference_parity(berlin_block: dict[str, Any] | None) -> dict[str, Any]:
    if berlin_block is None:
        return {
            "verdict": "NOT_EVALUATED",
            "berlin_ran": False,
            "scope": None,
            "metrics": [],
            "note": (
                "No external reference comparison executed. Set PIXVQA_BERLIN_REF and "
                "PIXVQA_BERLIN_TEST for optional CVVDP reproducibility only."
            ),
        }
    rows = berlin_block.get("berlin_cvvdp") or []
    failed = [r for r in rows if r.get("status") == "FAIL"]
    verdict = "FAIL" if failed else "PASS"
    return {
        "verdict": verdict,
        "berlin_ran": True,
        "scope": berlin_block.get("reference_scope", "cvvdp_jod"),
        "metrics": list(berlin_block.get("metrics") or ["cvvdp"]),
        "provenance_note": berlin_block.get("provenance_note"),
        "failed_frames": [r.get("frames") for r in failed],
        "note": (
            "Optional BerlinSeqs CVVDP JOD reproducibility against pinned constants in berlin.py. "
            "Does not assert parity for other metrics."
        ),
    }


def overall_verdict_includes_internal_fail(internal_summary: dict[str, Any]) -> bool:
    return internal_summary.get("verdict") == "FAIL"


def overall_verdict_includes_reference_fail(reference_summary: dict[str, Any]) -> bool:
    return reference_summary.get("verdict") == "FAIL"
