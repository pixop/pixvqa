"""Read-only comparison of a live run against an unapproved candidate baseline."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .baselines import compare_pass_row_to_reference, validate_baseline_cases
from .metric_fields import allowed_baseline_case_keys


def load_candidate_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("candidate file must be a JSON object")
    return data


def is_numerical_metric_row(row: dict[str, Any]) -> bool:
    """Executable numerical cases (excludes synthetic internal-consistency rows)."""
    return "metric" in row and "family" in row


def numerical_pass_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in results
        if r.get("status") == "PASS" and is_numerical_metric_row(r)
    ]


def mandatory_numerical_pass_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in numerical_pass_rows(results) if r.get("mandatory", True)]


PROVENANCE_PACKAGE_KEYS = (
    "cvvdp",
    "torch",
    "numpy",
    "torchvision",
    "open_clip_torch",
)


def compare_provenance_identity(run_meta: dict[str, Any], candidate_meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("pixvqa_commit", "tolerance_profile"):
        if run_meta.get(key) != candidate_meta.get(key):
            errors.append(
                f"provenance {key} mismatch: run={run_meta.get(key)!r} candidate={candidate_meta.get(key)!r}"
            )
    run_fix = run_meta.get("fixture_manifest_sha256") or {}
    cand_fix = candidate_meta.get("fixture_manifest_sha256") or {}
    if run_fix != cand_fix:
        errors.append("provenance fixture_manifest_sha256 mismatch")

    run_learned = run_meta.get("learned_provenance") or {}
    cand_learned = candidate_meta.get("learned_provenance") or {}
    run_cvqa = run_learned.get("cvqa_hdr_nr") or {}
    cand_cvqa = cand_learned.get("cvqa_hdr_nr") or {}
    for key in ("checkpoint_sha256", "profile_sha256", "compressed_vqa_hdr_commit"):
        cval = cand_cvqa.get(key)
        if cval is not None and run_cvqa.get(key) != cval:
            errors.append(
                f"provenance cvqa_hdr_nr.{key} mismatch: run={run_cvqa.get(key)!r} candidate={cval!r}"
            )

    run_cvvdp = run_learned.get("cvvdp") or {}
    cand_cvvdp = cand_learned.get("cvvdp") or {}
    for key, val in cand_cvvdp.items():
        if not key.endswith("_version"):
            continue
        if val and run_cvvdp.get(key) != val:
            errors.append(f"provenance cvvdp.{key} mismatch: run={run_cvvdp.get(key)!r} candidate={val!r}")

    run_pkgs = run_learned.get("python_packages") or {}
    cand_pkgs = cand_learned.get("python_packages") or {}
    for key in PROVENANCE_PACKAGE_KEYS:
        cval = cand_pkgs.get(key)
        if cval and run_pkgs.get(key) != cval:
            errors.append(
                f"provenance python_packages.{key} mismatch: "
                f"run={run_pkgs.get(key)!r} candidate={cval!r}"
            )
    return errors


@dataclass
class CandidateCompareReport:
    verdict: str
    candidate_path: str
    immutability_ok: bool
    provenance_errors: list[str] = field(default_factory=list)
    case_errors: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_validation_level(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "candidate_path": self.candidate_path,
            "immutability_ok": self.immutability_ok,
            "provenance_errors": self.provenance_errors,
            "case_error_count": len(self.case_errors),
            "case_errors": self.case_errors,
            "comparisons": self.comparisons,
            "note": self.note,
        }


def compare_run_to_candidate(
    results: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    run_meta: dict[str, Any],
    candidate_path: Path,
    immutability_ok: bool,
) -> CandidateCompareReport:
    cand_cases = candidate.get("cases") or {}
    if not isinstance(cand_cases, dict):
        return CandidateCompareReport(
            verdict="FAIL",
            candidate_path=str(candidate_path),
            immutability_ok=immutability_ok,
            case_errors=[{"kind": "invalid_candidate", "detail": "cases must be an object"}],
            note="Candidate structure invalid.",
        )

    case_errors: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for err in validate_baseline_cases(cand_cases):
        case_errors.append({"kind": "invalid_candidate_case", "detail": err})

    provenance_errors = compare_provenance_identity(run_meta, candidate.get("meta") or {})

    pass_by_id = {r["id"]: r for r in numerical_pass_rows(results)}
    mandatory_pass = {r["id"]: r for r in mandatory_numerical_pass_rows(results)}

    for cid in sorted(mandatory_pass):
        if cid not in cand_cases:
            case_errors.append(
                {"kind": "missing_candidate_case", "id": cid, "detail": f"mandatory PASS case {cid} absent from candidate"}
            )

    for cid in sorted(cand_cases):
        if cid not in pass_by_id:
            case_errors.append(
                {
                    "kind": "missing_run_case",
                    "id": cid,
                    "detail": f"candidate case {cid} has no matching PASS run row",
                }
            )
            continue
        row = pass_by_id[cid]
        expected = cand_cases[cid]
        extra = set(expected.keys()) - allowed_baseline_case_keys(
            expected.get("metric", row["metric"]),
            expected.get("family", row["family"]),
        )
        if extra:
            case_errors.append(
                {
                    "kind": "unexpected_candidate_fields",
                    "id": cid,
                    "fields": sorted(extra),
                }
            )
            continue
        ok, detail, comps = compare_pass_row_to_reference(row, expected, cid)
        if comps:
            comparisons.extend({**c, "case_id": cid} for c in comps)
        if not ok:
            case_errors.append({"kind": "metric_mismatch", "id": cid, "detail": detail, "comparisons": comps})

    if not immutability_ok:
        case_errors.append(
            {
                "kind": "candidate_modified",
                "detail": "candidate file bytes changed during the suite run",
            }
        )

    fail = bool(case_errors or provenance_errors or not immutability_ok)
    verdict = "FAIL" if fail else "PASS"
    note = (
        "Read-only comparison against unapproved candidate; does not approve baseline or "
        "change numerical_regression verdict."
    )
    return CandidateCompareReport(
        verdict=verdict,
        candidate_path=str(candidate_path),
        immutability_ok=immutability_ok,
        provenance_errors=provenance_errors,
        case_errors=case_errors,
        comparisons=comparisons,
        note=note,
    )
