"""Read-only comparison of a live run against an unapproved candidate baseline."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .baselines import compare_pass_row_to_reference, validate_baseline_cases
from .candidate_commit_compatibility import (
    POLICY_ID,
    verify_pixvqa_commit_compatibility,
)
from .config import ROOT
from .metric_fields import allowed_baseline_case_keys
from .validation_levels import is_numerical_metric_row


def load_candidate_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("candidate file must be a JSON object")
    return data


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
    """Strict artifact provenance (Git commit handled separately via compatibility policy)."""
    errors: list[str] = []
    if run_meta.get("tolerance_profile") != candidate_meta.get("tolerance_profile"):
        errors.append(
            "provenance tolerance_profile mismatch: "
            f"run={run_meta.get('tolerance_profile')!r} "
            f"candidate={candidate_meta.get('tolerance_profile')!r}"
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
    commit_compatibility: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_validation_level(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verdict": self.verdict,
            "candidate_path": self.candidate_path,
            "immutability_ok": self.immutability_ok,
            "provenance_errors": self.provenance_errors,
            "case_error_count": len(self.case_errors),
            "case_errors": self.case_errors,
            "comparisons": self.comparisons,
            "commit_compatibility": self.commit_compatibility,
            "note": self.note,
        }
        if self.commit_compatibility.get("compatibility_exception_accepted"):
            out["compatibility_exception"] = {
                "policy": self.commit_compatibility.get("policy", POLICY_ID),
                "candidate_pixvqa_commit": self.commit_compatibility.get("candidate_pixvqa_commit"),
                "run_pixvqa_commit": self.commit_compatibility.get("run_pixvqa_commit"),
                "changed_files": self.commit_compatibility.get("changed_files", []),
            }
        return out


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

    candidate_meta = candidate.get("meta") or {}
    provenance_errors = compare_provenance_identity(run_meta, candidate_meta)

    compat = verify_pixvqa_commit_compatibility(
        candidate_meta.get("pixvqa_commit"),
        run_meta.get("pixvqa_commit"),
        repo_root=ROOT,
    )
    commit_compatibility: dict[str, Any] = compat.to_report_dict()
    if compat.commits_match:
        commit_compatibility["compatibility_exception_accepted"] = False
    elif compat.verified and compat.acceptable:
        commit_compatibility["compatibility_exception_accepted"] = True
        commit_compatibility["note"] = (
            "pixvqa_commit differs; git diff confined to auditable numerical-conformance "
            "infrastructure allowlist."
        )
    else:
        commit_compatibility["compatibility_exception_accepted"] = False
        if not compat.verified:
            provenance_errors.extend(compat.errors)
        elif compat.disallowed_files:
            provenance_errors.append(
                "pixvqa_commit mismatch with changes outside infrastructure allowlist: "
                + ", ".join(compat.disallowed_files)
            )
        elif not compat.acceptable:
            provenance_errors.append("pixvqa_commit mismatch could not be accepted")

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
        commit_compatibility=commit_compatibility,
        note=note,
    )
