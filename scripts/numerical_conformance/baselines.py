"""Versioned baseline storage (never auto-overwrite approved baselines)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BASELINE_ROOT
from .metric_fields import allowed_baseline_case_keys, required_fields_for_case
from .provenance import cvvdp_has_package_identity
from .tolerances import within_tolerance


def baseline_path(profile: str) -> Path:
    return BASELINE_ROOT / f"{profile}.json"


def load_baseline(profile: str) -> dict:
    path = baseline_path(profile)
    if not path.is_file():
        return {"version": 1, "approved": False, "cases": {}}
    return json.loads(path.read_text(encoding="utf-8"))


PROVENANCE_REQUIRED_KEYS = (
    "pixvqa_commit",
    "pixvqa_version",
    "tolerance_profile",
    "tolerances",
    "fixtures",
    "generated_at",
    "learned_provenance",
)

PROVENANCE_OPTIONAL_KEYS = (
    "learned_python",
    "learned_config",
    "fixture_manifest_sha256",
    "device_note",
    "python_version",
    "encoding_note",
)


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def validate_provenance(meta: dict | None) -> list[str]:
    if not meta:
        return ["baseline meta missing"]
    errors: list[str] = []
    for key in PROVENANCE_REQUIRED_KEYS:
        if key not in meta:
            errors.append(f"baseline meta missing {key}")
    tol = meta.get("tolerances")
    if tol is not None and not isinstance(tol, dict):
        errors.append("baseline meta tolerances must be an object")

    learned = meta.get("learned_provenance")
    if not isinstance(learned, dict):
        errors.append("baseline meta learned_provenance must be an object")
        return errors

    if meta.get("learned_config"):
        cvqa = learned.get("cvqa_hdr_nr")
        if not isinstance(cvqa, dict):
            errors.append("learned_provenance.cvqa_hdr_nr missing")
        else:
            for key in ("checkpoint_sha256", "profile_sha256"):
                if not _is_sha256_hex(cvqa.get(key)):
                    errors.append(
                        f"learned_provenance.cvqa_hdr_nr.{key} missing or invalid "
                        "(regenerate candidate with readable checkpoint/profile paths)"
                    )
        cvvdp = learned.get("cvvdp")
        if not isinstance(cvvdp, dict):
            errors.append("learned_provenance.cvvdp missing")
        elif not cvvdp_has_package_identity(cvvdp):
            errors.append("learned_provenance.cvvdp has no package version identity")

    return errors


def save_candidate(profile: str, cases: dict, meta: dict) -> Path:
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    out = baseline_path(profile)
    payload = {
        "version": 1,
        "approved": False,
        "meta": meta,
        "cases": cases,
    }
    cand = out.with_suffix(".candidate.json")
    cand.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cand


def _row_from_row(row: dict[str, Any], status: str, detail: str, **extra: Any) -> dict[str, Any]:
    out = dict(row)
    out["status"] = status
    out["detail"] = detail
    out.update(extra)
    return out


def compare_to_baseline(row: dict[str, Any], baseline: dict) -> dict[str, Any] | None:
    """Return an updated row on regression failure, or None if baseline check passes/skipped."""
    if not baseline.get("approved"):
        mark_regression_not_evaluated(row)
        return None

    prov_errors = validate_provenance(baseline.get("meta"))
    if prov_errors:
        return _row_from_row(
            row,
            "FAIL",
            "baseline provenance invalid: " + "; ".join(prov_errors),
            numerical_regression="FAIL",
        )

    if row.get("status") != "PASS":
        row["numerical_regression"] = "SKIPPED"
        row["numerical_regression_reason"] = f"execution status {row.get('status')}"
        return None

    case_id = row["id"]
    cases = baseline.get("cases", {})
    if case_id not in cases:
        return _row_from_row(
            row,
            "FAIL",
            f"missing baseline case {case_id}",
            numerical_regression="FAIL",
        )

    expected = cases[case_id]
    metric = row.get("metric", expected.get("metric", ""))
    family = row.get("family", expected.get("family", "sdr"))
    allowed = allowed_baseline_case_keys(metric, family)
    extra = set(expected.keys()) - allowed
    if extra:
        return _row_from_row(
            row,
            "FAIL",
            f"baseline has unexpected fields for {case_id}: {sorted(extra)}",
            numerical_regression="FAIL",
        )

    if expected.get("status") and expected["status"] != "PASS":
        return _row_from_row(
            row,
            "FAIL",
            f"baseline status {expected['status']} vs execution PASS",
            numerical_regression="FAIL",
        )

    required = required_fields_for_case(metric, family)
    comparisons: list[dict[str, Any]] = []

    for field in required:
        if field not in expected:
            return _row_from_row(
                row,
                "FAIL",
                f"baseline missing required field {field} for {case_id}",
                numerical_regression="FAIL",
            )
        observed = row.get(field)
        if observed is None:
            return _row_from_row(
                row,
                "FAIL",
                f"missing observed {field} for baseline comparison",
                numerical_regression="FAIL",
            )
        try:
            obs_f = float(observed)
            exp_f = float(expected[field])
        except (TypeError, ValueError):
            return _row_from_row(
                row,
                "FAIL",
                f"non-numeric baseline values for {field}",
                numerical_regression="FAIL",
            )
        ok, detail = within_tolerance(field, obs_f, exp_f)
        comparisons.append(detail)
        if not ok:
            return _row_from_row(
                row,
                "FAIL",
                (
                    f"baseline regression {field}: observed={obs_f} expected={exp_f} "
                    f"abs_error={detail.get('abs_error')} abs_tol={detail.get('abs_tol')}"
                ),
                numerical_regression="FAIL",
                baseline_comparisons=comparisons,
            )

    row["numerical_regression"] = "PASS"
    row["baseline_comparisons"] = comparisons
    return None


def mark_regression_not_evaluated(row: dict[str, Any]) -> None:
    row.setdefault("numerical_regression", "NOT_EVALUATED")
    row.setdefault("numerical_regression_reason", "no approved baseline")


def audit_mandatory_baseline_coverage(
    results: list[dict[str, Any]],
    baseline: dict,
) -> list[str]:
    if not baseline.get("approved"):
        return []
    cases = baseline.get("cases", {})
    missing: list[str] = []
    for row in results:
        if not row.get("mandatory"):
            continue
        if row.get("status") != "PASS":
            continue
        if row["id"] not in cases:
            missing.append(row["id"])
    return missing


def validate_baseline_cases(cases: dict) -> list[str]:
    errors: list[str] = []
    for case_id, entry in cases.items():
        if not isinstance(entry, dict):
            errors.append(f"case {case_id} is not an object")
            continue
        metric = entry.get("metric")
        family = entry.get("family")
        if not metric or not family:
            errors.append(f"case {case_id} missing metric/family metadata")
            continue
        extra = set(entry.keys()) - allowed_baseline_case_keys(metric, family)
        if extra:
            errors.append(f"case {case_id} unexpected fields {sorted(extra)}")
    return errors


def approve_candidate(profile: str) -> Path:
    cand = baseline_path(profile).with_suffix(".candidate.json")
    if not cand.is_file():
        raise FileNotFoundError(cand)
    data = json.loads(cand.read_text(encoding="utf-8"))
    errors = validate_provenance(data.get("meta"))
    errors.extend(validate_baseline_cases(data.get("cases", {})))
    if errors:
        raise ValueError("candidate provenance incomplete: " + "; ".join(errors))
    data["approved"] = True
    dest = baseline_path(profile)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dest
