"""Versioned baseline storage (never auto-overwrite approved baselines)."""
from __future__ import annotations

import json
from pathlib import Path

from .config import BASELINE_ROOT


def baseline_path(profile: str) -> Path:
    return BASELINE_ROOT / f"{profile}.json"


def load_baseline(profile: str) -> dict:
    path = baseline_path(profile)
    if not path.is_file():
        return {"version": 1, "approved": False, "cases": {}}
    return json.loads(path.read_text(encoding="utf-8"))


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


def compare_to_baseline(row: dict[str, Any], baseline: dict) -> dict[str, Any] | None:
    if not baseline.get("approved"):
        return None
    expected = baseline.get("cases", {}).get(row["id"])
    if not expected or row.get("status") != "PASS":
        return None
    for key in ("psnr_avg", "cvvdp_jod"):
        if key not in expected or row.get(key) is None:
            continue
        if not close(key, float(expected[key]), float(row[key])):
            return _row_from_row(row, "FAIL", f"baseline regression {key}: {row[key]} vs {expected[key]}")
    if expected.get("status") and expected["status"] != row["status"]:
        return _row_from_row(row, "FAIL", f"baseline status {expected['status']} vs {row['status']}")
    return None


def _row_from_row(row: dict[str, Any], status: str, detail: str) -> dict[str, Any]:
    out = dict(row)
    out["status"] = status
    out["detail"] = detail
    return out


def approve_candidate(profile: str) -> Path:
    cand = baseline_path(profile).with_suffix(".candidate.json")
    if not cand.is_file():
        raise FileNotFoundError(cand)
    data = json.loads(cand.read_text(encoding="utf-8"))
    data["approved"] = True
    dest = baseline_path(profile)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dest
