#!/usr/bin/env python3
"""Regression tests for pixvqa_commit compatibility policy (candidate compare)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.candidate_commit_compatibility import (  # noqa: E402
    CANDIDATE_COMPARE_INFRASTRUCTURE_ALLOWLIST,
    verify_pixvqa_commit_compatibility,
)


def test_commits_match_skips_diff() -> None:
    result = verify_pixvqa_commit_compatibility("abc", "abc")
    assert result.verified and result.acceptable and result.commits_match


def test_permitted_infrastructure_only() -> None:
    allowed = sorted(CANDIDATE_COMPARE_INFRASTRUCTURE_ALLOWLIST)[:2]
    with mock.patch(
        "numerical_conformance.candidate_commit_compatibility._git_rev_parse",
        side_effect=[("c1", None), ("c2", None)],
    ), mock.patch(
        "numerical_conformance.candidate_commit_compatibility._git_diff_name_only",
        return_value=(allowed, None),
    ):
        result = verify_pixvqa_commit_compatibility("cand", "run")
    assert result.verified and result.acceptable and not result.commits_match
    assert result.changed_files == allowed


def test_prohibited_metric_path() -> None:
    with mock.patch(
        "numerical_conformance.candidate_commit_compatibility._git_rev_parse",
        side_effect=[("c1", None), ("c2", None)],
    ), mock.patch(
        "numerical_conformance.candidate_commit_compatibility._git_diff_name_only",
        return_value=(["src/metrics/psnr.cpp"], None),
    ):
        result = verify_pixvqa_commit_compatibility("cand", "run")
    assert result.verified and not result.acceptable
    assert "src/metrics/psnr.cpp" in result.disallowed_files


def test_mixed_allowlist_and_metric_change() -> None:
    mixed = [
        "scripts/numerical_conformance/runner.py",
        "src/learned/cvqa_wrapper.py",
    ]
    with mock.patch(
        "numerical_conformance.candidate_commit_compatibility._git_rev_parse",
        side_effect=[("c1", None), ("c2", None)],
    ), mock.patch(
        "numerical_conformance.candidate_commit_compatibility._git_diff_name_only",
        return_value=(mixed, None),
    ):
        result = verify_pixvqa_commit_compatibility("cand", "run")
    assert result.verified and not result.acceptable
    assert result.disallowed_files == ["src/learned/cvqa_wrapper.py"]


def test_unresolved_candidate_commit() -> None:
    with mock.patch(
        "numerical_conformance.candidate_commit_compatibility._git_rev_parse",
        return_value=(None, "missing object"),
    ):
        result = verify_pixvqa_commit_compatibility("deadbeef", "run")
    assert not result.verified and not result.acceptable
    assert result.errors


def test_unknown_commit_fails_closed() -> None:
    result = verify_pixvqa_commit_compatibility("unknown", "abc123")
    assert not result.verified and not result.acceptable


def test_real_b0906e5_to_36ab703_allowlist_if_present() -> None:
    """When both commits exist locally, the known RTX diagnosis diff must be allowlisted."""
    cand = "b0906e5e303728dfd36d064ac068945ece863027"
    run = "36ab7031efa3a58216d52634ee04c02a7c4fe301"
    result = verify_pixvqa_commit_compatibility(cand, run)
    if not result.verified:
        return
    assert result.acceptable, result.disallowed_files


def main() -> int:
    tests = [
        test_commits_match_skips_diff,
        test_permitted_infrastructure_only,
        test_prohibited_metric_path,
        test_mixed_allowlist_and_metric_change,
        test_unresolved_candidate_commit,
        test_unknown_commit_fails_closed,
        test_real_b0906e5_to_36ab703_allowlist_if_present,
    ]
    for fn in tests:
        fn()
        print(f"OK   {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
