"""Auditable Git commit compatibility for unapproved candidate reproducibility checks."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ROOT

POLICY_ID = "numerical_conformance_infrastructure_allowlist_v1"

# Explicit allowlist only — no broad prefixes (e.g. not all of scripts/).
CANDIDATE_COMPARE_INFRASTRUCTURE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "scripts/numerical_conformance/README.md",
        "scripts/numerical_conformance/baselines.py",
        "scripts/numerical_conformance/candidate_compare.py",
        "scripts/numerical_conformance/candidate_commit_compatibility.py",
        "scripts/numerical_conformance/runner.py",
        "scripts/test_metric_numerical_conformance.py",
        "scripts/test_numerical_baselines.py",
        "scripts/test_numerical_candidate_compare.py",
        "scripts/test_numerical_candidate_commit_compatibility.py",
    }
)


def _normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _git_rev_parse(ref: str, *, repo_root: Path) -> tuple[str | None, str | None]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"git rev-parse failed for {ref!r}: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return None, f"cannot resolve pixvqa commit {ref!r}: {err or 'unknown git error'}"
    return proc.stdout.strip(), None


def _git_diff_name_only(base: str, head: str, *, repo_root: Path) -> tuple[list[str] | None, str | None]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", base, head],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"git diff failed between {base} and {head}: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return None, f"git diff failed between {base} and {head}: {err or 'unknown git error'}"
    files = [_normalize_repo_path(line) for line in proc.stdout.splitlines() if line.strip()]
    return files, None


@dataclass
class CommitCompatibilityResult:
    verified: bool
    acceptable: bool
    commits_match: bool
    candidate_commit: str
    run_commit: str
    resolved_candidate: str | None = None
    resolved_run: str | None = None
    changed_files: list[str] = field(default_factory=list)
    disallowed_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "policy": POLICY_ID,
            "commits_match": self.commits_match,
            "candidate_pixvqa_commit": self.candidate_commit,
            "run_pixvqa_commit": self.run_commit,
            "resolved_candidate": self.resolved_candidate,
            "resolved_run": self.resolved_run,
            "verified": self.verified,
            "acceptable": self.acceptable,
            "changed_files": self.changed_files,
            "disallowed_files": self.disallowed_files,
            "errors": self.errors,
        }


def verify_pixvqa_commit_compatibility(
    candidate_commit: str | None,
    run_commit: str | None,
    *,
    repo_root: Path | None = None,
) -> CommitCompatibilityResult:
    repo = repo_root or ROOT
    cand = (candidate_commit or "").strip()
    run = (run_commit or "").strip()
    if not cand or not run or cand == "unknown" or run == "unknown":
        return CommitCompatibilityResult(
            verified=False,
            acceptable=False,
            commits_match=False,
            candidate_commit=cand,
            run_commit=run,
            errors=["pixvqa_commit missing or unknown; cannot verify compatibility"],
        )

    if cand == run:
        return CommitCompatibilityResult(
            verified=True,
            acceptable=True,
            commits_match=True,
            candidate_commit=cand,
            run_commit=run,
            resolved_candidate=cand,
            resolved_run=run,
        )

    resolved_cand, err_c = _git_rev_parse(cand, repo_root=repo)
    if err_c:
        return CommitCompatibilityResult(
            verified=False,
            acceptable=False,
            commits_match=False,
            candidate_commit=cand,
            run_commit=run,
            errors=[err_c],
        )
    resolved_run, err_r = _git_rev_parse(run, repo_root=repo)
    if err_r:
        return CommitCompatibilityResult(
            verified=False,
            acceptable=False,
            commits_match=False,
            candidate_commit=cand,
            run_commit=run,
            resolved_candidate=resolved_cand,
            errors=[err_r],
        )

    if resolved_cand == resolved_run:
        return CommitCompatibilityResult(
            verified=True,
            acceptable=True,
            commits_match=True,
            candidate_commit=cand,
            run_commit=run,
            resolved_candidate=resolved_cand,
            resolved_run=resolved_run,
        )

    changed, err_d = _git_diff_name_only(resolved_cand, resolved_run, repo_root=repo)
    if err_d or changed is None:
        return CommitCompatibilityResult(
            verified=False,
            acceptable=False,
            commits_match=False,
            candidate_commit=cand,
            run_commit=run,
            resolved_candidate=resolved_cand,
            resolved_run=resolved_run,
            errors=[err_d or "git diff produced no file list"],
        )

    disallowed = sorted(f for f in changed if f not in CANDIDATE_COMPARE_INFRASTRUCTURE_ALLOWLIST)
    acceptable = len(disallowed) == 0
    return CommitCompatibilityResult(
        verified=True,
        acceptable=acceptable,
        commits_match=False,
        candidate_commit=cand,
        run_commit=run,
        resolved_candidate=resolved_cand,
        resolved_run=resolved_run,
        changed_files=sorted(changed),
        disallowed_files=disallowed,
    )
