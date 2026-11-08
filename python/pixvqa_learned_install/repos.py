"""Git checkout helpers for learned-metric upstream sources."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pixvqa_learned_install.registry import REPOS, RepoSpec


class RepoError(RuntimeError):
    pass


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RepoError(f"{' '.join(cmd)} failed: {stderr or exc}") from exc


def head_commit(repo: Path) -> str:
    return _run(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()


def ensure_repo(
    spec: RepoSpec,
    dest: Path,
    *,
    offline: bool = False,
) -> tuple[str, bool]:
    """Ensure pinned checkout. Returns (status, cloned_or_reset)."""
    marker = dest / spec.marker
    if dest.exists() and not (dest / ".git").is_dir():
        raise RepoError(
            f"{dest} exists but is not a Git checkout; move it aside manually."
        )
    if not dest.exists():
        if offline:
            return "missing", False
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", spec.url, str(dest)])
        _run(["git", "-C", str(dest), "fetch", "origin", spec.commit])
        _run(["git", "-C", str(dest), "checkout", "--detach", spec.commit])
        if not marker.is_file():
            raise RepoError(f"{dest} checkout missing marker {spec.marker}")
        return "cloned", True
    actual = head_commit(dest)
    if actual != spec.commit:
        dirty = _run(["git", "-C", str(dest), "status", "--porcelain"]).stdout.strip()
        if dirty:
            raise RepoError(
                f"{dest} is at {actual[:12]} (expected {spec.commit[:12]}) with local changes; "
                "refusing to reset."
            )
        if offline:
            return "wrong_commit", False
        _run(["git", "-C", str(dest), "fetch", "origin", spec.commit])
        _run(["git", "-C", str(dest), "checkout", "--detach", spec.commit])
        actual = head_commit(dest)
    if actual != spec.commit:
        return "wrong_commit", False
    if not marker.is_file():
        raise RepoError(f"{dest} missing upstream marker {spec.marker}")
    return "ok", False


def ensure_all_repos(root: Path, *, offline: bool = False) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for key, spec in REPOS.items():
        dest = (root / spec.default_dir).resolve()
        status, _ = ensure_repo(spec, dest, offline=offline)
        statuses[key] = status
    return statuses
