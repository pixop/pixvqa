"""Build baseline candidate provenance metadata."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT
from .tolerances import catalog_export


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _fixture_digest(paths: dict[str, str]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for family, root in paths.items():
        p = Path(root)
        manifest = p / "manifest.json"
        if manifest.is_file():
            data = manifest.read_bytes()
            digests[family] = hashlib.sha256(data).hexdigest()
        else:
            digests[family] = "missing-manifest"
    return digests


def build_provenance(
    *,
    pixvqa_version: str,
    learned_python: str | None,
    learned_config: str | None,
    fixtures: dict[str, str],
    device_note: str | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pixvqa_commit": _git_head(),
        "pixvqa_version": pixvqa_version,
        "learned_python": learned_python,
        "learned_config": learned_config,
        "tolerance_profile": "same_backend_repro_v1",
        "tolerances": catalog_export(),
        "fixtures": fixtures,
        "fixture_manifest_sha256": _fixture_digest(fixtures),
        "device_note": device_note,
    }
