"""Prepare PYTHONPATH for hdr_chipqa without vendoring upstream sources."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _chipqa_tmp(hdr_repo: Path) -> Path:
    return hdr_repo / "ChipQA" / "ChipQA_tmp"


def make_extract_env(hdr_repo: Path) -> tuple[dict[str, str], str | None, Path | None]:
    """Return env, optional cleanup token, and temp dir for symlink namespace.

    hdr_chipqa imports ``ChipQA.*``; upstream submodule only ships ``ChipQA_tmp``.
    A runtime symlink ``ChipQA -> ChipQA_tmp`` avoids copying unlicensed sources.
    """
    hdr_repo = hdr_repo.resolve()
    tmp_pkg = _chipqa_tmp(hdr_repo)
    if not tmp_pkg.is_dir():
        raise FileNotFoundError(f"HDR-ChipQA ChipQA_tmp missing: {tmp_pkg}")

    td = Path(tempfile.mkdtemp(prefix="pixvqa-hdr-chipqa-path-"))
    link = td / "ChipQA"
    try:
        link.symlink_to(tmp_pkg, target_is_directory=True)
    except OSError as exc:
        raise OSError(f"could not symlink ChipQA namespace at {link}: {exc}") from exc

    parts = [str(td), str(hdr_repo)]
    shim_parent = Path(__file__).resolve().parent.parent / "pixvqa_chipqa"
    parts.insert(0, str(shim_parent))
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(parts + ([prior] if prior else []))
    env["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

    # Inject numpy legacy aliases before upstream import in child process.
    runner = Path(__file__).resolve().parent / "_extract_runner.py"
    env["PIXVQA_HDR_CHIPQA_RUNNER"] = str(runner)
    return env, str(td), td


def child_runner_script() -> str:
    """Small bootstrap executed inside the feature Python subprocess."""
    return r'''
import os, runpy, sys, tempfile
from pathlib import Path
# numpy legacy aliases before hdr_chipqa imports
import pixvqa_chipqa.numpy_legacy_shim  # noqa: F401
hdr = Path(os.environ["PIXVQA_HDR_CHIPQA_REPO"]).resolve()
tmp_pkg = hdr / "ChipQA" / "ChipQA_tmp"
td = Path(tempfile.mkdtemp(prefix="pixvqa-hdr-chipqa-path-"))
link = td / "ChipQA"
link.symlink_to(tmp_pkg, target_is_directory=True)
for p in (str(td), str(hdr), str(Path(__file__).resolve().parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ChipQA.save_stats as _ss
if not hasattr(_ss, "extract_subband_feats"):
    _ss.extract_subband_feats = _ss._extract_subband_feats
os.chdir(hdr)
runpy.run_path(str(hdr / "hdr_chipqa.py"), run_name="__main__")
'''
