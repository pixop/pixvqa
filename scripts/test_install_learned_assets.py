#!/usr/bin/env python3
"""Offline tests for pixvqa_learned_install (mock payloads, no network)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "python"
sys.path.insert(0, str(PY))

from pixvqa_learned_install.download import (  # noqa: E402
    DownloadError,
    ensure_asset,
    looks_like_html,
    reject_html_payload,
    validate_torch_checkpoint,
)
from pixvqa_learned_install.config_sync import merge_config  # noqa: E402
from pixvqa_learned_install.install import install_assets  # noqa: E402
from pixvqa_learned_install.registry import (  # noqa: E402
    CVQA_NR_CHECKPOINT_SHA256,
    CVQA_NR_PROFILE_SHA256,
    REPOS,
    SIMPLEVQA_CHECKPOINT_SHA256,
)
from pixvqa_hdr_chipqa import SCALER_SHA256, SVR_SHA256  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL {msg}", file=sys.stderr)
    raise SystemExit(1)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _minimal_pth() -> bytes:
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    try:
        import torch  # type: ignore

        torch.save({"layer.weight": torch.zeros(1)}, buf.name)
        return Path(buf.name).read_bytes()
    except ImportError:
        # Zip-wrapped empty state dict compatible with validate_torch_checkpoint.
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        with zipfile.ZipFile(out.name, "w") as zf:
            zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
        return Path(out.name).read_bytes()
    finally:
        Path(buf.name).unlink(missing_ok=True)


def test_html_rejection() -> None:
    if not looks_like_html(b"<!DOCTYPE html><title>x</title>"):
        fail("HTML detector missed doctype")
    try:
        reject_html_payload(b"<html>not a model</html>", label="t")
        fail("HTML payload accepted")
    except DownloadError:
        pass


def test_config_merge_preserves_user() -> None:
    work = Path(tempfile.mkdtemp())
    try:
        example = ROOT / "config" / "learned_metrics.example.toml"
        dest = work / "learned_metrics.local.toml"
        custom = "/custom/nr.pth"
        dest.write_text(
            '[learned]\npython = "../.venv/bin/python"\n\n'
            f'[cvqa-hdr-nr]\ncheckpoint = "{custom}"\n',
            encoding="utf-8",
        )
        defaults = {
            "learned.python": "/abs/.venv/bin/python",
            "cvqa-hdr-nr.checkpoint": "/should/not/overwrite",
            "cvqa-hdr-nr.repository": "/fresh/nr/repo",
        }
        merge_config(example, dest, defaults)
        text = dest.read_text(encoding="utf-8")
        if custom not in text:
            fail("config merge overwrote user checkpoint path")
        if "/fresh/nr/repo" not in text:
            fail("config merge did not add missing repository path")
        if "/abs/.venv/bin/python" not in text:
            fail("config merge did not migrate stock learned.python placeholder")
    finally:
        shutil.rmtree(work)


def test_offline_install_tree() -> None:
    work = Path(tempfile.mkdtemp())
    try:
        nr = work / REPOS["cvqa-hdr-nr"].default_dir
        nr.mkdir(parents=True)
        bad = nr / "NR_HDR_VQA.pth"
        bad.write_bytes(_minimal_pth())
        st, _ = ensure_asset(
            bad,
            expected_sha256=CVQA_NR_CHECKPOINT_SHA256,
            offline=True,
            kind="pth",
        )
        if st != "hash_mismatch":
            fail(f"expected hash_mismatch for wrong NR checkpoint, got {st}")

        # Full-tree verify only when this workstation already has pinned checkouts.
        real_nr = Path("/home/sergio/models/compressedvqa-hdr/NR_HDR_VQA.pth")
        hdr_src = ROOT / "deps/hdr_chipqa"
        sv_src = ROOT / "deps/SimpleVQA"
        if not hdr_src.is_dir() or not real_nr.is_file():
            return

        nr_tree = work / "deps"
        nr_tree.mkdir(parents=True, exist_ok=True)
        shutil.copytree(hdr_src, work / REPOS["hdr-chipqa"].default_dir, dirs_exist_ok=True)
        shutil.copytree(sv_src, work / REPOS["simplevqa"].default_dir, dirs_exist_ok=True)
        nr = work / REPOS["cvqa-hdr-nr"].default_dir
        nr.mkdir(parents=True, exist_ok=True)
        (nr / "NR").mkdir(exist_ok=True)
        (nr / "NR/NR_model.py").write_text("# stub\n", encoding="utf-8")
        shutil.copy(real_nr, nr / "NR_HDR_VQA.pth")
        shutil.copy(Path("/home/sergio/models/compressedvqa-hdr/NR_HDR_VQA.npy"), nr / "NR_HDR_VQA.npy")
        subprocess.run(["git", "init", "-q", str(nr)], check=True)
        subprocess.run(["git", "-C", str(nr), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(nr), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "t"],
            check=True,
        )

        report = install_assets(work, offline=True)
        if any("hash" in b.lower() for b in report.blockers):
            fail(f"unexpected hash blockers: {report.blockers}")
    finally:
        shutil.rmtree(work)


def test_shell_syntax() -> None:
    scripts = [
        "scripts/install_learned_metrics.sh",
        "scripts/setup_hdr_chipqa_envs.sh",
        "scripts/ensure_chipqa_inference_python.sh",
    ]
    for rel in scripts:
        subprocess.run(["bash", "-n", ROOT / rel], check=True)


def main() -> None:
    test_html_rejection()
    test_config_merge_preserves_user()
    test_offline_install_tree()
    test_shell_syntax()
    print("OK   learned asset installer offline tests passed")


if __name__ == "__main__":
    main()
