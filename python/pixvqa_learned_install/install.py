"""Orchestrate learned-metric repository and asset installation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from pixvqa_learned_install.config_sync import merge_config
from pixvqa_learned_install.download import ensure_asset
from pixvqa_learned_install.registry import (
    ASSETS,
    REPOS,
    SIMPLEVQA_GDRIVE_ID,
    default_config_paths,
)
from pixvqa_learned_install.repos import RepoError, ensure_all_repos
from pixvqa_learned_install.validate import validate_readiness


@dataclass
class InstallReport:
    messages: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False


def _log(report: InstallReport, msg: str) -> None:
    report.messages.append(msg)
    print(msg, flush=True)


def install_assets(root: Path, *, offline: bool = False) -> InstallReport:
    report = InstallReport()
    try:
        repo_status = ensure_all_repos(root, offline=offline)
    except RepoError as exc:
        report.blockers.append(str(exc))
        _log(report, f"STOP: {exc}")
        return report

    for key, status in repo_status.items():
        _log(report, f"upstream {key}: {status}")

    # CVQA-HDR NR weights beside repository checkout.
    nr_repo = root / REPOS["cvqa-hdr-nr"].default_dir
    for asset_key in ("cvqa_nr_checkpoint", "cvqa_nr_profile"):
        spec = ASSETS[asset_key]
        path = nr_repo / spec.relative_path
        if path.is_file():
            st, _ = ensure_asset(
                path,
                expected_sha256=spec.sha256,
                min_bytes=spec.min_bytes,
                kind=spec.kind,
                offline=True,
                label=spec.name,
            )
            if st == "hash_mismatch":
                report.blockers.append(
                    f"{spec.name} at {path} has unexpected SHA-256; remove it to re-download "
                    "or point config to your verified copy."
                )
                _log(report, f"STOP: {spec.name} hash mismatch at {path}")
            else:
                _log(report, f"reusing {spec.name}: {path}")
            continue
        st, changed = ensure_asset(
            path,
            expected_sha256=spec.sha256,
            url=spec.url,
            min_bytes=spec.min_bytes,
            kind=spec.kind,
            offline=offline,
            label=spec.name,
        )
        if st == "ok":
            _log(report, f"{'installed' if changed else 'verified'} {spec.name}: {path}")
        elif st == "missing":
            report.blockers.append(
                f"{spec.name} missing at {path}; offline={offline}. "
                f"Download from upstream README: {REPOS['cvqa-hdr-nr'].url}"
            )
        elif st == "download_failed":
            report.blockers.append(
                f"{spec.name}: download failed or payload rejected (HTML/invalid). "
                "Place verified NR_HDR_VQA.pth/.npy under deps/CompressedVQA-HDR manually."
            )
        elif st == "hash_mismatch":
            report.blockers.append(f"{spec.name}: downloaded bytes failed SHA-256 pin.")

    # SimpleVQA checkpoint — Google Drive only when pin-verifiable.
    sv_repo = root / REPOS["simplevqa"].default_dir
    sv_spec = ASSETS["simplevqa_checkpoint"]
    sv_path = sv_repo / sv_spec.relative_path
    if sv_path.is_file():
        st, _ = ensure_asset(
            sv_path,
            expected_sha256=sv_spec.sha256,
            min_bytes=sv_spec.min_bytes,
            kind=sv_spec.kind,
            offline=True,
            label=sv_spec.name,
        )
        if st == "hash_mismatch":
            report.blockers.append(
                f"SimpleVQA checkpoint at {sv_path} failed SHA-256 verification; "
                "installer will not treat it as the official UGC_BVQA_model.pth."
            )
        else:
            _log(report, f"reusing verified SimpleVQA checkpoint: {sv_path}")
    elif not offline:
        st, changed = ensure_asset(
            sv_path,
            expected_sha256=sv_spec.sha256,
            gdrive_id=SIMPLEVQA_GDRIVE_ID,
            min_bytes=sv_spec.min_bytes,
            kind=sv_spec.kind,
            offline=False,
            label=sv_spec.name,
        )
        if st == "ok":
            _log(report, f"{'installed' if changed else 'verified'} SimpleVQA checkpoint: {sv_path}")
        else:
            report.blockers.append(
                "SimpleVQA checkpoint not installed: Google Drive download failed or could not "
                f"be verified (id {SIMPLEVQA_GDRIVE_ID}). Manual download required — see "
                "docs/dev/SIMPLEVQA_REFERENCE.md."
            )
    else:
        report.blockers.append(
            "SimpleVQA checkpoint missing (offline mode); Google Drive asset not fetched."
        )

    # HDR-ChipQA models ship in the Git repo; verify only.
    hdr_repo = root / REPOS["hdr-chipqa"].default_dir
    for asset_key in ("hdr_chipqa_scaler", "hdr_chipqa_svr"):
        spec = ASSETS[asset_key]
        path = hdr_repo / spec.relative_path
        st, _ = ensure_asset(
            path,
            expected_sha256=spec.sha256,
            min_bytes=spec.min_bytes,
            kind=spec.kind,
            offline=True,
            label=spec.name,
        )
        if st != "ok":
            report.blockers.append(f"{spec.name} missing or corrupt under {hdr_repo}")
        else:
            _log(report, f"verified {spec.name} in upstream checkout")

    report.ready = not report.blockers
    return report


def sync_config(
    root: Path,
    learned_python: str,
    feature_python: str,
    inference_python: str,
) -> list[str]:
    example = root / "config" / "learned_metrics.example.toml"
    destination = root / "config" / "learned_metrics.local.toml"
    defaults = default_config_paths(root, learned_python, feature_python, inference_python)
    _, messages = merge_config(example, destination, defaults)
    return messages


def run_install(
    root: Path,
    *,
    learned_python: str,
    feature_python: str,
    inference_python: str,
    offline: bool = False,
    skip_load: bool = False,
) -> int:
    report = install_assets(root, offline=offline)
    for msg in sync_config(root, learned_python, feature_python, inference_python):
        print(f"Learned setup: {msg}", flush=True)

    ready, issues = validate_readiness(
        root,
        learned_python=learned_python,
        feature_python=feature_python,
        inference_python=inference_python,
        skip_load=skip_load or offline,
    )
    for issue in issues:
        print(f"Learned setup: note: {issue}", flush=True)

    if report.blockers:
        print("\nLearned setup: blocked items:", flush=True)
        for item in report.blockers:
            print(f"  - {item}", flush=True)

    if ready and not report.blockers:
        print("\nLearned setup: all pinned assets verified and readiness checks passed.", flush=True)
        return 0
    if report.blockers:
        print(
            "\nLearned setup: installation incomplete — fix blockers above before running learned metrics.",
            flush=True,
        )
        return 1
    print("\nLearned setup: partial install — some optional assets need manual steps.", flush=True)
    return 1
