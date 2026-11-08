"""Atomic downloads with integrity checks and HTML rejection."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import ssl
import urllib.error
import urllib.request
from pathlib import Path

_HTML_MARKERS = (b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML", b"<?xml")
_GDRIVE_CONFIRM_RE = re.compile(r"confirm=([0-9A-Za-z_]+)")


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def looks_like_html(head: bytes) -> bool:
    stripped = head.lstrip()
    return any(stripped.startswith(marker) for marker in _HTML_MARKERS)


def reject_html_payload(data: bytes, *, label: str) -> None:
    if len(data) < 64 and looks_like_html(data[:512]):
        raise DownloadError(
            f"{label}: response looks like HTML, not a model artifact "
            f"(size={len(data)}). Upstream may be blocked or the link may require a browser."
        )
    if looks_like_html(data[:256]):
        raise DownloadError(f"{label}: payload begins with HTML; refusing to install.")


def validate_torch_checkpoint(path: Path) -> None:
    """Reject empty/trivial files; full load happens in readiness validation."""
    size = path.stat().st_size
    if size < 1024:
        raise DownloadError(f"{path}: checkpoint too small ({size} bytes)")
    with path.open("rb") as handle:
        head = handle.read(64)
    reject_html_payload(head, label=str(path))
    if head[:2] not in (b"PK", b"\x80", b"\x83", b"\x85") and not head.startswith(b"\x80\x02"):
        # PyTorch saves are typically zip (PK) or pickle protocols.
        raise DownloadError(
            f"{path}: does not look like a PyTorch checkpoint (header={head[:8]!r})"
        )


def validate_npy(path: Path) -> None:
    with path.open("rb") as handle:
        magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise DownloadError(f"{path}: missing NumPy .npy magic header")
    if path.stat().st_size < 64:
        raise DownloadError(f"{path}: profile file too small")


def validate_joblib(path: Path) -> None:
    with path.open("rb") as handle:
        head = handle.read(32)
    reject_html_payload(head, label=str(path))
    if not (head.startswith(b"\x80") or head.startswith(b"PK")):
        raise DownloadError(f"{path}: does not look like a joblib/pickle artifact")


def _http_get(url: str, *, timeout: int = 300) -> bytes:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "pixvqa-learned-install/1.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise DownloadError(f"HTTP request failed for {url}: {exc}") from exc


def _gdrive_download(file_id: str) -> bytes:
    """Download a Google Drive file; fails on confirmation/HTML traps."""
    session_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    data = _http_get(session_url)
    if looks_like_html(data[:4096]):
        token_match = _GDRIVE_CONFIRM_RE.search(data[:8192].decode("utf-8", errors="replace"))
        if token_match:
            confirm = token_match.group(1)
            data = _http_get(f"{session_url}&confirm={confirm}")
        else:
            raise DownloadError(
                "SimpleVQA checkpoint: Google Drive returned an HTML interstitial "
                f"(file id {file_id}). Download manually and place the verified file at "
                "deps/SimpleVQA/ckpts/UGC_BVQA_model.pth — installer will not claim success."
            )
    reject_html_payload(data[:4096], label="SimpleVQA Google Drive checkpoint")
    return data


def atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        part.write_bytes(data)
        os.replace(part, dest)
    finally:
        if part.exists() and not dest.exists():
            part.unlink(missing_ok=True)


def fetch_url_to_file(url: str, dest: Path, *, label: str) -> None:
    data = _http_get(url)
    reject_html_payload(data[:4096], label=label)
    atomic_write(dest, data)


def fetch_gdrive_to_file(file_id: str, dest: Path) -> None:
    """Acquire a Google Drive asset with gdown's cookie/confirmation handling.

    The caller checks SHA-256 before treating the result as installed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    py = os.environ.get("PIXVQA_LEARNED_DOWNLOADER_PYTHON", "")
    if not py:
        raise DownloadError("Google Drive downloader interpreter not configured")
    try:
        proc = subprocess.run(
            [py, "-m", "gdown", "--id", file_id, "-O", str(temp)],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "unknown gdown error").strip()
            raise DownloadError(f"gdown failed (exit {proc.returncode}): {msg[-600:]}")
        if not temp.is_file():
            raise DownloadError("gdown did not create a checkpoint file")
        with temp.open("rb") as handle:
            reject_html_payload(handle.read(4096), label="SimpleVQA Google Drive checkpoint")
        os.replace(temp, dest)
    except subprocess.TimeoutExpired as exc:
        raise DownloadError("Google Drive download timed out") from exc
    except OSError as exc:
        raise DownloadError(f"Google Drive downloader could not run: {exc}") from exc
    finally:
        temp.unlink(missing_ok=True)


def ensure_asset(
    path: Path,
    *,
    expected_sha256: str,
    url: str | None = None,
    gdrive_id: str | None = None,
    min_bytes: int = 1,
    kind: str = "other",
    offline: bool = False,
    label: str = "",
) -> tuple[str, bool]:
    """Return (status, changed). status is ok | missing | hash_mismatch | download_failed."""
    label = label or str(path)
    if path.is_file():
        actual = sha256_file(path)
        if actual == expected_sha256:
            return "ok", False
        return "hash_mismatch", False
    if offline:
        return "missing", False
    if gdrive_id:
        try:
            fetch_gdrive_to_file(gdrive_id, path)
        except DownloadError:
            return "download_failed", False
    elif url:
        try:
            fetch_url_to_file(url, path, label=label)
        except DownloadError:
            return "download_failed", False
    else:
        return "missing", False
    if path.stat().st_size < min_bytes:
        path.unlink(missing_ok=True)
        return "download_failed", False
    actual = sha256_file(path)
    if actual != expected_sha256:
        path.unlink(missing_ok=True)
        return "hash_mismatch", False
    try:
        if kind == "pth":
            validate_torch_checkpoint(path)
        elif kind == "npy":
            validate_npy(path)
        elif kind == "joblib":
            validate_joblib(path)
    except DownloadError:
        path.unlink(missing_ok=True)
        return "download_failed", False
    return "ok", True
