"""Authkey and address helpers for shared SimpleVQA GPU daemon IPC."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

# Fixed-length authkey for multiprocessing.connection (local AF_UNIX only).
_AUTHKEY_PREFIX = b"pixvqa-simplevqa-shared-gpu-v1:"


def authkey(repo: Path, checkpoint: Path) -> bytes:
    repo_r = repo.expanduser().resolve()
    ckpt_r = checkpoint.expanduser().resolve()
    digest = hashlib.sha256(f"{repo_r}\0{ckpt_r}".encode()).hexdigest()[:32]
    key = _AUTHKEY_PREFIX + digest.encode("ascii")
    if len(key) > 256:
        key = key[:256]
    return key


def socket_address(repo: Path, checkpoint: Path) -> str:
    repo_r = repo.expanduser().resolve()
    ckpt_r = checkpoint.expanduser().resolve()
    key = hashlib.sha256(f"{repo_r}\0{ckpt_r}".encode()).hexdigest()[:16]
    base = Path(os.environ.get("PIXVQA_SIMPLEVQA_SHARED_GPU_DIR", tempfile.gettempdir()))
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"pixvqa_svqa_shared_gpu_{key}.sock")


def lock_path(repo: Path, checkpoint: Path) -> Path:
    repo_r = repo.expanduser().resolve()
    ckpt_r = checkpoint.expanduser().resolve()
    key = hashlib.sha256(f"{repo_r}\0{ckpt_r}".encode()).hexdigest()[:16]
    base = Path(os.environ.get("PIXVQA_SIMPLEVQA_SHARED_GPU_DIR", tempfile.gettempdir()))
    base.mkdir(parents=True, exist_ok=True)
    return base / f"pixvqa_svqa_shared_gpu_{key}.lock"
