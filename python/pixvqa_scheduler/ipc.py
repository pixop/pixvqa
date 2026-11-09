"""AF_UNIX paths for the cross-process metric admission daemon."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

_AUTHKEY = b"pixvqa-scheduler-admission-v1"


def runtime_dir() -> Path:
    base = os.environ.get("PIXVQA_SCHEDULER_DIR")
    if base:
        path = Path(base).expanduser()
    else:
        run = os.environ.get("XDG_RUNTIME_DIR")
        if run:
            path = Path(run) / "pixvqa"
        else:
            path = Path(tempfile.gettempdir()) / f"pixvqa-scheduler-{os.getuid()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def socket_address() -> str:
    override = os.environ.get("PIXVQA_SCHEDULER_DAEMON_SOCKET", "").strip()
    if override:
        return override
    return str(runtime_dir() / "admission.sock")


def lock_path() -> Path:
    return runtime_dir() / "admission.lock"


def authkey() -> bytes:
    override = os.environ.get("PIXVQA_SCHEDULER_AUTHKEY", "").strip()
    if override:
        return override.encode("ascii")[:256]
    uid = os.getuid()
    host = hashlib.sha256(os.uname().nodename.encode()).hexdigest()[:16]
    key = _AUTHKEY + f":{uid}:{host}".encode()
    return key[:256]
