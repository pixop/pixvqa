"""AF_UNIX paths for the cross-process metric admission daemon."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

_AUTHKEY = b"pixvqa-scheduler-admission-v1"

# Linux AF_UNIX sun_path is 108 bytes including the terminating NUL.
_AF_UNIX_PATH_MAX = 107
_SOCKET_NAME = "admission.sock"
_FALLBACK_PREFIX = "pixvqa-adm"


class SocketPathError(RuntimeError):
    """Raised when no AF_UNIX socket path can be constructed."""


def _socket_path_len(path: Path) -> int:
    return len(str(path / _SOCKET_NAME))


def _short_runtime_fallback(*, digest: str | None = None) -> Path:
    """Per-user short path under the system temp dir (WSL/Linux portable)."""
    tag = digest or str(os.getuid())
    return Path(tempfile.gettempdir()) / f"{_FALLBACK_PREFIX}-{os.getuid()}-{tag}"


def _resolve_runtime_base() -> tuple[Path, str | None]:
    """Return (directory, optional fallback_reason)."""
    override = os.environ.get("PIXVQA_SCHEDULER_DIR", "").strip()
    if override:
        requested = Path(override).expanduser()
        if _socket_path_len(requested) <= _AF_UNIX_PATH_MAX:
            return requested, None
        digest = hashlib.sha256(requested.resolve().as_posix().encode()).hexdigest()[:12]
        fallback = _short_runtime_fallback(digest=digest)
        reason = (
            f"PIXVQA_SCHEDULER_DIR socket path exceeds AF_UNIX {_AF_UNIX_PATH_MAX}-byte "
            f"limit ({_socket_path_len(requested)} bytes); using {fallback}"
        )
        return fallback, reason

    run = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if run:
        path = Path(run) / "pixvqa"
        if _socket_path_len(path) <= _AF_UNIX_PATH_MAX:
            return path, None
        fallback = _short_runtime_fallback()
        reason = (
            f"XDG_RUNTIME_DIR/pixvqa socket path exceeds AF_UNIX limit; using {fallback}"
        )
        return fallback, reason

    return Path(tempfile.gettempdir()) / f"pixvqa-scheduler-{os.getuid()}", None


_FALLBACK_WARNED: str | None = None


def runtime_dir() -> Path:
    path, reason = _resolve_runtime_base()
    if reason:
        global _FALLBACK_WARNED
        if _FALLBACK_WARNED != reason:
            import sys

            print(f"pixvqa-scheduler: notice: {reason}", file=sys.stderr)
            _FALLBACK_WARNED = reason
    if _socket_path_len(path) > _AF_UNIX_PATH_MAX:
        raise SocketPathError(
            f"cannot place admission socket under {path}: "
            f"path length {_socket_path_len(path)} exceeds AF_UNIX limit "
            f"({_AF_UNIX_PATH_MAX} bytes). Set PIXVQA_SCHEDULER_DIR to a shorter "
            f"directory or PIXVQA_SCHEDULER_DAEMON_SOCKET to a short absolute path."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def socket_address() -> str:
    override = os.environ.get("PIXVQA_SCHEDULER_DAEMON_SOCKET", "").strip()
    if override:
        if len(override) > _AF_UNIX_PATH_MAX:
            raise SocketPathError(
                f"PIXVQA_SCHEDULER_DAEMON_SOCKET exceeds AF_UNIX limit "
                f"({len(override)} > {_AF_UNIX_PATH_MAX} bytes): {override!r}"
            )
        return override
    return str(runtime_dir() / _SOCKET_NAME)


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
