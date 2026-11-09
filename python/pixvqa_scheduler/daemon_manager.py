"""Spawn and attach to the cross-process admission daemon."""
from __future__ import annotations

import fcntl
import json
import os
import select
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Client
from subprocess import PIPE, Popen

from .ipc import authkey, lock_path, socket_address

_DAEMON_MODULE = "pixvqa_scheduler.daemon"
_PROBE_TIMEOUT_S = 1.0
_POLL_S = 0.2
_STARTUP_TIMEOUT_S = float(os.environ.get("PIXVQA_SCHEDULER_DAEMON_STARTUP_S", "30"))

_MANAGER_LOCK = threading.Lock()
_MANAGER: "_DaemonManager | None" = None


def _probe(address: str, key: bytes) -> bool:
    if not os.path.exists(address):
        return False
    try:
        with Client(address, family="AF_UNIX", authkey=key) as conn:
            conn.send({"cmd": "ping", "id": "probe"})
            if not conn.poll(_PROBE_TIMEOUT_S):
                return False
            resp = conn.recv()
            return isinstance(resp, dict) and resp.get("status") == "pong"
    except (OSError, EOFError, ValueError):
        return False


class _DaemonManager:
    def __init__(self) -> None:
        self._address = socket_address()
        self._authkey = authkey()
        self._lock_path = lock_path()
        self._proc: Popen[bytes] | None = None
        self._lock_file = None
        self._owns_daemon = False

    def _spawn(self, startup_timeout: float) -> None:
        env = os.environ.copy()
        py_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = py_root + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        self._proc = Popen(
            [sys.executable, "-u", "-m", _DAEMON_MODULE],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            env=env,
        )
        assert self._proc.stdout is not None
        deadline = time.monotonic() + startup_timeout
        buf = ""
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() if self._proc.stderr else b"").decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError(f"admission daemon exited during startup: {err[-1200:]}")
            ready, _, _ = select.select([self._proc.stdout.fileno()], [], [], _POLL_S)
            if not ready:
                continue
            chunk = os.read(self._proc.stdout.fileno(), 65536)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            if "\n" in buf:
                line = buf.split("\n", 1)[0]
                payload = json.loads(line)
                if payload.get("status") != "ready":
                    raise RuntimeError(f"admission daemon startup rejected: {payload}")
                return
        raise RuntimeError(f"admission daemon startup timed out after {startup_timeout}s")

    def ensure(self, startup_timeout: float = _STARTUP_TIMEOUT_S) -> tuple[str, bytes]:
        if _probe(self._address, self._authkey):
            return self._address, self._authkey
        self._lock_file = open(self._lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
            if _probe(self._address, self._authkey):
                return self._address, self._authkey
            if os.path.exists(self._address):
                try:
                    os.unlink(self._address)
                except OSError:
                    pass
            self._spawn(startup_timeout)
            self._owns_daemon = True
            return self._address, self._authkey
        finally:
            if self._lock_file is not None:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)


def ensure_admission_daemon(
    *,
    startup_timeout: float = _STARTUP_TIMEOUT_S,
) -> tuple[str, bytes]:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = _DaemonManager()
        return _MANAGER.ensure(startup_timeout)


def admission_daemon_available() -> bool:
    """True when AF_UNIX + flock are available (Linux/WSL)."""
    try:
        import fcntl  # noqa: F401
    except ImportError:
        return False
    return hasattr(__import__("socket"), "AF_UNIX")
