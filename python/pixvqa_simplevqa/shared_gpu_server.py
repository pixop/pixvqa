"""Spawn and attach to the cross-process SimpleVQA shared GPU daemon."""
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
from pathlib import Path
from subprocess import PIPE, Popen

from .adapter import InferenceError, cuda_shared_gpu_overlay_test_demo
from .shared_gpu_ipc import authkey, lock_path, socket_address

_DAEMON_MODULE = "pixvqa_simplevqa.shared_gpu_daemon"
_STARTUP_TIMEOUT_S = 600.0
_POLL_S = 0.2
_PROBE_TIMEOUT_S = 1.0

_MANAGER_LOCK = threading.Lock()
_MANAGERS: dict[tuple[str, str], "_SharedGpuDaemonManager"] = {}


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


class _SharedGpuDaemonManager:
    def __init__(self, repo: Path, checkpoint: Path) -> None:
        self._repo = repo.expanduser().resolve()
        self._checkpoint = checkpoint.expanduser().resolve()
        self._address = socket_address(self._repo, self._checkpoint)
        self._authkey = authkey(self._repo, self._checkpoint)
        self._lock_path = lock_path(self._repo, self._checkpoint)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="pixvqa_svqa_shared_gpu_")
        self._root = Path(self._tmpdir.name)
        self._proc: Popen[bytes] | None = None
        self._lock_file = None
        self._owns_daemon = False
        self._setup_staging()

    def _setup_staging(self) -> None:
        upstream = self._repo / "test_demo.py"
        (self._root / "test_demo.py").write_text(
            cuda_shared_gpu_overlay_test_demo(upstream), encoding="utf-8"
        )
        (self._root / "model").symlink_to(self._repo / "model", target_is_directory=True)
        (self._root / "ckpts").mkdir()
        (self._root / "ckpts" / "UGC_BVQA_model.pth").symlink_to(self._checkpoint)

    def _spawn(self, startup_timeout: float) -> None:
        env = os.environ.copy()
        env["PIXVQA_SVQA_SHARED_GPU_ROOT"] = str(self._root)
        env["PIXVQA_SVQA_SHARED_GPU_SOCKET"] = self._address
        env["PIXVQA_SVQA_SHARED_REPO"] = str(self._repo)
        env["PIXVQA_SVQA_SHARED_CHECKPOINT"] = str(self._checkpoint)
        py_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = py_root + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        self._proc = Popen(
            [sys.executable, "-u", "-m", _DAEMON_MODULE],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            cwd=str(self._root),
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
                raise InferenceError(
                    f"shared GPU daemon exited during startup: {err[-1200:]}"
                )
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
                    raise InferenceError(f"shared GPU daemon startup rejected: {payload}")
                return
        raise InferenceError(f"shared GPU daemon startup timed out after {startup_timeout}s")

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

    def close(self, *, stop_daemon: bool = False) -> None:
        if stop_daemon and self._owns_daemon and self._proc is not None:
            if self._proc.poll() is None:
                try:
                    with Client(
                        self._address, family="AF_UNIX", authkey=self._authkey
                    ) as conn:
                        conn.send({"cmd": "shutdown", "id": "manager"})
                        if conn.poll(2.0):
                            conn.recv()
                except (OSError, EOFError, ValueError):
                    pass
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None
        self._tmpdir.cleanup()


def ensure_shared_gpu_daemon(
    repo: Path,
    checkpoint: Path,
    *,
    startup_timeout: float = _STARTUP_TIMEOUT_S,
) -> tuple[str, bytes]:
    repo_r = repo.expanduser().resolve()
    ckpt_r = checkpoint.expanduser().resolve()
    key = (str(repo_r), str(ckpt_r))
    with _MANAGER_LOCK:
        mgr = _MANAGERS.get(key)
        if mgr is None:
            mgr = _SharedGpuDaemonManager(repo_r, ckpt_r)
            _MANAGERS[key] = mgr
        return mgr.ensure(startup_timeout)


def shutdown_shared_gpu_daemons() -> None:
    with _MANAGER_LOCK:
        for mgr in list(_MANAGERS.values()):
            mgr.close(stop_daemon=True)
        _MANAGERS.clear()
