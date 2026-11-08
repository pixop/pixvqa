"""Persistent SimpleVQA worker: amortizes PyTorch import across batch requests.

Opt-in via ``run(..., use_worker=True)`` or ``--use-worker`` CLI flag.
Default subprocess-per-video behavior is unchanged.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import select
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from subprocess import PIPE, Popen

from .adapter import (
    InferenceError,
    cuda_single_decode_overlay_test_demo,
    parse_score,
)

_CHILD_SCRIPT = Path(__file__).resolve().parent / "worker_process.py"
_SESSION_LOCK = threading.Lock()
_SESSION: _WorkerSession | None = None
_SESSION_KEY: tuple[str, str, str] | None = None

_STDERR_TAIL_MAX = 8000
_READ_POLL_S = 0.2


def _stderr_tail_text(tail: deque[str]) -> str:
    return "".join(tail)[-_STDERR_TAIL_MAX:]


class _WorkerSession:
    """One long-lived child that imports heavy deps once."""

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        *,
        device: str = "cuda",
        child_script: Path | None = None,
        startup_timeout: float = 600.0,
    ) -> None:
        self._repo = repo.expanduser().resolve()
        self._checkpoint = checkpoint.expanduser().resolve()
        self._device = device
        self._child_script = child_script or _CHILD_SCRIPT
        self._tmpdir = tempfile.TemporaryDirectory(prefix="pixvqa_svqa_worker_")
        self._root = Path(self._tmpdir.name)
        self._proc: Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._broken = False
        self._setup_staging()
        self._start(startup_timeout=startup_timeout)

    def _setup_staging(self) -> None:
        upstream = self._repo / "test_demo.py"
        (self._root / "test_demo.py").write_text(
            cuda_single_decode_overlay_test_demo(upstream), encoding="utf-8"
        )
        (self._root / "model").symlink_to(self._repo / "model", target_is_directory=True)
        (self._root / "ckpts").mkdir()
        (self._root / "ckpts" / "UGC_BVQA_model.pth").symlink_to(self._checkpoint)

    def _start(self, *, startup_timeout: float) -> None:
        self._stop.clear()
        self._stdout_queue = queue.Queue()
        self._stderr_tail.clear()
        env = os.environ.copy()
        env["PIXVQA_SVQA_WORKER_ROOT"] = str(self._root)
        self._proc = Popen(
            [sys.executable, "-u", str(self._child_script)],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            cwd=str(self._root),
            env=env,
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._stdout_reader,
            name="svqa-worker-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_drainer,
            name="svqa-worker-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            ready_line = self._read_line_bounded(startup_timeout)
        except InferenceError:
            self._kill_child()
            raise
        if ready_line is None:
            tail = _stderr_tail_text(self._stderr_tail)
            self._kill_child()
            raise InferenceError(
                f"worker closed stdout before ready"
                + (f": {tail}" if tail else "")
            )
        try:
            payload = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            tail = _stderr_tail_text(self._stderr_tail)
            self._kill_child()
            raise InferenceError(f"worker failed to start: {tail}") from exc
        if payload.get("status") != "ready":
            self._kill_child()
            raise InferenceError(f"worker startup rejected: {payload}")
        self._broken = False

    def _stdout_reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._stdout_queue.put(None)
            return
        fd = proc.stdout.fileno()
        buf = ""
        try:
            while not self._stop.is_set():
                if proc.poll() is not None:
                    break
                try:
                    ready, _, _ = select.select([fd], [], [], _READ_POLL_S)
                except (ValueError, OSError):
                    break
                if not ready:
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    self._stdout_queue.put(line)
            if buf:
                self._stdout_queue.put(buf)
        finally:
            self._stdout_queue.put(None)

    def _stderr_drainer(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        fd = proc.stderr.fileno()
        try:
            while not self._stop.is_set():
                if proc.poll() is not None:
                    break
                try:
                    ready, _, _ = select.select([fd], [], [], _READ_POLL_S)
                except (ValueError, OSError):
                    break
                if not ready:
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                self._stderr_tail.append(text)
                joined = "".join(self._stderr_tail)
                if len(joined) > _STDERR_TAIL_MAX:
                    self._stderr_tail.clear()
                    self._stderr_tail.append(joined[-_STDERR_TAIL_MAX:])
        except OSError:
            pass

    def _read_line_bounded(self, timeout: float) -> str | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line = self._stdout_queue.get(timeout=min(remaining, _READ_POLL_S))
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    tail = _stderr_tail_text(self._stderr_tail)
                    raise InferenceError(
                        f"worker exited during startup: {tail or 'no stderr'}"
                    )
                continue
            if line is None:
                return None
            return line
        raise InferenceError(f"worker startup timed out after {timeout}s")

    def _wait_response(self, req_id: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line = self._stdout_queue.get(timeout=min(remaining, _READ_POLL_S))
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    tail = _stderr_tail_text(self._stderr_tail)
                    raise InferenceError(
                        f"worker exited unexpectedly: {tail or 'no stderr'}"
                    )
                continue
            if line is None:
                if self._proc is not None and self._proc.poll() is not None:
                    tail = _stderr_tail_text(self._stderr_tail)
                    raise InferenceError(
                        f"worker exited unexpectedly: {tail or 'no stderr'}"
                    )
                raise InferenceError("worker closed stdout without response")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("id") != req_id:
                continue
            return resp
        raise InferenceError(f"worker timed out after {timeout}s")

    def _ensure_running(self, startup_timeout: float) -> None:
        if self._broken or self._proc is None or self._proc.poll() is not None:
            self._kill_child()
            self._start(startup_timeout=startup_timeout)

    def score(self, video: Path, *, timeout: float, startup_timeout: float | None = None) -> float:
        startup_timeout = startup_timeout if startup_timeout is not None else timeout
        with self._lock:
            self._ensure_running(startup_timeout)
            assert self._proc is not None and self._proc.stdin is not None
            video = video.expanduser().resolve()
            req_id = uuid.uuid4().hex
            out = self._root / f"result_{req_id}.txt"
            req = json.dumps(
                {"cmd": "score", "id": req_id, "video": str(video), "output": str(out)}
            )
            try:
                self._proc.stdin.write(req.encode("utf-8") + b"\n")
                self._proc.stdin.flush()
            except OSError as exc:
                self._broken = True
                self._kill_child()
                raise InferenceError(f"could not write to worker: {exc}") from exc
            try:
                resp = self._wait_response(req_id, timeout)
            except InferenceError:
                self._broken = True
                self._kill_child()
                raise
            if resp.get("status") == "ok":
                return parse_score(f"{video},{resp['score']}\n", video)
            self._broken = True
            self._kill_child()
            raise InferenceError(resp.get("error") or "worker scoring failed")

    def _kill_child(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            if proc.poll() is None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.write(b'{"cmd":"shutdown"}\n')
                        proc.stdin.flush()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
            self._proc = None
        for th in (self._stdout_thread, self._stderr_thread):
            if th is not None and th.is_alive():
                th.join(timeout=2)
        self._stdout_thread = None
        self._stderr_thread = None
        self._stop.clear()

    def close(self) -> None:
        with self._lock:
            self._kill_child()
            self._tmpdir.cleanup()


def run_via_worker(repo: Path, checkpoint: Path, video: Path, *, timeout: int = 3600) -> float:
    global _SESSION, _SESSION_KEY
    repo_r = repo.expanduser().resolve()
    ckpt_r = checkpoint.expanduser().resolve()
    key = (str(repo_r), str(ckpt_r), "cuda")
    t = float(timeout)
    with _SESSION_LOCK:
        if _SESSION is None or _SESSION_KEY != key:
            if _SESSION is not None:
                _SESSION.close()
            try:
                _SESSION = _WorkerSession(repo_r, ckpt_r, startup_timeout=t)
                _SESSION_KEY = key
            except InferenceError:
                _SESSION = None
                _SESSION_KEY = None
                raise
        try:
            return _SESSION.score(video, timeout=t, startup_timeout=t)
        except InferenceError:
            raise


def shutdown_worker() -> None:
    global _SESSION, _SESSION_KEY
    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION.close()
        _SESSION = None
        _SESSION_KEY = None


atexit.register(shutdown_worker)
