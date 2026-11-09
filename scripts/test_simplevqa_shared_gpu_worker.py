#!/usr/bin/env python3
"""Mock tests for shared SimpleVQA GPU daemon (multiprocessing.connection IPC)."""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from multiprocessing.connection import Client, Listener
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pixvqa_simplevqa.adapter import InferenceError  # noqa: E402
from pixvqa_simplevqa import shared_gpu_server as svr_mod  # noqa: E402
from pixvqa_simplevqa.shared_gpu_client import run_via_shared_gpu_worker  # noqa: E402
from pixvqa_simplevqa.shared_gpu_server import (  # noqa: E402
    _SharedGpuDaemonManager,
    shutdown_shared_gpu_daemons,
)
from pixvqa_simplevqa.shared_gpu_ipc import authkey  # noqa: E402


def _minimal_repo(tmp: Path) -> tuple[Path, Path]:
    repo = tmp / "repo"
    repo.mkdir()
    (repo / "test_demo.py").write_text("# mock\npass\n", encoding="utf-8")
    (repo / "model").mkdir()
    (repo / "model" / "UGC_BVQA_model.py").write_text("# mock\n", encoding="utf-8")
    ckpt = tmp / "ckpt.pth"
    ckpt.write_bytes(b"fake")
    return repo, ckpt


def _stub_overlay(src: Path) -> str:
    return "# staged\n"


def _mock_listener(address: str, key: bytes) -> threading.Thread:
    expected = 3.8700499534606934

    def run() -> None:
        with Listener(address, family="AF_UNIX", authkey=key) as listener:
            while True:
                try:
                    conn = listener.accept()
                except OSError:
                    break
                with conn:
                    msg = conn.recv()
                    cmd = msg.get("cmd") if isinstance(msg, dict) else None
                    rid = msg.get("id", "") if isinstance(msg, dict) else ""
                    if cmd == "ping":
                        conn.send({"status": "pong", "id": rid})
                    elif cmd == "score":
                        conn.send(
                            {
                                "status": "ok",
                                "id": rid,
                                "score": expected,
                                "video": msg.get("video", ""),
                                "model_load_count": 1,
                            }
                        )
                    elif cmd == "shutdown":
                        conn.send({"status": "bye", "id": rid})
                        break
                    else:
                        conn.send({"status": "error", "id": rid, "error": "bad cmd"})

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.05)
    return t


def test_path_score_via_mock_daemon():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        repo, ckpt = _minimal_repo(tmp)
        video = tmp / "clip.mp4"
        video.write_bytes(b"v")
        address = str(tmp / "mock.sock")
        key = authkey(repo, ckpt)
        _mock_listener(address, key)
        shutdown_shared_gpu_daemons()

        def fake_spawn(self, startup_timeout):
            self._proc = type("P", (), {"poll": lambda _s: None, "stderr": None})()

        with patch.object(svr_mod, "cuda_shared_gpu_overlay_test_demo", _stub_overlay):
            with patch.object(svr_mod, "_probe", lambda _a, _k: True):
                with patch.object(_SharedGpuDaemonManager, "_spawn", fake_spawn):
                    with patch.object(svr_mod, "socket_address", lambda r, c: address):
                        with patch.object(svr_mod, "authkey", lambda r, c: key):
                            score, meta = run_via_shared_gpu_worker(repo, ckpt, video, timeout=30)
                            assert abs(score - 3.8700499534606934) < 1e-4
                            assert meta.get("model_load_count") == 1
                            assert meta.get("pipelined") is False
        shutdown_shared_gpu_daemons()


def test_probe_and_ping():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        repo, ckpt = _minimal_repo(tmp)
        address = str(tmp / "ping.sock")
        key = authkey(repo, ckpt)
        _mock_listener(address, key)
        assert svr_mod._probe(address, key)
        with Client(address, family="AF_UNIX", authkey=key) as conn:
            conn.send({"cmd": "ping", "id": "t"})
            assert conn.poll(2)
            assert conn.recv().get("status") == "pong"


def test_resource_guard_floor():
    from pixvqa_simplevqa.resource_guard import check_resources, mem_available_mib

    assert check_resources(mem_floor_mib=mem_available_mib() + 1_000_000) is not None


if __name__ == "__main__":
    tests = [test_path_score_via_mock_daemon, test_probe_and_ping, test_resource_guard_floor]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)
    print(f"PASS: {len(tests)} shared GPU worker mock tests")
