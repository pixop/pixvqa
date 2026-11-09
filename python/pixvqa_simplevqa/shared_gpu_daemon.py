"""Cross-process SimpleVQA GPU daemon — one loaded model, serialized path scoring.

Milestone prototype (not pipelined): each client sends a video path; the daemon
decodes and scores on GPU under a lock. CPU preprocessing is not concurrent across
clients — only GPU model weights are shared.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
from multiprocessing.connection import Listener
from pathlib import Path

import torch

from .resource_guard import check_resources
from .shared_gpu_ipc import authkey as ipc_authkey

_LOG = logging.getLogger("pixvqa_simplevqa.shared_gpu_daemon")
_GPU_LOCK = threading.Lock()
_SHUTDOWN = threading.Event()


def _load_staged_module(root: Path):
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    spec = importlib.util.spec_from_file_location("svqa_shared_gpu_demo", root / "test_demo.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load staged shared GPU test_demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["svqa_shared_gpu_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


def _handle(conn, mod, device: torch.device, stats: dict) -> None:
    try:
        msg = conn.recv()
    except (EOFError, OSError, ValueError) as exc:
        _LOG.warning("recv failed: %s", exc)
        return
    if not isinstance(msg, dict):
        conn.send({"status": "error", "error": "message must be a dict"})
        return
    cmd = msg.get("cmd")
    req_id = str(msg.get("id", ""))
    if cmd == "ping":
        conn.send({"status": "pong", "id": req_id})
        return
    if cmd == "shutdown":
        conn.send({"status": "bye", "id": req_id})
        _SHUTDOWN.set()
        return
    if cmd == "stats":
        conn.send(
            {
                "status": "ok",
                "id": req_id,
                "model_load_count": stats.get("model_load_count", 0),
                "jobs_completed": stats.get("jobs_completed", 0),
            }
        )
        return
    if cmd != "score":
        conn.send({"status": "error", "id": req_id, "error": f"unknown cmd {cmd!r}"})
        return
    video = str(msg.get("video", ""))
    if not video:
        conn.send({"status": "error", "id": req_id, "error": "missing video path"})
        return
    reason = check_resources()
    if reason:
        conn.send({"status": "error", "id": req_id, "error": reason})
        return
    with _GPU_LOCK:
        try:
            score, video_name = mod.score_video_path(video, device)
            stats["jobs_completed"] = stats.get("jobs_completed", 0) + 1
            conn.send(
                {
                    "status": "ok",
                    "id": req_id,
                    "score": float(score),
                    "video": video_name,
                    "model_load_count": stats.get("model_load_count", 1),
                }
            )
        except Exception as exc:
            _LOG.error("score failed for %s: %s", video, exc)
            conn.send({"status": "error", "id": req_id, "error": str(exc)})


def serve(address: str, root: Path, key: bytes) -> int:
    os.chdir(root)
    device = torch.device("cuda")
    mod = _load_staged_module(root)
    mod.load_models_once(device)
    stats = {"model_load_count": 1, "jobs_completed": 0}
    if os.path.exists(address):
        try:
            os.unlink(address)
        except OSError:
            pass
    listener = Listener(address, family="AF_UNIX", authkey=key)
    sys.stdout.write(
        json.dumps(
            {
                "status": "ready",
                "socket": address,
                "model_load_count": stats["model_load_count"],
            }
        )
        + "\n"
    )
    sys.stdout.flush()
    try:
        while not _SHUTDOWN.is_set():
            try:
                conn = listener.accept()
            except OSError as exc:
                if _SHUTDOWN.is_set():
                    break
                _LOG.warning("accept: %s", exc)
                continue
            threading.Thread(
                target=_serve_connection,
                args=(conn, mod, device, stats),
                daemon=True,
                name="svqa-shared-gpu-req",
            ).start()
    finally:
        listener.close()
        if os.path.exists(address):
            try:
                os.unlink(address)
            except OSError:
                pass
    return 0


def _serve_connection(conn, mod, device: torch.device, stats: dict) -> None:
    try:
        _handle(conn, mod, device, stats)
    finally:
        try:
            conn.close()
        except OSError:
            pass


def main() -> int:
    root = Path(os.environ["PIXVQA_SVQA_SHARED_GPU_ROOT"])
    address = os.environ["PIXVQA_SVQA_SHARED_GPU_SOCKET"]
    key = ipc_authkey(
        Path(os.environ["PIXVQA_SVQA_SHARED_REPO"]),
        Path(os.environ["PIXVQA_SVQA_SHARED_CHECKPOINT"]),
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return serve(address, root, key)


if __name__ == "__main__":
    raise SystemExit(main())
