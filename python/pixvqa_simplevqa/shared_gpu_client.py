"""Client for the shared SimpleVQA GPU path-scoring daemon."""
from __future__ import annotations

import time
import uuid
from multiprocessing.connection import Client
from pathlib import Path

from .adapter import InferenceError, parse_score
from .shared_gpu_server import ensure_shared_gpu_daemon

_POLL_INTERVAL_S = 0.05


def run_via_shared_gpu_worker(
    repo: Path,
    checkpoint: Path,
    video: Path,
    *,
    timeout: int = 3600,
) -> tuple[float, dict]:
    """Score via one shared CUDA model (daemon decodes + infers; not pipelined)."""
    t = float(timeout)
    video = video.expanduser().resolve()
    address, authkey = ensure_shared_gpu_daemon(repo, checkpoint, startup_timeout=t)
    req_id = uuid.uuid4().hex
    deadline = time.monotonic() + t
    meta: dict = {
        "mode": "shared_gpu_path_queue",
        "pipelined": False,
        "socket": address,
    }
    try:
        conn = Client(address, family="AF_UNIX", authkey=authkey)
    except OSError as exc:
        raise InferenceError(f"could not connect to shared GPU daemon: {exc}") from exc
    try:
        conn.send({"cmd": "score", "id": req_id, "video": str(video)})
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if not conn.poll(min(remaining, 1.0)):
                continue
            resp = conn.recv()
            if not isinstance(resp, dict):
                raise InferenceError("shared GPU daemon sent non-dict response")
            if resp.get("id") != req_id:
                continue
            if resp.get("status") != "ok":
                raise InferenceError(resp.get("error") or "shared GPU score failed")
            score = float(resp["score"])
            parse_score(f"{video},{score}\n", video)
            meta["model_load_count"] = resp.get("model_load_count", 1)
            return score, meta
        raise InferenceError(f"shared GPU daemon timed out after {timeout}s")
    finally:
        try:
            conn.close()
        except OSError:
            pass
