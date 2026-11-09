"""Client for cross-process metric admission leases."""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from multiprocessing.connection import Client

from .daemon_manager import admission_daemon_available, ensure_admission_daemon

_POLL_S = 0.15


@dataclass
class AdmissionLease:
    token: str
    metric: str
    wait_seconds: float
    address: str


class AdmissionError(RuntimeError):
    pass


def _connect() -> tuple[Client, str]:
    address, key = ensure_admission_daemon()
    try:
        return Client(address, family="AF_UNIX", authkey=key), address
    except OSError as exc:
        raise AdmissionError(f"could not connect to admission daemon: {exc}") from exc


def acquire(
    metric: str,
    *,
    shared_gpu: bool = False,
    timeout_s: float | None = None,
    pid: int | None = None,
) -> AdmissionLease:
    """Block until a global lease is granted or raise AdmissionError."""
    if not admission_daemon_available():
        raise AdmissionError("admission daemon unavailable on this platform")
    deadline = time.monotonic() + timeout_s if timeout_s and timeout_s > 0 else None
    req_id = uuid.uuid4().hex[:12]
    pid = pid if pid is not None else os.getpid()
    t0 = time.monotonic()
    conn, address = _connect()
    try:
        conn.send({
            "cmd": "acquire",
            "id": req_id,
            "metric": metric,
            "shared_gpu": shared_gpu,
            "pid": pid,
            "block": True,
        })
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise AdmissionError(f"admission wait timed out after {timeout_s}s")
            if not conn.poll(min(_POLL_S, deadline - time.monotonic() if deadline else _POLL_S)):
                continue
            resp = conn.recv()
            if not isinstance(resp, dict):
                raise AdmissionError("daemon sent non-dict response")
            if resp.get("id") not in (req_id, ""):
                continue
            status = resp.get("status")
            if status == "granted":
                return AdmissionLease(
                    token=str(resp["token"]),
                    metric=metric,
                    wait_seconds=float(resp.get("wait_seconds", time.monotonic() - t0)),
                    address=address,
                )
            if status in ("rejected", "timeout", "error"):
                raise AdmissionError(resp.get("reason") or resp.get("error") or status)
            if status == "queued":
                continue
            if status == "denied":
                time.sleep(_POLL_S)
                conn.send({
                    "cmd": "acquire",
                    "id": req_id,
                    "metric": metric,
                    "shared_gpu": shared_gpu,
                    "pid": pid,
                    "block": True,
                })
    finally:
        try:
            conn.close()
        except OSError:
            pass


def release(lease: AdmissionLease | str) -> None:
    token = lease.token if isinstance(lease, AdmissionLease) else lease
    conn, _ = _connect()
    try:
        conn.send({"cmd": "release", "id": "release", "token": token})
        if conn.poll(5.0):
            conn.recv()
    finally:
        try:
            conn.close()
        except OSError:
            pass


def heartbeat(lease: AdmissionLease) -> bool:
    conn, _ = _connect()
    try:
        conn.send({"cmd": "heartbeat", "id": "hb", "token": lease.token})
        if not conn.poll(2.0):
            return False
        resp = conn.recv()
        return isinstance(resp, dict) and resp.get("status") == "ok"
    except OSError:
        return False
    finally:
        try:
            conn.close()
        except OSError:
            pass
