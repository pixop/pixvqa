"""Per-user AF_UNIX admission daemon — cross-terminal metric resource leases."""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from multiprocessing.connection import Listener
from typing import Any

from .admission_ledger import AdmissionLedger
from .ipc import authkey, socket_address
from .profiles import MetricProfile

_LOG = logging.getLogger("pixvqa_scheduler.daemon")
_HEARTBEAT_TIMEOUT_S = float(os.environ.get("PIXVQA_SCHEDULER_LEASE_HEARTBEAT_S", "60"))
_CLEANUP_INTERVAL_S = 2.0
_SHUTDOWN = threading.Event()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class AdmissionDaemon:
    def __init__(self) -> None:
        self._ledger = AdmissionLedger()
        self._ledger.ensure_baselines()
        self._leases: dict[str, dict[str, Any]] = {}
        self._waiters: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

    def _lease_profile(self, lease: dict[str, Any]) -> MetricProfile:
        return self._ledger.resolve_profile(
            lease["metric"], shared_gpu=bool(lease.get("shared_gpu"))
        )

    def _shared_vram_key(self, profile: MetricProfile) -> str | None:
        if profile.execution == "shared_gpu":
            return f"{profile.metric}:{profile.execution}"
        return None

    def _expire_stale(self) -> None:
        now = time.monotonic()
        dead: list[str] = []
        for token, lease in self._leases.items():
            pid = int(lease.get("pid", 0))
            last = float(lease.get("last_heartbeat", lease.get("created", now)))
            if not _pid_alive(pid):
                dead.append(token)
            elif lease.get("heartbeats", 0) > 0 and now - last > _HEARTBEAT_TIMEOUT_S:
                dead.append(token)
        for token in dead:
            self._release_token(token, reason="stale lease cleanup")

    def _release_token(self, token: str, *, reason: str = "released") -> dict | None:
        lease = self._leases.pop(token, None)
        if lease is None:
            return None
        profile = self._lease_profile(lease)
        self._ledger.release(profile, leases=self._leases)
        return {"status": "released", "token": token, "reason": reason}

    def _grant_waiter(self, waiter: dict[str, Any]) -> dict:
        profile: MetricProfile = waiter.get("profile") or self._ledger.resolve_profile(
            waiter["metric"], shared_gpu=bool(waiter.get("shared_gpu"))
        )
        token = uuid.uuid4().hex[:16]
        lease = {
            "token": token,
            "metric": waiter["metric"],
            "shared_gpu": waiter.get("shared_gpu", False),
            "pid": waiter["pid"],
            "created": time.monotonic(),
            "last_heartbeat": time.monotonic(),
            "shared_vram_key": self._shared_vram_key(profile),
        }
        self._leases[token] = lease
        self._ledger.reserve(profile, leases=self._leases)
        return {
            "status": "granted",
            "token": token,
            "metric": waiter["metric"],
            "wait_seconds": time.monotonic() - waiter["enqueued_at"],
        }

    def _try_admit_queue(self) -> None:
        if not self._waiters:
            return
        kept: list[dict[str, Any]] = []
        for waiter in self._waiters:
            profile: MetricProfile = waiter["profile"]
            never = self._ledger.profile_never_admissible(profile, leases=self._leases)
            if never:
                conn = waiter.get("conn")
                if conn is not None:
                    try:
                        conn.send({"status": "rejected", "reason": never, "id": waiter.get("id")})
                    except (OSError, EOFError, ValueError):
                        pass
                continue
            reason = self._ledger.can_admit(profile, leases=self._leases)
            if reason:
                if time.monotonic() - waiter["enqueued_at"] > self._ledger.budget.queue_timeout_s:
                    conn = waiter.get("conn")
                    if conn is not None:
                        try:
                            conn.send({
                                "status": "timeout",
                                "reason": f"queue wait exceeded {self._ledger.budget.queue_timeout_s:.0f}s",
                                "id": waiter.get("id"),
                            })
                        except (OSError, EOFError, ValueError):
                            pass
                else:
                    kept.append(waiter)
                continue
            resp = self._grant_waiter(waiter)
            resp["id"] = waiter.get("id")
            conn = waiter.get("conn")
            if conn is not None:
                try:
                    conn.send(resp)
                except (OSError, EOFError, ValueError):
                    self._release_token(resp["token"], reason="client disconnected before run")
                    kept.append(waiter)
        self._waiters = kept

    def _handle(self, conn, msg: dict) -> dict:
        cmd = msg.get("cmd")
        req_id = str(msg.get("id", ""))
        if cmd == "ping":
            return {"status": "pong", "id": req_id}
        if cmd == "shutdown":
            _SHUTDOWN.set()
            return {"status": "bye", "id": req_id}
        if cmd == "status":
            with self._lock:
                snap = self._ledger.snapshot()
                snap["leases"] = len(self._leases)
                snap["queued"] = len(self._waiters)
            return {"status": "ok", "id": req_id, "snapshot": snap}
        if cmd == "acquire":
            metric = str(msg.get("metric", "")).lower()
            if not metric:
                return {"status": "error", "id": req_id, "error": "missing metric"}
            shared_gpu = bool(msg.get("shared_gpu"))
            pid = int(msg.get("pid") or os.getpid())
            profile = self._ledger.resolve_profile(metric, shared_gpu=shared_gpu)
            with self._lock:
                self._expire_stale()
                never = self._ledger.profile_never_admissible(profile, leases=self._leases)
                if never:
                    return {"status": "rejected", "id": req_id, "reason": never}
                reason = self._ledger.can_admit(profile, leases=self._leases)
                if reason is None:
                    resp = self._grant_waiter({
                        "metric": metric,
                        "shared_gpu": shared_gpu,
                        "pid": pid,
                        "profile": profile,
                        "enqueued_at": time.monotonic(),
                    })
                    resp["id"] = req_id
                    return resp
                if msg.get("block"):
                    self._waiters.append({
                        "metric": metric,
                        "shared_gpu": shared_gpu,
                        "pid": pid,
                        "profile": profile,
                        "enqueued_at": time.monotonic(),
                        "id": req_id,
                        "conn": conn,
                    })
                    self._cv.notify_all()
                    return {"status": "queued", "id": req_id, "reason": reason}
                return {"status": "denied", "id": req_id, "reason": reason, "retry": True}
        if cmd == "release":
            token = str(msg.get("token", ""))
            with self._lock:
                released = self._release_token(token)
                self._try_admit_queue()
                self._cv.notify_all()
            if released is None:
                return {"status": "error", "id": req_id, "error": f"unknown token {token!r}"}
            released["id"] = req_id
            return released
        if cmd == "heartbeat":
            token = str(msg.get("token", ""))
            with self._lock:
                lease = self._leases.get(token)
                if lease is None:
                    return {"status": "error", "id": req_id, "error": "unknown lease"}
                lease["last_heartbeat"] = time.monotonic()
                lease["heartbeats"] = int(lease.get("heartbeats", 0)) + 1
                if not _pid_alive(int(lease.get("pid", 0))):
                    self._release_token(token, reason="pid exited")
                    self._try_admit_queue()
                    return {"status": "released", "id": req_id, "token": token, "reason": "pid exited"}
            return {"status": "ok", "id": req_id, "token": token}
        return {"status": "error", "id": req_id, "error": f"unknown cmd {cmd!r}"}

    def _dispatcher_loop(self) -> None:
        while not _SHUTDOWN.is_set():
            with self._cv:
                self._expire_stale()
                self._try_admit_queue()
                self._cv.wait(timeout=_CLEANUP_INTERVAL_S)

    def serve(self, address: str | None = None, key: bytes | None = None) -> int:
        address = address or socket_address()
        key = key or authkey()
        if os.path.exists(address):
            try:
                os.unlink(address)
            except OSError:
                pass
        listener = Listener(address, family="AF_UNIX", authkey=key)
        sys.stdout.write(
            json.dumps({"status": "ready", "socket": address, "pid": os.getpid()}) + "\n"
        )
        sys.stdout.flush()
        dispatcher = threading.Thread(
            target=self._dispatcher_loop, daemon=True, name="pixvqa-admission-dispatch"
        )
        dispatcher.start()
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
                    target=self._serve_connection,
                    args=(conn,),
                    daemon=True,
                    name="pixvqa-admission-req",
                ).start()
        finally:
            listener.close()
            if os.path.exists(address):
                try:
                    os.unlink(address)
                except OSError:
                    pass
        return 0

    def _serve_connection(self, conn) -> None:
        try:
            while not _SHUTDOWN.is_set():
                try:
                    if not conn.poll(1.0):
                        continue
                    msg = conn.recv()
                except (EOFError, OSError, ValueError):
                    break
                if not isinstance(msg, dict):
                    conn.send({"status": "error", "error": "message must be a dict"})
                    continue
                if msg.get("cmd") == "acquire" and msg.get("block"):
                    with self._lock:
                        resp = self._handle(conn, msg)
                    if resp.get("status") == "queued":
                        continue
                    try:
                        conn.send(resp)
                    except (OSError, EOFError, ValueError):
                        if resp.get("status") == "granted":
                            self._release_token(str(resp["token"]), reason="client disconnected")
                        break
                else:
                    with self._lock:
                        resp = self._handle(conn, msg)
                        if resp.get("status") in ("granted", "released"):
                            self._try_admit_queue()
                            self._cv.notify_all()
                    try:
                        conn.send(resp)
                    except (OSError, EOFError, ValueError):
                        break
                if msg.get("cmd") == "shutdown":
                    break
        finally:
            try:
                conn.close()
            except OSError:
                pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, lambda *_: _SHUTDOWN.set())
    signal.signal(signal.SIGINT, lambda *_: _SHUTDOWN.set())
    return AdmissionDaemon().serve()


if __name__ == "__main__":
    raise SystemExit(main())
