"""Host RAM and device-global CUDA VRAM telemetry for metric scheduling."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")

MEM_AVAILABLE_FLOOR_MIB = 2048
GPU_HEADROOM_MIB = 512
SWAP_HEAVY_MIB = 512


def mem_available_mib() -> float:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 0.0


def swap_used_mib() -> float:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        total = free = 0
        for line in fh:
            if line.startswith("SwapTotal:"):
                total = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                free = int(line.split()[1])
    return (total - free) / 1024 if total else 0.0


def gpu_memory_mib(nvidia_smi: Path | None = None) -> tuple[int, int]:
    """Return (used_mib, total_mib) for GPU 0.

    Values come from ``nvidia-smi`` and reflect **device-global** memory use
  (all processes on the GPU), not PID-exclusive accounting.
    """
    exe = nvidia_smi or NVIDIA_SMI
    try:
        out = subprocess.check_output(
            [str(exe), "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        used, total = (int(x) for x in out.split(", "))
        return used, total
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError, OSError):
        return -1, -1


def check_host_resources(
    *,
    mem_floor_mib: float = MEM_AVAILABLE_FLOOR_MIB,
    gpu_headroom_mib: int = GPU_HEADROOM_MIB,
    swap_limit_mib: float = SWAP_HEAVY_MIB,
    nvidia_smi: Path | None = None,
) -> str | None:
    """Return abort reason when a guard trips, else None."""
    avail = mem_available_mib()
    if avail < mem_floor_mib:
        return f"MemAvailable {avail:.0f} MiB below floor {mem_floor_mib:.0f} MiB"
    swap = swap_used_mib()
    if swap > swap_limit_mib:
        return f"swap usage {swap:.0f} MiB exceeds threshold {swap_limit_mib:.0f} MiB"
    gpu_used, gpu_total = gpu_memory_mib(nvidia_smi)
    if gpu_total > 0 and gpu_used >= gpu_total - gpu_headroom_mib:
        return (
            f"GPU memory {gpu_used} MiB near total {gpu_total} MiB "
            f"(headroom < {gpu_headroom_mib} MiB)"
        )
    return None


@dataclass
class TelemetrySample:
    t_monotonic: float
    mem_available_mib: float
    swap_used_mib: float
    gpu_used_mib: int
    gpu_total_mib: int


@dataclass
class HostTelemetry:
    """Background sampler; GPU figures are device-global, not per-process."""

    poll_interval_s: float = 0.25
    nvidia_smi: Path | None = None
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    samples: list[TelemetrySample] = field(default_factory=list)
    min_mem_available_mib: float = field(default=0.0)
    max_swap_used_mib: float = 0.0
    max_gpu_used_mib: int = 0
    gpu_total_mib: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def sample_once(self) -> TelemetrySample:
        avail = mem_available_mib()
        swap = swap_used_mib()
        gpu_used, gpu_total = gpu_memory_mib(self.nvidia_smi)
        ts = TelemetrySample(time.monotonic(), avail, swap, gpu_used, gpu_total)
        with self._lock:
            self.samples.append(ts)
            self.min_mem_available_mib = min(self.min_mem_available_mib or avail, avail)
            self.max_swap_used_mib = max(self.max_swap_used_mib, swap)
            if gpu_used >= 0:
                self.max_gpu_used_mib = max(self.max_gpu_used_mib, gpu_used)
            if gpu_total > 0:
                self.gpu_total_mib = gpu_total
        return ts

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.poll_interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self.min_mem_available_mib = mem_available_mib()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pixvqa-sched-telemetry")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None

    def should_abort(
        self,
        *,
        mem_floor_mib: float = MEM_AVAILABLE_FLOOR_MIB,
        gpu_headroom_mib: int = GPU_HEADROOM_MIB,
        swap_limit_mib: float = SWAP_HEAVY_MIB,
    ) -> str | None:
        return check_host_resources(
            mem_floor_mib=mem_floor_mib,
            gpu_headroom_mib=gpu_headroom_mib,
            swap_limit_mib=swap_limit_mib,
            nvidia_smi=self.nvidia_smi,
        )

    def summary(self) -> dict:
        with self._lock:
            return {
                "samples": len(self.samples),
                "min_mem_available_mib": round(self.min_mem_available_mib, 1),
                "max_swap_used_mib": round(self.max_swap_used_mib, 1),
                "max_gpu_used_mib": self.max_gpu_used_mib,
                "gpu_total_mib": self.gpu_total_mib,
                "note": "GPU memory samples are device-global (nvidia-smi), not PID-exclusive.",
            }
