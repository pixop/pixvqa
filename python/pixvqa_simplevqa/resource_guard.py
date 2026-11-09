"""Host RAM / GPU VRAM guards for concurrent SimpleVQA jobs."""
from __future__ import annotations

import subprocess
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
    """Return (used_mib, total_mib) for GPU 0, or (-1, -1) when unavailable."""
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


def check_resources(
    *,
    mem_floor_mib: float = MEM_AVAILABLE_FLOOR_MIB,
    gpu_headroom_mib: int = GPU_HEADROOM_MIB,
    swap_limit_mib: float = SWAP_HEAVY_MIB,
) -> str | None:
    """Return abort reason string when a guard trips, else None."""
    avail = mem_available_mib()
    if avail < mem_floor_mib:
        return f"MemAvailable {avail:.0f} MiB below floor {mem_floor_mib:.0f} MiB"
    swap = swap_used_mib()
    if swap > swap_limit_mib:
        return f"swap usage {swap:.0f} MiB exceeds threshold {swap_limit_mib:.0f} MiB"
    gpu_used, gpu_total = gpu_memory_mib()
    if gpu_total > 0 and gpu_used >= gpu_total - gpu_headroom_mib:
        return (
            f"GPU memory {gpu_used} MiB near total {gpu_total} MiB "
            f"(headroom < {gpu_headroom_mib} MiB)"
        )
    return None
