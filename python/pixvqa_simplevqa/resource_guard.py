"""Host RAM / GPU VRAM guards for concurrent SimpleVQA jobs.

Re-exports shared telemetry helpers from ``pixvqa_scheduler`` for backward
compatibility with the shared-GPU daemon.
"""
from __future__ import annotations

from pixvqa_scheduler.telemetry import (
    GPU_HEADROOM_MIB,
    MEM_AVAILABLE_FLOOR_MIB,
    SWAP_HEAVY_MIB,
    check_host_resources,
    gpu_memory_mib,
    mem_available_mib,
    swap_used_mib,
)

check_resources = check_host_resources  # backward-compat alias for shared_gpu_daemon

__all__ = [
    "GPU_HEADROOM_MIB",
    "MEM_AVAILABLE_FLOOR_MIB",
    "SWAP_HEAVY_MIB",
    "check_host_resources",
    "check_resources",
    "gpu_memory_mib",
    "mem_available_mib",
    "swap_used_mib",
]
