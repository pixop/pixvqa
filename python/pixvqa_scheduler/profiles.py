"""Per-metric resource profiles and execution strategies."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

ExecutionStrategy = Literal[
    "independent",   # separate subprocess; each loads its own model (default for CVQA)
    "shared_gpu",    # one cross-process GPU model + serialized queue (SimpleVQA opt-in)
    "cpu_only",      # no GPU budget reservation
    "serialized",    # at most one in-flight job for this metric in the scheduler instance
]


@dataclass(frozen=True)
class MetricProfile:
    metric: str
    execution: ExecutionStrategy
    max_concurrent: int
    ram_reserve_mib: float
    """Conservative per-job host RAM reservation (scheduler accounting only)."""
    vram_reserve_mib: float
    """Per-job VRAM budget for admission. Device-global VRAM is polled separately."""
    module: str
    """``python -m`` module when launched by the scheduler helper."""
    verified_concurrency: str
    """What was actually measured/verified for this metric (audit text)."""

    def effective_max_concurrent(self) -> int:
        key = f"PIXVQA_SCHEDULER_MAX_CONCURRENT_{self.metric.upper().replace('-', '_')}"
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return self.max_concurrent


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Defaults are conservative and keyed to PR #22 measured evidence where available.
# CompressedVQA-HDR-NR: each ``python -m pixvqa_cvqahdr`` subprocess loads its own
# model in-process (see adapter ``run()``). Five concurrent pixvqa/python invocations
# have been reported to complete, but simultaneous independent CUDA model residency
# was NOT instrumented — do not assume 5× full VRAM without local verification.
_CVQA_HDR_NR = MetricProfile(
    metric="cvqa-hdr-nr",
    execution="independent",
    max_concurrent=_int_env("PIXVQA_SCHEDULER_MAX_CONCURRENT_CVQA_HDR_NR", 5),
    ram_reserve_mib=_float_env("PIXVQA_SCHEDULER_CVQA_RAM_MIB", 2500.0),
    vram_reserve_mib=_float_env("PIXVQA_SCHEDULER_CVQA_VRAM_MIB", 2800.0),
    module="pixvqa_cvqahdr",
    verified_concurrency=(
        "Each job is a separate Python process (fork/exec from pixvqa C++ or direct "
        "``python -m pixvqa_cvqahdr``). Model load occurs inside that process "
        "(``NR_model.SigLIP2_384_multi_dataset()``). Concurrent calls observed to "
        "succeed in user testing; independent CUDA instances not verified."
    ),
)

_SIMPLEVQA_INDEPENDENT = MetricProfile(
    metric="simplevqa",
    execution="independent",
    max_concurrent=_int_env("PIXVQA_SCHEDULER_MAX_CONCURRENT_SIMPLEVQA", 2),
    ram_reserve_mib=_float_env("PIXVQA_SCHEDULER_SIMPLEVQA_RAM_MIB", 1800.0),
    vram_reserve_mib=_float_env("PIXVQA_SCHEDULER_SIMPLEVQA_VRAM_MIB", 4000.0),
    module="pixvqa_simplevqa",
    verified_concurrency=(
        "Streaming-default FP32: 2 concurrent 60s BasketballDrive jobs succeeded "
        "(identical scores). 5 concurrent independent CUDA jobs on RTX 5080 WSL "
        "tripped GPU headroom abort (~15.9 GiB device-global VRAM vs 16 GiB total)."
    ),
)

_SIMPLEVQA_SHARED = MetricProfile(
    metric="simplevqa",
    execution="shared_gpu",
    max_concurrent=_int_env("PIXVQA_SCHEDULER_MAX_CONCURRENT_SIMPLEVQA_SHARED", 8),
    ram_reserve_mib=_float_env("PIXVQA_SCHEDULER_SIMPLEVQA_SHARED_RAM_MIB", 400.0),
    vram_reserve_mib=_float_env("PIXVQA_SCHEDULER_SIMPLEVQA_SHARED_VRAM_MIB", 4200.0),
    module="pixvqa_simplevqa",
    verified_concurrency=(
        "``--use-shared-gpu-worker``: one daemon loads one CUDA model; clients "
        "serialize on a GPU lock. Five concurrent clients scored 3.8700499534606934 "
        "in 152.6 s; device-global VRAM peak 3950 MiB (not PID-exclusive)."
    ),
)

_CVVDP = MetricProfile(
    metric="cvvdp",
    execution="independent",
    max_concurrent=_int_env("PIXVQA_SCHEDULER_MAX_CONCURRENT_CVVDP", 2),
    ram_reserve_mib=3000.0,
    vram_reserve_mib=6000.0,
    module="pixvqa_cvvdp",
    verified_concurrency="Not re-measured on PR #22 branch; conservative caps.",
)

_HDR_CHIPQA = MetricProfile(
    metric="hdr-chipqa",
    execution="independent",
    max_concurrent=_int_env("PIXVQA_SCHEDULER_MAX_CONCURRENT_HDR_CHIPQA", 2),
    ram_reserve_mib=2000.0,
    vram_reserve_mib=4000.0,
    module="pixvqa_hdr_chipqa",
    verified_concurrency="Not re-measured on PR #22 branch; conservative caps.",
)


def default_profiles() -> dict[str, MetricProfile]:
    return {
        "cvqa-hdr-nr": _CVQA_HDR_NR,
        "simplevqa": _SIMPLEVQA_INDEPENDENT,
        "simplevqa:shared_gpu": _SIMPLEVQA_SHARED,
        "cvvdp": _CVVDP,
        "hdr-chipqa": _HDR_CHIPQA,
    }


def profile_for(metric: str, *, shared_gpu: bool = False) -> MetricProfile:
    profiles = default_profiles()
    if metric == "simplevqa" and shared_gpu:
        return profiles["simplevqa:shared_gpu"]
    key = metric.lower()
    if key not in profiles:
        raise KeyError(f"no scheduler profile for metric {metric!r}")
    return profiles[key]


@dataclass
class SchedulerBudget:
    """Global host budgets applied across metrics in one scheduler instance."""

    mem_available_floor_mib: float = field(
        default_factory=lambda: _float_env("PIXVQA_SCHEDULER_MEM_FLOOR_MIB", 2048.0)
    )
    gpu_headroom_mib: int = field(
        default_factory=lambda: _int_env("PIXVQA_SCHEDULER_GPU_HEADROOM_MIB", 512)
    )
    swap_limit_mib: float = field(
        default_factory=lambda: _float_env("PIXVQA_SCHEDULER_SWAP_LIMIT_MIB", 512.0)
    )
    max_queue_depth: int = field(
        default_factory=lambda: _int_env("PIXVQA_SCHEDULER_MAX_QUEUE", 64)
    )
    queue_timeout_s: float = field(
        default_factory=lambda: _float_env("PIXVQA_SCHEDULER_QUEUE_TIMEOUT_S", 3600.0)
    )
    max_total_concurrent: int = field(
        default_factory=lambda: _int_env("PIXVQA_SCHEDULER_MAX_TOTAL", 16)
    )
