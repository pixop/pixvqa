"""Compute-device selection shared by the PixVQA learned-metric adapters.

auto picks CUDA when this interpreter's PyTorch reports a CUDA device, else
CPU. cuda never falls back: an unavailable device is an error. MPS is not
selected; upstream CVVDP's own auto mode is never used.
"""

from __future__ import annotations

import platform
import sys

DEVICE_CHOICES = ("auto", "cpu", "cuda")


class DeviceError(Exception):
    pass


def resolve_device(requested: str, torch) -> str:
    if requested not in DEVICE_CHOICES:
        raise DeviceError(f"unsupported device {requested!r}; use auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise DeviceError(
            f"CUDA was requested but PyTorch {torch.__version__} (CUDA build {torch.version.cuda}) "
            "reports no CUDA device; not falling back to CPU"
        )
    return "cuda" if available else "cpu"


def auto_notice(requested: str, effective: str) -> str | None:
    if requested == "auto" and effective == "cpu":
        return "no CUDA device is visible to PyTorch; auto selected cpu"
    return None


def device_record(requested: str, effective: str, torch) -> dict:
    record = {
        "requested": requested, "effective": effective,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    if torch is None:
        return record
    record.update({
        "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads(),
    })
    if effective == "cuda":
        props = torch.cuda.get_device_properties(0)
        record["gpu"] = {
            "name": props.name, "capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory, "cudnn": torch.backends.cudnn.version(),
        }
    return record


def peak_rss_bytes(children: bool = False) -> int | None:
    """Peak resident set size of this process or its reaped children; None on Windows."""
    try:
        import resource
    except ImportError:
        return None
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(who).ru_maxrss * scale
