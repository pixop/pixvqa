"""Build argv for metric subprocesses (scheduler does not load models)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def simplevqa_command(
    *,
    python: str | None = None,
    repo: str | Path,
    checkpoint: str | Path,
    test: str | Path,
    device: str = "cuda",
    use_shared_gpu_worker: bool = False,
    output_json: str | Path | None = None,
) -> list[str]:
    py = python or sys.executable
    out = output_json or Path(tempfile.gettempdir()) / f"pixvqa_svqa_{os.getpid()}.json"
    cmd = [
        py, "-m", "pixvqa_simplevqa",
        "--repo", str(repo),
        "--checkpoint", str(checkpoint),
        "--test", str(test),
        "--device", device,
        "--output-json", str(out),
        "--transient-output",
    ]
    if use_shared_gpu_worker:
        cmd.append("--use-shared-gpu-worker")
    return cmd


def cvqa_hdr_nr_command(
    *,
    python: str | None = None,
    test: str | Path,
    repo: str | Path,
    checkpoint: str | Path,
    profile: str | Path,
    device: str = "cuda",
    output_json: str | Path | None = None,
) -> list[str]:
    py = python or sys.executable
    out = output_json or Path(tempfile.gettempdir()) / f"pixvqa_cvqa_{os.getpid()}.json"
    return [
        py, "-m", "pixvqa_cvqahdr",
        "--test", str(test),
        "--repo", str(repo),
        "--checkpoint", str(checkpoint),
        "--profile", str(profile),
        "--device", device,
        "--output-json", str(out),
        "--transient-output",
    ]


def pixvqa_command(
    *,
    pixvqa_bin: str | Path,
    metrics: list[str],
    test: str | Path,
    reference: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [str(pixvqa_bin), "--metrics", ",".join(metrics), "-t", str(test)]
    if reference:
        cmd.extend(["-r", str(reference)])
    if extra_args:
        cmd.extend(extra_args)
    return cmd
