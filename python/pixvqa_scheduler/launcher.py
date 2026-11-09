"""Subprocess launch helpers — no model loading in the scheduler."""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class RunningJob:
    job_id: str
    metric: str
    process: subprocess.Popen[str]
    started_monotonic: float

    def poll(self) -> int | None:
        return self.process.poll()


def launch(
    job_id: str,
    metric: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> RunningJob:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return RunningJob(job_id, metric, proc, time.monotonic())


def wait_job(
    job: RunningJob,
    *,
    timeout_s: float | None = None,
    on_tick: Callable[[], None] | None = None,
    tick_interval_s: float = 0.1,
) -> tuple[int, str, str]:
    deadline = time.monotonic() + timeout_s if timeout_s and timeout_s > 0 else None
    while True:
        try:
            stdout, stderr = job.process.communicate(timeout=tick_interval_s)
            return job.process.returncode or 0, stdout or "", stderr or ""
        except subprocess.TimeoutExpired:
            if on_tick:
                on_tick()
            if deadline is not None and time.monotonic() >= deadline:
                job.process.kill()
                stdout, stderr = job.process.communicate()
                return -9, stdout or "", stderr or ""
