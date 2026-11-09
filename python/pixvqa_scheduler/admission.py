"""Opt-in global admission wrapper for metric backends and batch launcher."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Generator

from .client import AdmissionError, AdmissionLease, acquire, release

__all__ = ["AdmissionError", "AdmissionLease", "admission_enabled", "optional_admission"]
from .daemon_manager import admission_daemon_available


def admission_enabled() -> bool:
    return os.environ.get("PIXVQA_SCHEDULER_ADMISSION", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def admission_skip() -> bool:
    """Parent batch scheduler already holds a lease for this subprocess."""
    return os.environ.get("PIXVQA_SCHEDULER_ADMISSION_SKIP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _timeout_s() -> float | None:
    raw = os.environ.get("PIXVQA_SCHEDULER_ADMISSION_TIMEOUT_S", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@contextmanager
def optional_admission(
    metric: str,
    *,
    shared_gpu: bool = False,
) -> Generator[AdmissionLease | None, None, None]:
    """Acquire a cross-process lease when PIXVQA_SCHEDULER_ADMISSION=1."""
    if not admission_enabled() or admission_skip():
        yield None
        return
    if not admission_daemon_available():
        print(
            "pixvqa-scheduler: warning: PIXVQA_SCHEDULER_ADMISSION set but "
            "AF_UNIX/flock unavailable; proceeding without global admission",
            file=sys.stderr,
        )
        yield None
        return
    lease: AdmissionLease | None = None
    try:
        lease = acquire(metric, shared_gpu=shared_gpu, timeout_s=_timeout_s())
        yield lease
    except AdmissionError:
        raise
    finally:
        if lease is not None:
            try:
                release(lease)
            except OSError:
                pass
