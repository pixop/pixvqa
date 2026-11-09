"""Central resource-aware scheduler for PixVQA Python/PyTorch metric jobs.

Opt-in: set ``PIXVQA_SCHEDULER=1`` or construct ``MetricScheduler`` explicitly.
The scheduler does not load CUDA models; it only admits and launches subprocesses.
"""
from __future__ import annotations

from .admission import admission_enabled, optional_admission
from .client import AdmissionError, AdmissionLease, acquire, release
from .profiles import MetricProfile, default_profiles, profile_for
from .scheduler import (
    JobResult,
    JobSpec,
    MetricScheduler,
    SchedulerConfig,
    SchedulerReport,
    scheduler_enabled,
)
from .telemetry import HostTelemetry, TelemetrySample, check_host_resources

__all__ = [
    "AdmissionError",
    "AdmissionLease",
    "HostTelemetry",
    "acquire",
    "admission_enabled",
    "JobResult",
    "JobSpec",
    "MetricProfile",
    "MetricScheduler",
    "SchedulerConfig",
    "SchedulerReport",
    "TelemetrySample",
    "check_host_resources",
    "default_profiles",
    "optional_admission",
    "profile_for",
    "release",
    "scheduler_enabled",
]
