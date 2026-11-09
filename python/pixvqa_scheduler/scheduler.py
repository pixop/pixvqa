"""Resource-aware metric job scheduler (opt-in, subprocess-only)."""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .admission import admission_enabled
from .client import AdmissionError, acquire, release
from .launcher import RunningJob, launch, wait_job
from .profiles import MetricProfile, SchedulerBudget, default_profiles, profile_for
from .telemetry import HostTelemetry, check_host_resources


def scheduler_enabled() -> bool:
    return os.environ.get("PIXVQA_SCHEDULER", "").strip().lower() in ("1", "true", "yes")


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class JobSpec:
    metric: str
    command: list[str]
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    timeout_s: float | None = None
    shared_gpu: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    spec: JobSpec
    state: JobState
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    wall_seconds: float = 0.0
    admission_wait_seconds: float = 0.0
    error: str | None = None
    profile: MetricProfile | None = None


@dataclass
class SchedulerConfig:
    enabled: bool = field(default_factory=scheduler_enabled)
    budget: SchedulerBudget = field(default_factory=SchedulerBudget)
    profiles: dict[str, MetricProfile] = field(default_factory=default_profiles)
    telemetry_poll_s: float = 0.25
    stagger_start_s: float = field(
        default_factory=lambda: float(os.environ.get("PIXVQA_SCHEDULER_STAGGER_S", "0"))
    )
    cross_invocation_warning: str = (
        "Set PIXVQA_SCHEDULER_ADMISSION=1 (or pixvqa --scheduler-admission) for "
        "cross-terminal leases via the AF_UNIX admission daemon."
    )


@dataclass
class SchedulerReport:
    jobs: list[JobResult]
    telemetry: dict
    warnings: list[str]
    total_wall_seconds: float


class _QueuedJob:
    __slots__ = ("spec", "enqueued_at", "profile", "event", "result")

    def __init__(self, spec: JobSpec, profile: MetricProfile):
        self.spec = spec
        self.profile = profile
        self.enqueued_at = time.monotonic()
        self.event = threading.Event()
        self.result: JobResult | None = None


class MetricScheduler:
    """Admits metric subprocess jobs under per-metric and global resource caps."""

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()
        self._budget = self.config.budget
        self._profiles = dict(self.config.profiles)
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._queue: deque[_QueuedJob] = deque()
        self._active: dict[str, tuple[_QueuedJob, RunningJob, float]] = {}
        self._results: list[JobResult] = []
        self._metric_active: dict[str, int] = {}
        self._ram_reserved = 0.0
        self._vram_reserved = 0.0
        self._mem_baseline_mib: float | None = None
        self._gpu_baseline_mib: int | None = None
        self._telemetry = HostTelemetry(poll_interval_s=self.config.telemetry_poll_s)
        self._dispatcher: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._started = False
        self._abort_reason: str | None = None
        self._global_admission = admission_enabled()
        self._job_leases: dict[str, object] = {}

    def _resolve_profile(self, spec: JobSpec) -> MetricProfile:
        if spec.metric == "simplevqa" and spec.shared_gpu:
            return profile_for("simplevqa", shared_gpu=True)
        key = spec.metric.lower()
        if key in self._profiles:
            return self._profiles[key]
        return profile_for(spec.metric, shared_gpu=spec.shared_gpu)

    def _shared_vram_key(self, profile: MetricProfile) -> str | None:
        if profile.execution == "shared_gpu":
            return f"{profile.metric}:{profile.execution}"
        return None

    def _shared_slot_occupied(self, profile: MetricProfile) -> bool:
        key = self._shared_vram_key(profile)
        if key is None:
            return False
        for qj, _, _ in self._active.values():
            if self._shared_vram_key(qj.profile) == key:
                return True
        return False

    def _vram_delta_for_admit(self, profile: MetricProfile) -> float:
        if profile.execution == "shared_gpu" and self._shared_slot_occupied(profile):
            return 0.0
        return profile.vram_reserve_mib

    def _vram_delta_for_release(self, profile: MetricProfile) -> float:
        if profile.execution == "shared_gpu" and self._shared_slot_occupied(profile):
            return 0.0
        return profile.vram_reserve_mib

    def _ensure_baselines(self) -> None:
        """Capture host baselines once for unrealized reservation accounting."""
        from .telemetry import gpu_memory_mib, mem_available_mib

        if self._mem_baseline_mib is None:
            self._mem_baseline_mib = mem_available_mib()
        if self._gpu_baseline_mib is None:
            used, _ = gpu_memory_mib()
            self._gpu_baseline_mib = max(0, used)

    def _scheduler_ram_realized_mib(self) -> float:
        from .telemetry import mem_available_mib

        if self._mem_baseline_mib is None:
            return 0.0
        return max(0.0, self._mem_baseline_mib - mem_available_mib())

    def _scheduler_vram_realized_mib(self) -> float:
        from .telemetry import gpu_memory_mib

        if self._gpu_baseline_mib is None:
            return 0.0
        gpu_used, _ = gpu_memory_mib()
        if gpu_used < 0:
            return 0.0
        return max(0.0, float(gpu_used - self._gpu_baseline_mib))

    def _unrealized_ram_mib(self) -> float:
        """Scheduler RAM commitments not yet reflected in MemAvailable."""
        return max(0.0, self._ram_reserved - self._scheduler_ram_realized_mib())

    def _unrealized_vram_mib(self) -> float:
        """Scheduler VRAM commitments not yet reflected in nvidia-smi used."""
        return max(0.0, self._vram_reserved - self._scheduler_vram_realized_mib())

    def _effective_vram_load_mib(self) -> float:
        """Device-global GPU used vs scheduler commitments (no double-count)."""
        from .telemetry import gpu_memory_mib

        gpu_used, _ = gpu_memory_mib()
        telemetry = float(gpu_used) if gpu_used >= 0 else 0.0
        return max(telemetry, self._vram_reserved)

    def _profile_never_admissible(self, profile: MetricProfile) -> str | None:
        """Reject when a single job's reserve can never satisfy budgets."""
        from .telemetry import gpu_memory_mib, mem_available_mib

        floor = self._budget.mem_available_floor_mib
        headroom = self._budget.gpu_headroom_mib
        if profile.ram_reserve_mib + floor > mem_available_mib():
            return (
                f"profile RAM reserve {profile.ram_reserve_mib:.0f} MiB + floor "
                f"{floor:.0f} MiB exceeds MemAvailable {mem_available_mib():.0f} MiB"
            )
        vram_need = self._vram_delta_for_admit(profile)
        if vram_need <= 0:
            return None
        _, gpu_total = gpu_memory_mib()
        if gpu_total > 0 and vram_need + headroom > gpu_total:
            return (
                f"profile VRAM reserve {vram_need:.0f} MiB + headroom "
                f"{headroom} MiB exceeds GPU total {gpu_total} MiB"
            )
        return None

    def _can_admit(self, profile: MetricProfile) -> str | None:
        if self._global_admission:
            return self._profile_never_admissible(profile)
        metric = profile.metric
        cap = profile.effective_max_concurrent()
        active_metric = self._metric_active.get(metric, 0)
        if active_metric >= cap:
            return f"{metric} at per-metric cap {cap}"
        if len(self._active) >= self._budget.max_total_concurrent:
            return f"global concurrent cap {self._budget.max_total_concurrent}"
        reason = check_host_resources(
            mem_floor_mib=self._budget.mem_available_floor_mib,
            gpu_headroom_mib=self._budget.gpu_headroom_mib,
            swap_limit_mib=self._budget.swap_limit_mib,
        )
        if reason:
            return reason
        from .telemetry import gpu_memory_mib, mem_available_mib

        never = self._profile_never_admissible(profile)
        if never:
            return never

        floor = self._budget.mem_available_floor_mib
        ram_need = floor + self._unrealized_ram_mib() + profile.ram_reserve_mib
        avail = mem_available_mib()
        if avail < ram_need:
            return (
                f"RAM need {ram_need:.0f} MiB (floor {floor:.0f} + unrealized "
                f"{self._unrealized_ram_mib():.0f} + new {profile.ram_reserve_mib:.0f}) "
                f"exceeds MemAvailable {avail:.0f} MiB"
            )
        vram_add = self._vram_delta_for_admit(profile)
        _, gpu_total = gpu_memory_mib()
        if gpu_total > 0 and vram_add > 0:
            projected = (
                self._effective_vram_load_mib() + vram_add + self._budget.gpu_headroom_mib
            )
            if projected > gpu_total:
                return (
                    f"VRAM budget would exceed device total "
                    f"(effective {self._effective_vram_load_mib():.0f} + new {vram_add:.0f} + "
                    f"headroom {self._budget.gpu_headroom_mib} > {gpu_total} MiB)"
                )
        return None

    def _reserve(self, profile: MetricProfile) -> None:
        self._ram_reserved += profile.ram_reserve_mib
        self._vram_reserved += self._vram_delta_for_admit(profile)
        self._metric_active[profile.metric] = self._metric_active.get(profile.metric, 0) + 1

    def _release(self, profile: MetricProfile) -> None:
        self._ram_reserved = max(0.0, self._ram_reserved - profile.ram_reserve_mib)
        self._vram_reserved = max(0.0, self._vram_reserved - self._vram_delta_for_release(profile))
        count = self._metric_active.get(profile.metric, 1) - 1
        if count <= 0:
            self._metric_active.pop(profile.metric, None)
        else:
            self._metric_active[profile.metric] = count

    def _start_telemetry(self) -> None:
        if not self._started:
            self._ensure_baselines()
            self._telemetry.start()
            self._started = True

    def _reject_queued(self, qj: _QueuedJob, reason: str, state: JobState) -> None:
        qj.result = JobResult(
            spec=qj.spec,
            state=state,
            error=reason,
            profile=qj.profile,
            admission_wait_seconds=time.monotonic() - qj.enqueued_at,
        )
        self._results.append(qj.result)
        qj.event.set()

    def _expire_queued_jobs(self) -> None:
        """Drop jobs that waited too long or can never be admitted."""
        if not self._queue:
            return
        now = time.monotonic()
        timeout = self._budget.queue_timeout_s
        kept: deque[_QueuedJob] = deque()
        while self._queue:
            qj = self._queue.popleft()
            never = self._profile_never_admissible(qj.profile)
            if never:
                self._reject_queued(qj, never, JobState.REJECTED)
                continue
            if timeout > 0 and now - qj.enqueued_at >= timeout:
                self._reject_queued(
                    qj,
                    f"queue wait exceeded {timeout:.0f}s",
                    JobState.TIMEOUT,
                )
                continue
            kept.append(qj)
        self._queue = kept

    def _dispatcher_loop(self) -> None:
        while not self._shutdown.is_set():
            with self._cv:
                self._expire_queued_jobs()
                while self._queue and not self._abort_reason:
                    qj = self._queue[0]
                    reason = self._can_admit(qj.profile)
                    if reason:
                        if self._profile_never_admissible(qj.profile):
                            self._queue.popleft()
                            self._reject_queued(qj, reason, JobState.REJECTED)
                            continue
                        break
                    self._queue.popleft()
                    if not self._global_admission:
                        self._reserve(qj.profile)
                    if self.config.stagger_start_s > 0 and self._active:
                        self._cv.release()
                        time.sleep(self.config.stagger_start_s)
                        self._cv.acquire()
                    child_env = dict(qj.spec.env or {})
                    lease = None
                    if self._global_admission:
                        try:
                            lease = acquire(
                                qj.spec.metric,
                                shared_gpu=qj.spec.shared_gpu,
                            )
                            child_env["PIXVQA_SCHEDULER_ADMISSION_SKIP"] = "1"
                        except AdmissionError as exc:
                            qj.result = JobResult(
                                spec=qj.spec,
                                state=JobState.REJECTED,
                                error=str(exc),
                                profile=qj.profile,
                                admission_wait_seconds=time.monotonic() - qj.enqueued_at,
                            )
                            self._results.append(qj.result)
                            qj.event.set()
                            continue
                    try:
                        running = launch(
                            qj.spec.job_id,
                            qj.spec.metric,
                            qj.spec.command,
                            env=child_env or qj.spec.env,
                            cwd=qj.spec.cwd,
                        )
                    except OSError as exc:
                        if not self._global_admission:
                            self._release(qj.profile)
                        elif lease is not None:
                            try:
                                release(lease)
                            except (AdmissionError, OSError):
                                pass
                        qj.result = JobResult(
                            spec=qj.spec,
                            state=JobState.FAILED,
                            error=str(exc),
                            profile=qj.profile,
                            admission_wait_seconds=time.monotonic() - qj.enqueued_at,
                        )
                        self._results.append(qj.result)
                        qj.event.set()
                        continue
                    if lease is not None:
                        self._job_leases[qj.spec.job_id] = lease
                    self._active[qj.spec.job_id] = (qj, running, time.monotonic())
                    threading.Thread(
                        target=self._watch_job,
                        args=(qj, running),
                        daemon=True,
                        name=f"pixvqa-sched-{qj.spec.job_id}",
                    ).start()
                self._cv.wait(timeout=0.2)

    def _watch_job(self, qj: _QueuedJob, running: RunningJob) -> None:
        def on_tick() -> None:
            reason = self._telemetry.should_abort(
                mem_floor_mib=self._budget.mem_available_floor_mib,
                gpu_headroom_mib=self._budget.gpu_headroom_mib,
                swap_limit_mib=self._budget.swap_limit_mib,
            )
            if reason:
                with self._lock:
                    self._abort_reason = reason
                running.process.kill()

        t0 = time.monotonic()
        rc, stdout, stderr = wait_job(
            running,
            timeout_s=qj.spec.timeout_s,
            on_tick=on_tick,
        )
        wall = time.monotonic() - t0
        if self._abort_reason:
            state = JobState.CANCELLED
            err = self._abort_reason
        elif rc == -9 and qj.spec.timeout_s:
            state = JobState.TIMEOUT
            err = "job timeout"
        elif rc != 0:
            state = JobState.FAILED
            err = f"exit {rc}"
        else:
            state = JobState.COMPLETED
            err = None
        result = JobResult(
            spec=qj.spec,
            state=state,
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            wall_seconds=wall,
            admission_wait_seconds=time.monotonic() - qj.enqueued_at - wall,
            error=err,
            profile=qj.profile,
        )
        lease = self._job_leases.pop(qj.spec.job_id, None)
        if lease is not None:
            try:
                release(lease)
            except (AdmissionError, OSError):
                pass
        with self._lock:
            self._active.pop(qj.spec.job_id, None)
            if not self._global_admission:
                self._release(qj.profile)
            self._results.append(result)
            qj.result = result
            qj.event.set()
            self._cv.notify_all()

    def submit(self, spec: JobSpec) -> str:
        """Queue a job; returns job_id."""
        if not self.config.enabled:
            raise RuntimeError(
                "MetricScheduler is disabled (set PIXVQA_SCHEDULER=1 to enable)"
            )
        self._start_telemetry()
        profile = self._resolve_profile(spec)
        qj = _QueuedJob(spec, profile)
        with self._lock:
            never = self._profile_never_admissible(profile)
            if never:
                self._reject_queued(qj, never, JobState.REJECTED)
                return spec.job_id
            if self._dispatcher is None or not self._dispatcher.is_alive():
                self._dispatcher = threading.Thread(
                    target=self._dispatcher_loop, daemon=True, name="pixvqa-sched-dispatch"
                )
                self._dispatcher.start()
            if len(self._queue) >= self._budget.max_queue_depth:
                self._reject_queued(qj, "queue depth limit", JobState.REJECTED)
                return spec.job_id
            self._queue.append(qj)
            self._cv.notify_all()
        return spec.job_id

    def submit_many(self, specs: list[JobSpec]) -> list[str]:
        return [self.submit(s) for s in specs]

    def _find_queued(self, job_id: str) -> _QueuedJob | None:
        for qj in self._queue:
            if qj.spec.job_id == job_id:
                return qj
        return None

    def wait_job(self, job_id: str, *, timeout_s: float | None = None) -> JobResult:
        deadline = time.monotonic() + timeout_s if timeout_s else None
        qj: _QueuedJob | None = None
        while True:
            with self._lock:
                for r in self._results:
                    if r.spec.job_id == job_id:
                        return r
                qj = self._find_queued(job_id)
                if qj is None and job_id in self._active:
                    qj = self._active[job_id][0]
                if qj is None:
                    raise KeyError(job_id)
                if qj.result is not None:
                    return qj.result
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining == 0.0:
                raise TimeoutError(f"job {job_id} wait timed out")
            qj.event.wait(timeout=remaining if remaining is not None else 1.0)

    def run_batch(
        self,
        specs: list[JobSpec],
        *,
        batch_timeout_s: float | None = None,
    ) -> SchedulerReport:
        """Submit all specs and block until each reaches a terminal state."""
        if not self.config.enabled:
            # Pass-through: run sequentially without admission (backward compatible).
            return self._run_batch_passthrough(specs)
        t0 = time.monotonic()
        ids = self.submit_many(specs)
        results: list[JobResult] = []
        deadline = time.monotonic() + batch_timeout_s if batch_timeout_s else None
        for job_id in ids:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                results.append(self.wait_job(job_id, timeout_s=remaining))
            except TimeoutError:
                results.append(
                    JobResult(
                        spec=next(s for s in specs if s.job_id == job_id),
                        state=JobState.TIMEOUT,
                        error="batch wait timeout",
                    )
                )
        self.shutdown()
        warnings = []
        if not self._global_admission:
            warnings.append(self.config.cross_invocation_warning)
        if self._abort_reason:
            warnings.append(f"safety abort: {self._abort_reason}")
        return SchedulerReport(
            jobs=results,
            telemetry=self._telemetry.summary(),
            warnings=warnings,
            total_wall_seconds=time.monotonic() - t0,
        )

    def _run_batch_passthrough(self, specs: list[JobSpec]) -> SchedulerReport:
        t0 = time.monotonic()
        results: list[JobResult] = []
        for spec in specs:
            profile = self._resolve_profile(spec)
            t_job = time.monotonic()
            try:
                running = launch(spec.job_id, spec.metric, spec.command, env=spec.env, cwd=spec.cwd)
                rc, stdout, stderr = wait_job(running, timeout_s=spec.timeout_s)
                state = JobState.COMPLETED if rc == 0 else JobState.FAILED
                results.append(
                    JobResult(
                        spec=spec,
                        state=state,
                        returncode=rc,
                        stdout=stdout,
                        stderr=stderr,
                        wall_seconds=time.monotonic() - t_job,
                        profile=profile,
                    )
                )
            except OSError as exc:
                results.append(
                    JobResult(
                        spec=spec,
                        state=JobState.FAILED,
                        error=str(exc),
                        profile=profile,
                    )
                )
        return SchedulerReport(
            jobs=results,
            telemetry={},
            warnings=["scheduler disabled: sequential pass-through"],
            total_wall_seconds=time.monotonic() - t0,
        )

    def shutdown(self) -> None:
        self._shutdown.set()
        with self._cv:
            self._cv.notify_all()
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=5)
        self._telemetry.stop()

    def cancel_all(self, reason: str = "cancelled") -> None:
        with self._lock:
            self._abort_reason = reason
            while self._queue:
                qj = self._queue.popleft()
                self._reject_queued(qj, reason, JobState.CANCELLED)
            for job_id, (_, running, _) in list(self._active.items()):
                if running.process.poll() is None:
                    running.process.kill()
            self._cv.notify_all()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "queued": len(self._queue),
                "active": len(self._active),
                "ram_reserved_mib": round(self._ram_reserved, 1),
                "vram_reserved_mib": round(self._vram_reserved, 1),
                "unrealized_ram_mib": round(self._unrealized_ram_mib(), 1),
                "unrealized_vram_mib": round(self._unrealized_vram_mib(), 1),
                "metric_active": dict(self._metric_active),
                "abort_reason": self._abort_reason,
            }
