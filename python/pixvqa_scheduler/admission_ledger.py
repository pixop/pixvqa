"""Shared RAM/VRAM admission accounting (no double-count vs telemetry)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .profiles import MetricProfile, SchedulerBudget, default_profiles, profile_for
from .telemetry import check_host_resources, gpu_memory_mib, mem_available_mib


@dataclass
class AdmissionLedger:
    """Tracks scheduler reservations and answers admit / reserve / release."""

    budget: SchedulerBudget = field(default_factory=SchedulerBudget)
    profiles: dict[str, MetricProfile] = field(default_factory=default_profiles)
    ram_reserved: float = 0.0
    vram_reserved: float = 0.0
    metric_active: dict[str, int] = field(default_factory=dict)
    active_leases: int = 0
    mem_baseline_mib: float | None = None
    gpu_baseline_mib: int | None = None

    def resolve_profile(self, metric: str, *, shared_gpu: bool = False) -> MetricProfile:
        if metric == "simplevqa" and shared_gpu:
            return profile_for("simplevqa", shared_gpu=True)
        key = metric.lower()
        if key in self.profiles:
            return self.profiles[key]
        return profile_for(metric, shared_gpu=shared_gpu)

    def _shared_vram_key(self, profile: MetricProfile) -> str | None:
        if profile.execution == "shared_gpu":
            return f"{profile.metric}:{profile.execution}"
        return None

    def _shared_slot_occupied(self, profile: MetricProfile, *, excluding_token: str | None = None,
                              leases: dict | None = None) -> bool:
        if leases is None:
            return False
        key = self._shared_vram_key(profile)
        if key is None:
            return False
        for token, lease in leases.items():
            if excluding_token and token == excluding_token:
                continue
            if lease.get("shared_vram_key") == key:
                return True
        return False

    def vram_delta_for_admit(self, profile: MetricProfile, *, leases: dict | None = None) -> float:
        if profile.execution == "shared_gpu" and self._shared_slot_occupied(profile, leases=leases):
            return 0.0
        return profile.vram_reserve_mib

    def vram_delta_for_release(self, profile: MetricProfile, *, leases: dict | None = None) -> float:
        if profile.execution == "shared_gpu" and self._shared_slot_occupied(profile, leases=leases):
            return 0.0
        return profile.vram_reserve_mib

    def ensure_baselines(self) -> None:
        if self.mem_baseline_mib is None:
            self.mem_baseline_mib = mem_available_mib()
        if self.gpu_baseline_mib is None:
            used, _ = gpu_memory_mib()
            self.gpu_baseline_mib = max(0, used)

    def scheduler_ram_realized_mib(self) -> float:
        if self.mem_baseline_mib is None:
            return 0.0
        return max(0.0, self.mem_baseline_mib - mem_available_mib())

    def scheduler_vram_realized_mib(self) -> float:
        if self.gpu_baseline_mib is None:
            return 0.0
        gpu_used, _ = gpu_memory_mib()
        if gpu_used < 0:
            return 0.0
        return max(0.0, float(gpu_used - self.gpu_baseline_mib))

    def unrealized_ram_mib(self) -> float:
        return max(0.0, self.ram_reserved - self.scheduler_ram_realized_mib())

    def unrealized_vram_mib(self) -> float:
        return max(0.0, self.vram_reserved - self.scheduler_vram_realized_mib())

    def effective_vram_load_mib(self) -> float:
        gpu_used, _ = gpu_memory_mib()
        telemetry = float(gpu_used) if gpu_used >= 0 else 0.0
        return max(telemetry, self.vram_reserved)

    def profile_never_admissible(self, profile: MetricProfile, *, leases: dict | None = None) -> str | None:
        headroom = self.budget.gpu_headroom_mib
        # MemAvailable is dynamic, not a machine-capacity limit. A shortage
        # here is transient contention and must be handled by can_admit()
        # (queued with a bounded timeout), never rejected as impossible.
        vram_need = self.vram_delta_for_admit(profile, leases=leases)
        if vram_need <= 0:
            return None
        _, gpu_total = gpu_memory_mib()
        if gpu_total > 0 and vram_need + headroom > gpu_total:
            return (
                f"profile VRAM reserve {vram_need:.0f} MiB + headroom "
                f"{headroom} MiB exceeds GPU total {gpu_total} MiB"
            )
        return None

    def can_admit(self, profile: MetricProfile, *, leases: dict | None = None) -> str | None:
        metric = profile.metric
        cap = profile.effective_max_concurrent()
        active_metric = self.metric_active.get(metric, 0)
        if active_metric >= cap:
            return f"{metric} at per-metric cap {cap}"
        if self.active_leases >= self.budget.max_total_concurrent:
            return f"global concurrent cap {self.budget.max_total_concurrent}"
        reason = check_host_resources(
            mem_floor_mib=self.budget.mem_available_floor_mib,
            gpu_headroom_mib=self.budget.gpu_headroom_mib,
            swap_limit_mib=self.budget.swap_limit_mib,
        )
        if reason:
            return reason
        never = self.profile_never_admissible(profile, leases=leases)
        if never:
            return never
        floor = self.budget.mem_available_floor_mib
        ram_need = floor + self.unrealized_ram_mib() + profile.ram_reserve_mib
        avail = mem_available_mib()
        if avail < ram_need:
            return (
                f"RAM need {ram_need:.0f} MiB exceeds MemAvailable {avail:.0f} MiB"
            )
        vram_add = self.vram_delta_for_admit(profile, leases=leases)
        _, gpu_total = gpu_memory_mib()
        if gpu_total > 0 and vram_add > 0:
            projected = self.effective_vram_load_mib() + vram_add + self.budget.gpu_headroom_mib
            if projected > gpu_total:
                return (
                    f"VRAM budget would exceed device total "
                    f"({self.effective_vram_load_mib():.0f} + {vram_add:.0f} + "
                    f"{self.budget.gpu_headroom_mib} > {gpu_total} MiB)"
                )
        return None

    def reserve(self, profile: MetricProfile, *, leases: dict | None = None) -> None:
        self.ram_reserved += profile.ram_reserve_mib
        self.vram_reserved += self.vram_delta_for_admit(profile, leases=leases)
        self.metric_active[profile.metric] = self.metric_active.get(profile.metric, 0) + 1
        self.active_leases += 1

    def release(self, profile: MetricProfile, *, leases: dict | None = None) -> None:
        self.ram_reserved = max(0.0, self.ram_reserved - profile.ram_reserve_mib)
        self.vram_reserved = max(
            0.0, self.vram_reserved - self.vram_delta_for_release(profile, leases=leases)
        )
        count = self.metric_active.get(profile.metric, 1) - 1
        if count <= 0:
            self.metric_active.pop(profile.metric, None)
        else:
            self.metric_active[profile.metric] = count
        self.active_leases = max(0, self.active_leases - 1)

    def snapshot(self) -> dict:
        return {
            "active_leases": self.active_leases,
            "ram_reserved_mib": round(self.ram_reserved, 1),
            "vram_reserved_mib": round(self.vram_reserved, 1),
            "unrealized_ram_mib": round(self.unrealized_ram_mib(), 1),
            "unrealized_vram_mib": round(self.unrealized_vram_mib(), 1),
            "metric_active": dict(self.metric_active),
        }
