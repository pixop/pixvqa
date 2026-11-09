#!/usr/bin/env python3
"""Unit tests for pixvqa_scheduler admission, queue, release, and failure paths."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pixvqa_scheduler import MetricScheduler, JobSpec, SchedulerConfig  # noqa: E402
from pixvqa_scheduler.profiles import MetricProfile, SchedulerBudget, profile_for  # noqa: E402
from pixvqa_scheduler.scheduler import JobState, scheduler_enabled  # noqa: E402


def _sleep_script(seconds: float = 0.05) -> list[str]:
    return [
        sys.executable,
        "-c",
        f"import time; time.sleep({seconds})",
    ]


def _fail_script() -> list[str]:
    return [sys.executable, "-c", "import sys; sys.exit(3)"]


def _echo_script(payload: str) -> list[str]:
    return [sys.executable, "-c", f"print({payload!r})"]


def test_scheduler_disabled_passthrough():
    cfg = SchedulerConfig(enabled=False)
    sched = MetricScheduler(cfg)
    report = sched.run_batch([
        JobSpec(metric="simplevqa", command=_echo_script("ok")),
    ])
    assert len(report.jobs) == 1
    assert report.jobs[0].state == JobState.COMPLETED
    assert "ok" in report.jobs[0].stdout
    print("OK   disabled-passthrough")


def test_admission_and_release():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    os.environ["PIXVQA_SCHEDULER_MAX_CONCURRENT_CVQA_HDR_NR"] = "2"
    os.environ["PIXVQA_SCHEDULER_CVQA_VRAM_MIB"] = "100"
    os.environ["PIXVQA_SCHEDULER_CVQA_RAM_MIB"] = "100"
    profiles = {
        "cvqa-hdr-nr": MetricProfile(
            metric="cvqa-hdr-nr",
            execution="independent",
            max_concurrent=2,
            ram_reserve_mib=100,
            vram_reserve_mib=100,
            module="pixvqa_cvqahdr",
            verified_concurrency="test",
        ),
    }
    cfg = SchedulerConfig(enabled=True, profiles=profiles)
    sched = MetricScheduler(cfg)
    specs = [
        JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(0.08), job_id=f"j{i}")
        for i in range(3)
    ]
    report = sched.run_batch(specs, batch_timeout_s=30)
    states = [r.state for r in report.jobs]
    assert states.count(JobState.COMPLETED) == 3
    snap = sched.snapshot()
    assert snap["active"] == 0
    assert snap["queued"] == 0
    print("OK   admission-release")


def test_failure_releases_slot():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    profiles = {
        "cvqa-hdr-nr": MetricProfile(
            metric="cvqa-hdr-nr",
            execution="independent",
            max_concurrent=1,
            ram_reserve_mib=10,
            vram_reserve_mib=10,
            module="pixvqa_cvqahdr",
            verified_concurrency="test",
        ),
    }
    sched = MetricScheduler(SchedulerConfig(enabled=True, profiles=profiles))
    report = sched.run_batch([
        JobSpec(metric="cvqa-hdr-nr", command=_fail_script(), job_id="fail"),
        JobSpec(metric="cvqa-hdr-nr", command=_echo_script("after"), job_id="ok"),
    ], batch_timeout_s=30)
    by_id = {r.spec.job_id: r for r in report.jobs}
    assert by_id["fail"].state == JobState.FAILED
    assert by_id["ok"].state == JobState.COMPLETED
    print("OK   failure-release")


def test_queue_order_with_cap():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    cvqa = MetricProfile(
        metric="cvqa-hdr-nr",
        execution="independent",
        max_concurrent=1,
        ram_reserve_mib=10,
        vram_reserve_mib=10,
        module="pixvqa_cvqahdr",
        verified_concurrency="test",
    )
    sched = MetricScheduler(
        SchedulerConfig(enabled=True, profiles={"cvqa-hdr-nr": cvqa})
    )
    sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(0.2), job_id="a"))
    sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_echo_script("b"), job_id="b"))
    r_a = sched.wait_job("a", timeout_s=30)
    r_b = sched.wait_job("b", timeout_s=30)
    assert r_a.state == JobState.COMPLETED
    assert r_b.state == JobState.COMPLETED
    sched.shutdown()
    print("OK   queue-order")


def test_vram_no_double_count_five_cvqa():
    """Five 2.8 GiB CVQA jobs should admit on a ~16 GiB GPU without double-counting."""
    os.environ["PIXVQA_SCHEDULER"] = "1"
    cvqa = MetricProfile(
        metric="cvqa-hdr-nr",
        execution="independent",
        max_concurrent=5,
        ram_reserve_mib=100,
        vram_reserve_mib=2800,
        module="pixvqa_cvqahdr",
        verified_concurrency="test",
    )
    budget = SchedulerBudget(gpu_headroom_mib=512, mem_available_floor_mib=256)
    sched = MetricScheduler(SchedulerConfig(enabled=True, profiles={"cvqa-hdr-nr": cvqa}, budget=budget))

    gpu_used = {"value": 0}

    def fake_gpu(_=None):
        return gpu_used["value"], 16_384

    def fake_mem():
        return 32_768.0

    with patch("pixvqa_scheduler.telemetry.gpu_memory_mib", side_effect=fake_gpu), patch(
        "pixvqa_scheduler.telemetry.mem_available_mib", side_effect=fake_mem
    ):
        specs = [
            JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(0.04), job_id=f"c{i}")
            for i in range(5)
        ]
        report = sched.run_batch(specs, batch_timeout_s=60)
    completed = [r for r in report.jobs if r.state == JobState.COMPLETED]
    assert len(completed) == 5, [r.state for r in report.jobs]
    print("OK   vram-five-cvqa")


def test_vram_sixth_cvqa_blocked_not_hung():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    cvqa = MetricProfile(
        metric="cvqa-hdr-nr",
        execution="independent",
        max_concurrent=5,
        ram_reserve_mib=100,
        vram_reserve_mib=2800,
        module="pixvqa_cvqahdr",
        verified_concurrency="test",
    )
    budget = SchedulerBudget(
        gpu_headroom_mib=512,
        mem_available_floor_mib=256,
        queue_timeout_s=2.0,
    )
    sched = MetricScheduler(SchedulerConfig(enabled=True, profiles={"cvqa-hdr-nr": cvqa}, budget=budget))

    def fake_gpu(_=None):
        return 0, 16_384

    def fake_mem():
        return 32_768.0

    with patch("pixvqa_scheduler.telemetry.gpu_memory_mib", side_effect=fake_gpu), patch(
        "pixvqa_scheduler.telemetry.mem_available_mib", side_effect=fake_mem
    ):
        for i in range(5):
            sched.submit(
                JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(3.0), job_id=f"hold{i}")
            )
        time.sleep(0.15)
        sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_echo_script("x"), job_id="sixth"))
        r6 = sched.wait_job("sixth", timeout_s=10)
    assert r6.state in (JobState.TIMEOUT, JobState.REJECTED, JobState.FAILED)
    sched.cancel_all("cleanup")
    sched.shutdown()
    print("OK   vram-sixth-blocked")


def test_impossible_vram_profile_rejected():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    huge = MetricProfile(
        metric="cvqa-hdr-nr",
        execution="independent",
        max_concurrent=1,
        ram_reserve_mib=100,
        vram_reserve_mib=20_000,
        module="pixvqa_cvqahdr",
        verified_concurrency="test",
    )
    sched = MetricScheduler(SchedulerConfig(enabled=True, profiles={"cvqa-hdr-nr": huge}))

    def fake_gpu(_=None):
        return 0, 16_384

    with patch("pixvqa_scheduler.telemetry.gpu_memory_mib", side_effect=fake_gpu):
        job_id = sched.submit(
            JobSpec(metric="cvqa-hdr-nr", command=_echo_script("nope"), job_id="big")
        )
        r = sched.wait_job(job_id, timeout_s=5)
    assert r.state == JobState.REJECTED
    sched.shutdown()
    print("OK   impossible-vram-rejected")


def test_mix_cvqa_simplevqa_admission():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    cvqa = MetricProfile(
        metric="cvqa-hdr-nr",
        execution="independent",
        max_concurrent=3,
        ram_reserve_mib=100,
        vram_reserve_mib=2800,
        module="pixvqa_cvqahdr",
        verified_concurrency="test",
    )
    shared = MetricProfile(
        metric="simplevqa",
        execution="shared_gpu",
        max_concurrent=8,
        ram_reserve_mib=100,
        vram_reserve_mib=4200,
        module="pixvqa_simplevqa",
        verified_concurrency="test",
    )
    profiles = {"cvqa-hdr-nr": cvqa, "simplevqa:shared_gpu": shared}
    sched = MetricScheduler(
        SchedulerConfig(enabled=True, profiles=profiles, budget=SchedulerBudget(gpu_headroom_mib=512))
    )

    def fake_gpu(_=None):
        return 0, 16_384

    def fake_mem():
        return 32_768.0

    with patch("pixvqa_scheduler.telemetry.gpu_memory_mib", side_effect=fake_gpu), patch(
        "pixvqa_scheduler.telemetry.mem_available_mib", side_effect=fake_mem
    ):
        specs = [
            JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(0.03), job_id="c1"),
            JobSpec(
                metric="simplevqa",
                command=_sleep_script(0.03),
                job_id="s1",
                shared_gpu=True,
            ),
            JobSpec(
                metric="simplevqa",
                command=_sleep_script(0.03),
                job_id="s2",
                shared_gpu=True,
            ),
        ]
        report = sched.run_batch(specs, batch_timeout_s=30)
    assert all(r.state == JobState.COMPLETED for r in report.jobs)
    snap = sched.snapshot()
    assert snap["vram_reserved_mib"] == 0.0
    print("OK   mix-cvqa-shared-svqa")


def test_shared_gpu_vram_release_once():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    shared = MetricProfile(
        metric="simplevqa",
        execution="shared_gpu",
        max_concurrent=4,
        ram_reserve_mib=50,
        vram_reserve_mib=4200,
        module="pixvqa_simplevqa",
        verified_concurrency="test",
    )
    sched = MetricScheduler(
        SchedulerConfig(
            enabled=True,
            profiles={"simplevqa:shared_gpu": shared},
            budget=SchedulerBudget(gpu_headroom_mib=256),
        )
    )

    def fake_gpu(_=None):
        return 0, 16_384

    with patch("pixvqa_scheduler.telemetry.gpu_memory_mib", side_effect=fake_gpu):
        specs = [
            JobSpec(
                metric="simplevqa",
                command=_sleep_script(0.05),
                job_id=f"sh{i}",
                shared_gpu=True,
            )
            for i in range(3)
        ]
        report = sched.run_batch(specs, batch_timeout_s=30)
    assert all(r.state == JobState.COMPLETED for r in report.jobs)
    print("OK   shared-gpu-release")


def test_gpu_guard_abort_during_job():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    sched = MetricScheduler(
        SchedulerConfig(
            enabled=True,
            profiles={
                "cvqa-hdr-nr": MetricProfile(
                    metric="cvqa-hdr-nr",
                    execution="independent",
                    max_concurrent=1,
                    ram_reserve_mib=10,
                    vram_reserve_mib=10,
                    module="pixvqa_cvqahdr",
                    verified_concurrency="test",
                )
            },
        )
    )
    sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(2.0), job_id="guarded"))
    time.sleep(0.05)
    with patch.object(sched._telemetry, "should_abort", return_value="gpu spike"):
        time.sleep(0.3)
    r = sched.wait_job("guarded", timeout_s=10)
    assert r.state == JobState.CANCELLED
    assert "gpu spike" in (r.error or "")
    sched.shutdown()
    print("OK   gpu-guard-abort")


def test_accounting_mock_no_double_count():
    """With telemetry showing full VRAM, scheduler must not add reserved again."""
    os.environ["PIXVQA_SCHEDULER"] = "1"
    cvqa = MetricProfile(
        metric="cvqa-hdr-nr",
        execution="independent",
        max_concurrent=5,
        ram_reserve_mib=100,
        vram_reserve_mib=2800,
        module="pixvqa_cvqahdr",
        verified_concurrency="test",
    )
    sched = MetricScheduler(
        SchedulerConfig(
            enabled=True,
            profiles={"cvqa-hdr-nr": cvqa},
            budget=SchedulerBudget(gpu_headroom_mib=512, mem_available_floor_mib=256),
        )
    )
    sched._vram_reserved = 5600.0
    sched._ram_reserved = 200.0
    sched._gpu_baseline_mib = 0
    sched._mem_baseline_mib = 32_768.0

    def fake_gpu(_=None):
        return 5600, 16_384

    def fake_mem():
        return 32_500.0

    with patch("pixvqa_scheduler.telemetry.gpu_memory_mib", side_effect=fake_gpu), patch(
        "pixvqa_scheduler.telemetry.mem_available_mib", side_effect=fake_mem
    ):
        reason = sched._can_admit(cvqa)
    assert reason is None, reason
    print("OK   accounting-no-double-count")


def test_mix_cvqa_simplevqa_profiles():
    cvqa = profile_for("cvqa-hdr-nr")
    svqa = profile_for("simplevqa")
    shared = profile_for("simplevqa", shared_gpu=True)
    assert cvqa.execution == "independent"
    assert cvqa.max_concurrent >= 5
    assert svqa.max_concurrent <= 2
    assert shared.execution == "shared_gpu"
    assert shared.max_concurrent >= 5
    print("OK   mix-profiles")


def test_cvqa_1080p_batch_profile_defaults():
    os.environ.pop("PIXVQA_SCHEDULER_CVQA_RAM_MIB", None)
    os.environ.pop("PIXVQA_SCHEDULER_CVQA_VRAM_MIB", None)
    os.environ["PIXVQA_SCHEDULER_CVQA_PROFILE"] = "1080p-batch"
    # Re-import tier defaults (profile_for caches module-level profile).
    from pixvqa_scheduler import profiles as prof_mod  # noqa: E402

    ram, vram = prof_mod._cvqa_default_reserves()
    assert ram == 1650.0
    assert vram == 1300.0
    os.environ["PIXVQA_SCHEDULER_CVQA_PROFILE"] = "conservative"
    ram_c, vram_c = prof_mod._cvqa_default_reserves()
    assert ram_c == 2500.0 and vram_c == 2800.0
    print("OK   cvqa-profile-tiers")


def test_cancel_all():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    sched = MetricScheduler(SchedulerConfig(enabled=True))
    sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(2.0), job_id="slow"))
    sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(2.0), job_id="queued"))
    time.sleep(0.05)
    sched.cancel_all("test-cancel")
    r_slow = sched.wait_job("slow", timeout_s=5)
    r_queued = sched.wait_job("queued", timeout_s=5)
    assert r_slow.state == JobState.CANCELLED
    assert r_queued.state == JobState.CANCELLED
    sched.shutdown()
    print("OK   cancel-all")


def test_host_resource_guard_trips():
    from pixvqa_scheduler.telemetry import check_host_resources

    with patch("pixvqa_scheduler.telemetry.mem_available_mib", return_value=100.0):
        reason = check_host_resources(mem_floor_mib=2048)
        assert reason is not None
    print("OK   host-guard")


def test_cli_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = Path(tmpdir) / "jobs.json"
        spec.write_text(
            json.dumps([
                {"metric": "cvqa-hdr-nr", "command": [sys.executable, "-c", "pass"]},
            ]),
            encoding="utf-8",
        )
        import subprocess

        proc = subprocess.run(
            [
                sys.executable, "-m", "pixvqa_scheduler", "--dry-run", "mix-example",
                "--spec-json", str(spec),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr)
        payload = json.loads(proc.stdout)
        assert payload["jobs"] == 1
    print("OK   cli-dry-run")


def test_pixvqa_cli_help_unchanged():
    """Integration: pixvqa --help still lists learned metrics (no scheduler flag required)."""
    pixvqa = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
    if not pixvqa.is_file():
        print("SKIP pixvqa-cli-help (binary missing)")
        return
    import subprocess

    proc = subprocess.run([str(pixvqa), "--help"], capture_output=True, text=True)
    if "cvqa-hdr-nr" not in proc.stdout or "simplevqa" not in proc.stdout:
        raise SystemExit(f"CLI help missing metrics:\n{proc.stdout}")
    print("OK   pixvqa-cli-help")


def test_resource_guard_reexport():
    from pixvqa_simplevqa.resource_guard import check_resources, mem_available_mib

    assert callable(check_resources)
    assert mem_available_mib() > 0
    print("OK   resource-guard-reexport")


def test_scheduler_enabled_env():
    os.environ.pop("PIXVQA_SCHEDULER", None)
    assert not scheduler_enabled()
    os.environ["PIXVQA_SCHEDULER"] = "1"
    assert scheduler_enabled()
    print("OK   scheduler-enabled-env")


def main() -> int:
    test_scheduler_enabled_env()
    test_mix_cvqa_simplevqa_profiles()
    test_cvqa_1080p_batch_profile_defaults()
    test_host_resource_guard_trips()
    test_resource_guard_reexport()
    test_scheduler_disabled_passthrough()
    test_admission_and_release()
    test_failure_releases_slot()
    test_queue_order_with_cap()
    test_vram_no_double_count_five_cvqa()
    test_vram_sixth_cvqa_blocked_not_hung()
    test_impossible_vram_profile_rejected()
    test_mix_cvqa_simplevqa_admission()
    test_shared_gpu_vram_release_once()
    test_gpu_guard_abort_during_job()
    test_accounting_mock_no_double_count()
    test_cancel_all()
    test_cli_dry_run()
    test_pixvqa_cli_help_unchanged()
    print("PASS: metric scheduler tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
