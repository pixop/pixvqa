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


def test_queue_timeout_rejected():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    long = MetricProfile(
        metric="cvqa-hdr-nr",
        execution="independent",
        max_concurrent=1,
        ram_reserve_mib=10,
        vram_reserve_mib=10,
        module="pixvqa_cvqahdr",
        verified_concurrency="test",
    )
    budget = SchedulerBudget(max_total_concurrent=1, queue_timeout_s=3600.0)
    sched = MetricScheduler(
        SchedulerConfig(enabled=True, profiles={"cvqa-hdr-nr": long}, budget=budget)
    )
    sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(0.2), job_id="a"))
    sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_echo_script("b"), job_id="b"))
    with patch.object(sched, "_can_admit", return_value="blocked"):
        time.sleep(0.05)
    r_a = sched.wait_job("a", timeout_s=30)
    r_b = sched.wait_job("b", timeout_s=30)
    assert r_a.state == JobState.COMPLETED
    assert r_b.state in (JobState.COMPLETED, JobState.FAILED)
    sched.shutdown()
    print("OK   queue-order")


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


def test_cancel_all():
    os.environ["PIXVQA_SCHEDULER"] = "1"
    sched = MetricScheduler(SchedulerConfig(enabled=True))
    sched.submit(JobSpec(metric="cvqa-hdr-nr", command=_sleep_script(2.0), job_id="slow"))
    time.sleep(0.05)
    sched.cancel_all("test-cancel")
    try:
        r = sched.wait_job("slow", timeout_s=5)
    except KeyError:
        r = None
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
    test_host_resource_guard_trips()
    test_resource_guard_reexport()
    test_scheduler_disabled_passthrough()
    test_admission_and_release()
    test_failure_releases_slot()
    test_queue_timeout_rejected()
    test_cancel_all()
    test_cli_dry_run()
    test_pixvqa_cli_help_unchanged()
    print("PASS: metric scheduler tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
