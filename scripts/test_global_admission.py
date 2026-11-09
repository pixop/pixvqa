#!/usr/bin/env python3
"""Cross-process admission daemon tests (subprocess clients, crashes, budgets)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pixvqa_scheduler.admission_ledger import AdmissionLedger  # noqa: E402
from pixvqa_scheduler.client import acquire, release  # noqa: E402
from pixvqa_scheduler.daemon_manager import admission_daemon_available, ensure_admission_daemon  # noqa: E402
from pixvqa_scheduler.ipc import runtime_dir, socket_address  # noqa: E402
from pixvqa_scheduler.profiles import MetricProfile, SchedulerBudget  # noqa: E402


def _sleep_cmd(seconds: float = 0.2) -> list[str]:
    return [
        sys.executable,
        "-c",
        f"import os, time; os.environ['PIXVQA_SCHEDULER_ADMISSION']='1'; "
        f"os.environ['PYTHONPATH']={str(ROOT / 'python')!r}; "
        f"from pixvqa_scheduler.admission import optional_admission; "
        f"import contextlib; "
        f"ctx = optional_admission('cvqa-hdr-nr'); "
        f"ctx.__enter__(); time.sleep({seconds}); ctx.__exit__(None,None,None)",
    ]


def test_daemon_probe():
    if not admission_daemon_available():
        print("SKIP daemon-probe (no AF_UNIX)")
        return
    os.environ["PIXVQA_SCHEDULER_DIR"] = tempfile.mkdtemp(prefix="pixvqa-adm-test-")
    ensure_admission_daemon()
    lease = acquire("cvqa-hdr-nr", timeout_s=10)
    release(lease)
    print("OK   daemon-probe")


def test_parallel_clients():
    if not admission_daemon_available():
        print("SKIP parallel-clients")
        return
    os.environ["PIXVQA_SCHEDULER_ADMISSION"] = "1"
    os.environ["PIXVQA_SCHEDULER_DIR"] = tempfile.mkdtemp(prefix="pixvqa-adm-test-")
    ensure_admission_daemon()
    procs = [
        subprocess.Popen(_sleep_cmd(0.35), env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
        for _ in range(3)
    ]
    codes = [p.wait(timeout=60) for p in procs]
    assert all(c == 0 for c in codes), codes
    print("OK   parallel-clients")


def test_crash_releases_slot():
    if not admission_daemon_available():
        print("SKIP crash-release")
        return
    os.environ["PIXVQA_SCHEDULER_DIR"] = tempfile.mkdtemp(prefix="pixvqa-adm-test-")
    ensure_admission_daemon()
    crash = [
        sys.executable,
        "-c",
        "import os, sys; os.environ['PIXVQA_SCHEDULER_ADMISSION']='1'; "
        f"sys.path.insert(0,{str(ROOT / 'python')!r}); "
        "from pixvqa_scheduler.admission import optional_admission; "
        "optional_admission('cvqa-hdr-nr').__enter__(); os._exit(9)",
    ]
    subprocess.Popen(crash, env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
    time.sleep(0.5)
    lease = acquire("cvqa-hdr-nr", timeout_s=15)
    release(lease)
    print("OK   crash-release")


def test_vram_budget_five_cvqa():
    if not admission_daemon_available():
        print("SKIP vram-five-daemon")
        return
    os.environ["PIXVQA_SCHEDULER_DIR"] = tempfile.mkdtemp(prefix="pixvqa-adm-test-")
    ledger = AdmissionLedger(
        budget=SchedulerBudget(gpu_headroom_mib=512, mem_available_floor_mib=256),
        profiles={
            "cvqa-hdr-nr": MetricProfile(
                metric="cvqa-hdr-nr",
                execution="independent",
                max_concurrent=5,
                ram_reserve_mib=100,
                vram_reserve_mib=2800,
                module="pixvqa_cvqahdr",
                verified_concurrency="test",
            )
        },
    )

    def fake_gpu(_=None):
        return 0, 16_384

    def fake_mem():
        return 32_768.0

    with patch("pixvqa_scheduler.admission_ledger.gpu_memory_mib", side_effect=fake_gpu), patch(
        "pixvqa_scheduler.admission_ledger.mem_available_mib", side_effect=fake_mem
    ):
        profile = ledger.resolve_profile("cvqa-hdr-nr")
        for _ in range(5):
            assert ledger.can_admit(profile, leases={}) is None
            ledger.reserve(profile, leases={})
        assert ledger.can_admit(profile, leases={}) is not None
    print("OK   vram-five-daemon")


def test_skip_env_no_double_acquire():
    script = """
import os, sys
sys.path.insert(0, %r)
os.environ["PIXVQA_SCHEDULER_ADMISSION"] = "1"
os.environ["PIXVQA_SCHEDULER_ADMISSION_SKIP"] = "1"
from pixvqa_scheduler.admission import admission_skip, admission_enabled
assert admission_enabled() and admission_skip()
print("skipped")
""" % (str(ROOT / "python"),)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    print("OK   admission-skip")


def test_cli_status():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    proc = subprocess.run(
        [sys.executable, "-m", "pixvqa_scheduler", "daemon", "--status"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "socket" in payload
    print("OK   cli-daemon-status")


def main() -> int:
    test_skip_env_no_double_acquire()
    test_vram_budget_five_cvqa()
    test_cli_status()
    test_daemon_probe()
    test_parallel_clients()
    test_crash_releases_slot()
    print("PASS: global admission tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
