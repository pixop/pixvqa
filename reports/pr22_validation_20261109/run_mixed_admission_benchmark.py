#!/usr/bin/env python3
"""Mixed SimpleVQA (shared GPU) + CVQA-HDR-NR under global admission control."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
CLIP = REPORT_DIR / "clips" / "BasketballDrive_1080p50_60s_loop6x_streamcopy.mp4"
SIMPLEVQA_REPO = Path("/mnt/d/projects/pixvqa-public/deps/SimpleVQA")
SIMPLEVQA_CKPT = SIMPLEVQA_REPO / "ckpts" / "UGC_BVQA_model.pth"
SIMPLEVQA_PY = Path("/home/sergio/pixvqa-simplevqa-env/bin/python")
REF_SCORE = 3.8700499534606934
TOL = 1e-4

NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")


def mem_available_mib() -> float:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 0.0


def gpu_used_mib() -> int:
    if not NVIDIA_SMI.is_file():
        return -1
    try:
        out = subprocess.check_output(
            [str(NVIDIA_SMI), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return int(out.split(",")[0].strip())
    except (subprocess.CalledProcessError, ValueError, OSError):
        return -1


class Poller:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.max_gpu = 0
        self.min_mem = float("inf")
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            g = gpu_used_mib()
            if g > self.max_gpu:
                self.max_gpu = g
            m = mem_available_mib()
            if m < self.min_mem:
                self.min_mem = m
            self._stop.wait(0.25)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def discover_cvqa() -> dict | None:
    bases = [
        Path("/mnt/d/projects/pixvqa-public/deps/CompressedVQA-HDR"),
        ROOT / "deps" / "CompressedVQA-HDR",
    ]
    for base in bases:
        ckpt = base / "ckpts" / "NR_checkpoint.pth"
        profile = base / "profiles" / "NR_profile.json"
        if base.is_dir() and ckpt.is_file() and profile.is_file():
            py = shutil_which("python3")
            return {"repo": base, "checkpoint": ckpt, "profile": profile, "python": py}
    return None


def shutil_which(name: str) -> str:
    from shutil import which

    return which(name) or sys.executable


def run_simplevqa_shared(poller: Poller, out: Path) -> dict:
    env = os.environ.copy()
    env["PIXVQA_SCHEDULER_ADMISSION"] = "1"
    env["PIXVQA_SCHEDULER_DIR"] = str(REPORT_DIR / ".admission_daemon")
    env["PYTHONPATH"] = str(ROOT / "python") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [
        str(SIMPLEVQA_PY),
        "-m",
        "pixvqa_simplevqa",
        "--repo",
        str(SIMPLEVQA_REPO),
        "--checkpoint",
        str(SIMPLEVQA_CKPT),
        "--test",
        str(CLIP),
        "--device",
        "cuda",
        "--use-shared-gpu-worker",
        "--output-json",
        str(out),
        "--transient-output",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    wall = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "rc": proc.returncode, "stderr": proc.stderr[-800:]}
    payload = json.loads(proc.stdout)
    score = float(payload["quality_score"])
    return {
        "status": "ok",
        "score": score,
        "parity_ok": abs(score - REF_SCORE) <= TOL,
        "wall_s": round(wall, 2),
    }


def run_cvqa(cvqa: dict, poller: Poller, out: Path) -> dict:
    env = os.environ.copy()
    env["PIXVQA_SCHEDULER_ADMISSION"] = "1"
    env["PIXVQA_SCHEDULER_DIR"] = str(REPORT_DIR / ".admission_daemon")
    env["PYTHONPATH"] = str(ROOT / "python") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [
        str(cvqa["python"]),
        "-m",
        "pixvqa_cvqahdr",
        "--test",
        str(CLIP),
        "--repo",
        str(cvqa["repo"]),
        "--checkpoint",
        str(cvqa["checkpoint"]),
        "--profile",
        str(cvqa["profile"]),
        "--device",
        "cuda",
        "--output-json",
        str(out),
        "--transient-output",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    wall = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "rc": proc.returncode, "stderr": proc.stderr[-800:]}
    payload = json.loads(proc.stdout)
    return {
        "status": "ok",
        "score": payload.get("quality_score"),
        "wall_s": round(wall, 2),
    }


def main() -> int:
    blockers: list[str] = []
    if not CLIP.is_file():
        blockers.append(f"missing clip {CLIP}")
    if not SIMPLEVQA_REPO.is_dir() or not SIMPLEVQA_CKPT.is_file():
        blockers.append(f"missing SimpleVQA repo/checkpoint under {SIMPLEVQA_REPO}")
    if not SIMPLEVQA_PY.is_file():
        blockers.append(f"missing python {SIMPLEVQA_PY}")
    cvqa = discover_cvqa()
    if cvqa is None:
        blockers.append(
            "CompressedVQA-HDR-NR: no deps/CompressedVQA-HDR with ckpts/NR_checkpoint.pth "
            "and profiles/NR_profile.json"
        )

    report = {
        "clip": str(CLIP),
        "reference_simplevqa_score": REF_SCORE,
        "blockers": blockers,
        "telemetry": {},
        "jobs": [],
    }

    if blockers:
        out_path = REPORT_DIR / "mixed_admission_benchmark.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("BLOCKED:", "; ".join(blockers), file=sys.stderr)
        return 2

    poller = Poller()
    poller.start()
    t0 = time.monotonic()
    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="pixvqa-mixed-") as tmp:
        tmp_path = Path(tmp)
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    {"metric": "simplevqa-shared", **run_simplevqa_shared(poller, tmp_path / "sv.json")}
                ),
                name="svqa",
            ),
            threading.Thread(
                target=lambda: results.append(
                    {"metric": "cvqa-hdr-nr", **run_cvqa(cvqa, poller, tmp_path / "cv.json")}
                ),
                name="cvqa",
            ),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    poller.stop()
    report["jobs"] = results
    report["telemetry"] = {
        "max_gpu_used_mib": poller.max_gpu,
        "min_mem_available_mib": round(poller.min_mem, 1),
        "total_wall_s": round(time.monotonic() - t0, 2),
        "note": "GPU used is device-global (nvidia-smi), not PID-exclusive.",
    }
    out_path = REPORT_DIR / "mixed_admission_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    ok = all(j.get("status") == "ok" for j in results)
    sv = next(j for j in results if j["metric"] == "simplevqa-shared")
    if not sv.get("parity_ok"):
        print("WARN: SimpleVQA score parity drift", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
