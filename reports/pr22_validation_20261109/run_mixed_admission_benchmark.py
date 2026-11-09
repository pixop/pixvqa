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
CVQA_PY = Path("/home/sergio/pixvqa-compressedvqa-env/bin/python")
CVQA_REPO = ROOT / "deps" / "CompressedVQA-HDR"
CVQA_CKPT = CVQA_REPO / "NR_HDR_VQA.pth"
CVQA_PROFILE = CVQA_REPO / "NR_HDR_VQA.npy"
REF_SIMPLEVQA_SCORE = 3.8700499534606934
REF_CVQA_SCORE = 8.395390956596998
PINNED_CVQA_COMMIT = "76bff6d5c078d51be06eddf1ed11591f3678145a"
TOL = 1e-4

NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")


def meminfo() -> dict[str, float]:
    out: dict[str, float] = {}
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            key = line.split(":")[0]
            if key in ("MemAvailable", "SwapTotal", "SwapFree"):
                out[key] = int(line.split()[1]) / 1024
    return out


def mem_available_mib() -> float:
    return meminfo().get("MemAvailable", 0.0)


def swap_used_mib() -> float:
    info = meminfo()
    return info.get("SwapTotal", 0.0) - info.get("SwapFree", 0.0)


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
        self.max_swap = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            g = gpu_used_mib()
            if g > self.max_gpu:
                self.max_gpu = g
            m = mem_available_mib()
            if m < self.min_mem:
                self.min_mem = m
            s = swap_used_mib()
            if s > self.max_swap:
                self.max_swap = s
            self._stop.wait(0.25)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def discover_cvqa() -> dict | None:
    bases = [
        CVQA_REPO,
        Path("/mnt/d/projects/pixvqa-public/deps/CompressedVQA-HDR"),
    ]
    for base in bases:
        ckpt = base / "NR_HDR_VQA.pth"
        profile = base / "NR_HDR_VQA.npy"
        if base.is_dir() and ckpt.is_file() and profile.is_file():
            py = CVQA_PY if CVQA_PY.is_file() else Path(shutil_which("python3"))
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
        "parity_ok": abs(score - REF_SIMPLEVQA_SCORE) <= TOL,
        "wall_s": round(wall, 2),
    }


def run_cvqa(cvqa: dict, poller: Poller, out: Path, ref_score: float) -> dict:
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
    score = float(payload["quality_score"])
    return {
        "status": "ok",
        "score": score,
        "parity_ok": abs(score - ref_score) <= TOL,
        "wall_s": round(wall, 2),
    }


def run_cvqa_baseline(cvqa: dict) -> dict:
    """Sequential CVQA baseline without admission contention."""
    env = os.environ.copy()
    env.pop("PIXVQA_SCHEDULER_ADMISSION", None)
    env["PYTHONPATH"] = str(ROOT / "python") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    with tempfile.NamedTemporaryFile(prefix="cvqa-base-", suffix=".json", delete=False) as tmp:
        out = Path(tmp.name)
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
    out.unlink(missing_ok=True)
    if proc.returncode != 0:
        return {"status": "failed", "rc": proc.returncode, "stderr": proc.stderr[-800:]}
    payload = json.loads(proc.stdout)
    score = float(payload["quality_score"])
    return {"status": "ok", "score": score, "wall_s": round(wall, 2)}


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
            "CompressedVQA-HDR-NR: no deps/CompressedVQA-HDR with NR_HDR_VQA.pth "
            "and NR_HDR_VQA.npy"
        )
    if not CVQA_PY.is_file():
        blockers.append(f"missing python {CVQA_PY}")

    report = {
        "clip": str(CLIP),
        "branch_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cvqa_repo_commit": PINNED_CVQA_COMMIT if cvqa else None,
        "reference_simplevqa_score": REF_SIMPLEVQA_SCORE,
        "reference_cvqa_score": REF_CVQA_SCORE,
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

    baseline = run_cvqa_baseline(cvqa)
    report["cvqa_baseline"] = baseline
    if baseline.get("status") != "ok":
        report["blockers"].append(f"CVQA baseline failed: rc={baseline.get('rc')}")
        out_path = REPORT_DIR / "mixed_admission_benchmark.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1
    report["reference_cvqa_score"] = baseline["score"]

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
                    {
                        "metric": "cvqa-hdr-nr",
                        **run_cvqa(cvqa, poller, tmp_path / "cv.json", baseline["score"]),
                    }
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
        "max_swap_used_mib": round(poller.max_swap, 1),
        "total_wall_s": round(time.monotonic() - t0, 2),
        "admission_env": "PIXVQA_SCHEDULER_ADMISSION=1",
        "note": "GPU used is device-global (nvidia-smi), not PID-exclusive.",
    }
    out_path = REPORT_DIR / "mixed_admission_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    ok = all(j.get("status") == "ok" for j in results)
    sv = next(j for j in results if j["metric"] == "simplevqa-shared")
    if not sv.get("parity_ok"):
        print("WARN: SimpleVQA score parity drift", file=sys.stderr)
    cv = next((j for j in results if j["metric"] == "cvqa-hdr-nr"), None)
    if cv and not cv.get("parity_ok"):
        print("WARN: CVQA score parity drift vs baseline", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
