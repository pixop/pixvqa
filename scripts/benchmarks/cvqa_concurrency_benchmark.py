#!/usr/bin/env python3
"""Guarded CompressedVQA-HDR-NR concurrency benchmark; no project-file modifications.

Stages: 1, 2, 3, 5 parallel jobs; then 5 jobs with PixVQA cross-process admission.
Runs only if preflight finds sufficient resources. Stops only child jobs this harness launched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

def mem_available_mib():
    with open("/proc/meminfo", encoding="ascii") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("Cannot read MemAvailable")

def swap_used_mib():
    vals = {}
    with open("/proc/meminfo", encoding="ascii") as f:
        for line in f:
            if line.startswith(("SwapTotal:", "SwapFree:")):
                vals[line.split(":")[0]] = int(line.split()[1]) / 1024.0
    return vals["SwapTotal"] - vals["SwapFree"]

def gpu_metrics():
    smi = shutil.which("nvidia-smi")
    if not smi and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        smi = "/usr/lib/wsl/lib/nvidia-smi"
    if not smi:
        raise RuntimeError("nvidia-smi unavailable; refusing unsafe GPU benchmark")
    output = subprocess.check_output(
        [smi, "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10
    ).strip().splitlines()[0]
    vals = [int(v.strip()) for v in output.split(",")]
    return dict(used_mib=vals[0], total_mib=vals[1], utilization_percent=vals[2])

def popen_job(cmd, env, output_dir, name):
    out = (output_dir / f"{name}.stdout.log").open("w", encoding="utf8")
    err = (output_dir / f"{name}.stderr.log").open("w", encoding="utf8")
    try:
        p = subprocess.Popen(cmd, env=env, stdout=out, stderr=err,
                             start_new_session=True)
    finally:
        out.close()
        err.close()
    return p

def stop_children(jobs):
    for j in jobs:
        p = j["process"]
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for j in jobs:
        p = j["process"]
        if p.poll() is None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()

def recursive_job_pids(pid):
    """Discover descendants for evidence of subprocess model instances (not CUDA ownership)."""
    found = {pid}
    for _ in range(8):
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, timeout=5)
            edges = []
            for row in result.stdout.splitlines():
                fields = row.split()
                if len(fields) == 2:
                    edges.append((int(fields[0]), int(fields[1])))
            more = {child for child, parent in edges if parent in found}
            if more.issubset(found):
                break
            found.update(more)
        except Exception:
            break
    return sorted(found)

def execute_stage(args, count, admitted, folder):
    stage = folder / f"{'admission_' if admitted else ''}{count}_jobs"
    stage.mkdir(parents=True, exist_ok=True)
    jobs = []
    samples = []
    stage_start = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.worktree / "python") + os.pathsep + env.get("PYTHONPATH", "")
    # Separate scheduler namespace so this test never changes other experiments' leases.
    if admitted:
        env["PIXVQA_SCHEDULER_ADMISSION"] = "1"
        env["PIXVQA_SCHEDULER_DIR"] = str(folder / "admission_namespace")
    else:
        env.pop("PIXVQA_SCHEDULER_ADMISSION", None)
        env.pop("PIXVQA_SCHEDULER_ADMISSION_SKIP", None)

    initial = gpu_metrics()
    initial_mem = mem_available_mib()
    if (initial["total_mib"] - initial["used_mib"] < args.gpu_headroom_mib
            or initial_mem < args.ram_floor_mib + 2000):
        return {"count": count, "admission": admitted, "status": "preflight_rejected",
                "gpu": initial, "mem_available_mib": initial_mem}

    try:
        for i in range(count):
            name = f"job_{i+1}"
            output_json = stage / f"{name}.json"
            cmd = [
                str(args.python), "-m", "pixvqa_cvqahdr",
                "--test", str(args.clip), "--repo", str(args.repo),
                "--checkpoint", str(args.checkpoint), "--profile", str(args.profile),
                "--device", "cuda", "--output-json", str(output_json),
                "--transient-output",
            ]
            p = popen_job(cmd, env, stage, name)
            jobs.append({"name": name, "pid": p.pid, "process": p,
                         "started": time.monotonic(), "output_json": output_json})
        while True:
            gm = gpu_metrics()
            ma = mem_available_mib()
            sw = swap_used_mib()
            now = time.monotonic()
            samples.append({"elapsed_s": round(now-stage_start, 2), **gm,
                            "mem_available_mib": round(ma, 1),
                            "swap_used_mib": round(sw, 1),
                            "live_pids": [j["pid"] for j in jobs
                                          if j["process"].poll() is None]})
            violation = None
            if gm["total_mib"] - gm["used_mib"] < args.gpu_headroom_mib:
                violation = "VRAM headroom threshold reached"
            elif ma < args.ram_floor_mib:
                violation = "MemAvailable floor reached"
            elif sw > args.max_swap_mib:
                violation = "Swap usage threshold reached"
            elif now - stage_start > args.timeout_s:
                violation = "Stage timed out"
            if violation:
                stop_children(jobs)
                state = "safety_stopped"
                break
            if all(j["process"].poll() is not None for j in jobs):
                violation = None
                state = "finished"
                break
            time.sleep(args.poll_s)
    except (KeyboardInterrupt, Exception):
        stop_children(jobs)
        raise

    finished = time.monotonic()
    results = []
    for j in jobs:
        p = j["process"]
        score = None
        if j["output_json"].exists():
            try:
                record = json.loads(j["output_json"].read_text())
                # Preserve full output, rather than assuming the schema of the score field.
                if isinstance(record, dict):
                    for key in ("score", "mos", "quality_score", "value"):
                        if isinstance(record.get(key), (int, float)):
                            score = float(record[key])
                            break
            except (ValueError, OSError):
                pass
        results.append({"name": j["name"], "pid": j["pid"], "returncode": p.poll(),
                        "score_if_top_level": score,
                        "json_file": str(j["output_json"]),
                        "elapsed_until_stage_end_s": round(finished-j["started"], 2)})
    result = {
        "count": count, "admission": admitted, "status": state,
        "stop_reason": violation, "duration_s": round(finished-stage_start, 2),
        "gpu_peak_used_mib": max(s["used_mib"] for s in samples),
        "min_mem_available_mib": min(s["mem_available_mib"] for s in samples),
        "max_swap_used_mib": max(s["swap_used_mib"] for s in samples),
        "jobs": results, "samples": samples,
        "caveats": [
            "nvidia-smi memory is GPU-global; it does not prove per-PID CUDA allocations",
            "Processes started simultaneously need not perform GPU inference simultaneously",
            "Per-job end time is not sampled separately; result includes only stage-end elapsed",
        ],
    }
    (stage / "summary.json").write_text(json.dumps(result, indent=2))
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", type=Path, default=Path("/home/sergio/pixvqa-pr22-wt"))
    p.add_argument("--python", type=Path, default=Path("/home/sergio/pixvqa-compressedvqa-env/bin/python"))
    p.add_argument("--clip", type=Path, default=Path("/home/sergio/pixvqa-pr22-wt/reports/pr22_validation_20261109/clips/BasketballDrive_1080p50_60s_loop6x_streamcopy.mp4"))
    p.add_argument("--repo", type=Path, default=Path("/home/sergio/pixvqa-pr22-wt/deps/CompressedVQA-HDR"))
    p.add_argument("--checkpoint", type=Path, default=Path("/home/sergio/pixvqa-pr22-wt/deps/CompressedVQA-HDR/NR_HDR_VQA.pth"))
    p.add_argument("--profile", type=Path, default=Path("/home/sergio/pixvqa-pr22-wt/deps/CompressedVQA-HDR/NR_HDR_VQA.npy"))
    p.add_argument("--output-dir", type=Path, default=Path("/home/sergio/pixvqa-pr22-wt/reports/pr22_validation_20261109/cvqa_concurrency"))
    p.add_argument("--timeout-s", type=float, default=900)
    p.add_argument("--poll-s", type=float, default=0.75)
    p.add_argument("--ram-floor-mib", type=float, default=2048)
    p.add_argument("--gpu-headroom-mib", type=float, default=768)
    p.add_argument("--max-swap-mib", type=float, default=1024)
    args = p.parse_args()

    for f in (args.python, args.clip, args.repo / "NR" / "NR_model.py",
              args.checkpoint, args.profile):
        if not f.exists():
            p.error(f"Required input not found: {f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final = {"started_utc": datetime.now(timezone.utc).isoformat(),
             "command": sys.argv, "stages": []}
    for count, admitted in [(1, False), (2, False), (3, False),
                            (5, False), (5, True)]:
        stage = execute_stage(args, count, admitted, args.output_dir)
        final["stages"].append(stage)
        (args.output_dir / "overall.json").write_text(json.dumps(final, indent=2))
        print(f"Stage count={count} admission={admitted}: {stage['status']}, "
              f"wall={stage.get('duration_s', 'n/a')}, "
              f"peak_vram={stage.get('gpu_peak_used_mib','n/a')}", flush=True)
        if stage["status"] != "finished":
            print("Stopped staged experiments after rejected/unsafe stage.", flush=True)
            break
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
