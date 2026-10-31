#!/usr/bin/env python3
"""Score one clip pair with CVVDP and CVQA-HDR-NR on the CPU and on CUDA, then compare.

Runs pixvqa twice (--device cpu, --device gpu) with --qcs and persistent JSON
records, checks that each backend reports the device it was given, and compares
pooled scores, per-frame or per-sample predictions, and QCS against the
tolerances below. Writes report.json to --out. Exit 1 when a tolerance or a
device check fails. Linux and WSL2 only (it measures children with wait4).

  python3 scripts/validate_learned_devices.py --pixvqa build/swdec/pixvqa \\
      -r ref.mp4 -t dist.mp4 --out /tmp/pixvqa-devices
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Absolute CPU-vs-CUDA tolerances. The cause and the observed spread are in
# docs/LEARNED_DEVICES.md; CVVDP JOD is on its 0-10 scale, NR on its 0-10
# calibrated scale, QCS in percentage points.
TOLERANCES = {
    "cvvdp_jod": 1e-3,
    "cvvdp_frame_jod_equivalent": 5e-3,
    "cvvdp_frame_distortion": 5e-3,
    "cvvdp_qcs": 0.5,
    "nr_quality_score": 1e-3,
    "nr_raw_model_score": 1e-3,
    "nr_sample_quality_score": 2e-3,
    "nr_qcs": 0.5,
}

QCS_LINE = re.compile(r"\[ (CVVDP|CVQA-HDR-NR)-QCS value:(\S+)")


class GpuMemorySampler:
    """Device-wide memory.used from nvidia-smi; includes the CUDA context."""

    def __init__(self) -> None:
        self.tool = shutil.which("nvidia-smi") or (
            "/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
        self.peak = None
        self.baseline = self.read()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def read(self) -> int | None:
        if not self.tool:
            return None
        proc = subprocess.run([self.tool, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True)
        try:
            return int(proc.stdout.split()[0])
        except (IndexError, ValueError):
            return None

    def loop(self) -> None:
        while not self.stop.is_set():
            value = self.read()
            if value is not None:
                self.peak = value if self.peak is None else max(self.peak, value)
            self.stop.wait(0.2)

    def __enter__(self):
        if self.tool:
            self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        if self.tool:
            self.thread.join()

    def delta_mib(self) -> int | None:
        if self.peak is None or self.baseline is None:
            return None
        return self.peak - self.baseline


def run_pixvqa(args, device: str, out: Path, env: dict) -> dict:
    cvvdp_json, nr_json = out / f"{device}-cvvdp.json", out / f"{device}-nr.json"
    cmd = [args.pixvqa, "--metrics", "cvvdp,cvqa-hdr-nr", "--qcs", "--device", device,
           "-r", args.ref, "-t", args.test, "--cvvdp-interpretation", args.interpretation,
           "--cvvdp-display", args.display, "--cvvdp-output-json", str(cvvdp_json),
           "--cvqahdr-output-json", str(nr_json)]
    stdout_path, stderr_path = out / f"{device}.stdout", out / f"{device}.stderr"
    t0 = time.perf_counter()
    with GpuMemorySampler() as gpu, stdout_path.open("w") as so, stderr_path.open("w") as se:
        proc = subprocess.Popen(cmd, stdout=so, stderr=se, env=env)
        # wait4 reports the peak RSS over pixvqa and the backends it reaped.
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0
    stdout, stderr = stdout_path.read_text(encoding="utf-8"), stderr_path.read_text(encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"pixvqa --device {device} exited {proc.returncode}:\n{stderr[-3000:]}")
    qcs = {m.group(1): float(m.group(2)) for m in QCS_LINE.finditer(stdout) if m.group(2) != "unavailable"}
    return {
        "command": cmd, "wall_seconds": wall, "gpu_memory_used_delta_mib": gpu.delta_mib(),
        "summary": [line for line in stdout.splitlines() if line.startswith("[ ")],
        "qcs": qcs, "peak_rss_bytes_process_tree": usage.ru_maxrss * 1024,
        "cvvdp": json.loads(cvvdp_json.read_text(encoding="utf-8")),
        "nr": json.loads(nr_json.read_text(encoding="utf-8")),
    }


def max_abs(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def compare(cpu: dict, gpu: dict) -> dict:
    cc, gc, cn, gn = cpu["cvvdp"], gpu["cvvdp"], cpu["nr"], gpu["nr"]
    cts, gts = cc["temporal"]["samples"], gc["temporal"]["samples"]
    cns, gns = cn["temporal"]["samples"], gn["temporal"]["samples"]
    diffs = {
        "cvvdp_jod": abs(cc["jod"] - gc["jod"]),
        "cvvdp_frame_jod_equivalent": max_abs([s["jod_equivalent"] for s in cts],
                                              [s["jod_equivalent"] for s in gts]),
        "cvvdp_frame_distortion": max_abs([s["distortion"] for s in cts], [s["distortion"] for s in gts]),
        "cvvdp_qcs": abs(cpu["qcs"].get("CVVDP", float("nan")) - gpu["qcs"].get("CVVDP", float("nan"))),
        "nr_quality_score": abs(cn["quality_score"] - gn["quality_score"]),
        "nr_raw_model_score": abs(cn["raw_model_score"] - gn["raw_model_score"]),
        "nr_sample_quality_score": max_abs([s["quality_score"] for s in cns], [s["quality_score"] for s in gns]),
        "nr_qcs": abs(cpu["qcs"].get("CVQA-HDR-NR", float("nan")) -
                      gpu["qcs"].get("CVQA-HDR-NR", float("nan"))),
    }
    return {key: {"abs_diff": value, "tolerance": TOLERANCES[key], "pass": value <= TOLERANCES[key]}
            for key, value in diffs.items()}


def device_checks(result: dict, expected: str) -> list[str]:
    problems = []
    cv, nr = result["cvvdp"], result["nr"]
    if cv["device"] != expected or (cv.get("device_logged_by_cvvdp") or "").split(":")[0] != expected:
        problems.append(f"CVVDP ran on {cv['device']} (cvvdp log {cv.get('device_logged_by_cvvdp')})")
    if nr["device"] != expected:
        problems.append(f"NR ran on {nr['device']}")
    if not cv["temporal"]["pooled_parity"] or not nr["temporal"]["pooled_parity"]:
        problems.append("a pooled-parity check failed")
    if f"device:{expected} ]" not in "\n".join(result["summary"]):
        problems.append("summary lines do not report the device")
    return problems


def provenance(result: dict) -> dict:
    cv, nr = result["cvvdp"], result["nr"]
    return {
        "device_info": cv["device_info"],
        "cvvdp": {"commit": cv["runtime"]["cvvdp"]["actual_commit"],
                  "version": cv["runtime"]["cvvdp"]["version"], "display": cv["display"]},
        "nr": {"upstream": nr["upstream"], "checkpoint": nr["checkpoint"],
               "calibration_profile": {k: nr["calibration_profile"][k] for k in ("path", "sha256")}},
        "torch": nr["runtime"]["torch"],
    }


def performance(result: dict) -> dict:
    return {
        "pixvqa_wall_seconds": result["wall_seconds"],
        "peak_rss_bytes_largest_process": result["peak_rss_bytes_process_tree"],
        "gpu_memory_used_delta_mib": result["gpu_memory_used_delta_mib"],
        "cvvdp": {**result["cvvdp"]["performance"], "frames": result["cvvdp"]["temporal"]["frame_count"]},
        "nr": {**result["nr"]["performance"], "samples": len(result["nr"]["temporal"]["samples"])},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixvqa", required=True)
    parser.add_argument("-r", "--ref", required=True)
    parser.add_argument("-t", "--test", required=True)
    parser.add_argument("--interpretation", default="pq")
    parser.add_argument("--display", default="standard_hdr_pq")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    results = {device: run_pixvqa(args, device, out, env) for device in ("cpu", "gpu")}
    expected = {"cpu": "cpu", "gpu": "cuda"}
    problems = {device: device_checks(results[device], expected[device]) for device in results}
    comparison = compare(results["cpu"], results["gpu"])
    report = {
        "reference": args.ref, "test": args.test, "interpretation": args.interpretation,
        "display": args.display, "tolerances": TOLERANCES, "comparison": comparison,
        "scores": {device: {"cvvdp_jod": r["cvvdp"]["jod"], "nr_quality_score": r["nr"]["quality_score"],
                            "nr_raw_model_score": r["nr"]["raw_model_score"], "qcs": r["qcs"]}
                   for device, r in results.items()},
        "device_problems": problems,
        "provenance": {device: provenance(r) for device, r in results.items()},
        "performance": {device: performance(r) for device, r in results.items()},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, row in comparison.items():
        print(f"{'OK  ' if row['pass'] else 'FAIL'} {key}: |cpu-cuda| {row['abs_diff']:.3g} "
              f"(tolerance {row['tolerance']:g})")
    for device, items in problems.items():
        for item in items:
            print(f"FAIL device {device}: {item}")
    for device, perf in report["performance"].items():
        print(f"{device}: pixvqa {perf['pixvqa_wall_seconds']:.1f} s, CVVDP CLI "
              f"{perf['cvvdp']['official_cli_seconds']:.1f} s, NR inference "
              f"{perf['nr']['inference_seconds']:.2f} s, NR preprocessing "
              f"{perf['nr']['preprocessing_seconds']:.2f} s")
    print(f"report: {out / 'report.json'}")
    failed = any(not row["pass"] for row in comparison.values()) or any(problems.values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
