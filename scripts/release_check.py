#!/usr/bin/env python3
"""Orchestrate PixVQA release validation without publishing anything."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "release"


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def execute(name: str, command: list[str], env: dict[str, str]) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORT_DIR / f"{name}.log"

    if name == "qa-full":
        qa_report = REPORT_DIR / "qa" / "full.json"
        qa_report.unlink(missing_ok=True)

    print(f"\nRUN  {name}", flush=True)
    started = time.monotonic()

    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        rc = result.returncode
        error = None
    except OSError as exc:
        rc = 1
        error = str(exc)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nExecution error: {exc}\n")

    elapsed = round(time.monotonic() - started, 2)
    status = "PASS" if rc == 0 else "FAIL"

    print(f"{status} {name} ({elapsed:.2f}s)", flush=True)

    if status == "FAIL":
        print(f"  Log: {log_path}", flush=True)
        if error:
            print(f"  Error: {error}", flush=True)

    return {
        "name": name,
        "status": status,
        "returncode": rc,
        "duration_seconds": elapsed,
        "command": command,
        "log": str(
            log_path.relative_to(ROOT)
            if log_path.is_relative_to(ROOT)
            else log_path
        ),
    }

def read_qa_results() -> dict:
    """Read structured results from the existing QA runner."""
    path = REPORT_DIR / "qa" / "full.json"

    if not path.is_file():
        raise ValueError(f"Missing QA report: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    counts = data["counts"]

    return {
        "status": data["verdict"],
        "selected": data["selected"],
        "executed": data["executed"],
        "passed": counts["PASS"],
        "failed": counts["FAIL"],
        "skipped": counts["SKIP"],
        "coverage_gaps": data.get("gaps", []),
    }


def read_gpu_results() -> dict:
    """Extract qualification counts and skipped checks from the GPU log."""
    import re

    path = REPORT_DIR / "gpu-qualification.log"

    if not path.is_file():
        raise ValueError(f"Missing GPU qualification log: {path}")

    output = path.read_text(encoding="utf-8")

    matches = re.findall(
        r"^QUALIFY checks=(\d+) fails=(\d+)$",
        output,
        flags=re.MULTILINE,
    )

    if len(matches) != 1:
        raise ValueError("GPU qualification summary missing or ambiguous")

    passed, failed = map(int, matches[0])

    skipped = [
        line.strip()
        for line in output.splitlines()
        if line.startswith("SKIP ")
    ]

    return {
        "status": (
            "FAIL" if failed else "INCOMPLETE" if skipped else "PASS"
        ),
        "passed": passed,
        "failed": failed,
        "skipped": len(skipped),
        "skipped_checks": skipped,
    }

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PixVQA release qualification"
    )
    parser.add_argument(
        "--profile",
        choices=("cpu", "full"),
        required=True,
    )
    args = parser.parse_args()

    env = os.environ.copy()
    python = sys.executable

    report = {
        "profile": args.profile,
        "commit": git_commit(),
        "started": datetime.now(timezone.utc).isoformat(),
        "stages": [],
    }

    def stage(name: str, command: list[str], extra_env=None) -> bool:
        stage_env = env.copy()
        if extra_env:
            stage_env.update(extra_env)

        result = execute(name, command, stage_env)
        report["stages"].append(result)
        return result["status"] == "PASS"

    try:
        builds = ["cpu", "swdec"]
        if args.profile == "full":
            builds.append("full")

        for variant in builds:
            if not stage(
                f"build-{variant}",
                ["make", "-j4", variant],
            ):
                break
        else:
            qa_ok = stage(
                "qa-full",
                [
                    python,
                    "scripts/run_qa.py",
                    "--profile", "full",
                    "--cpu", str(ROOT / "build/cpu/pixvqa"),
                    "--output-dir", str(REPORT_DIR / "qa"),
                ],
            )

            if qa_ok and args.profile == "full":
                stage(
                    "gpu-qualification",
                    [python, "scripts/qualify_gpu_release.py"],
                    {
                        "PIXVQA_BIN": str(ROOT / "build/full/pixvqa"),
                        "PIXVQA_CPU": str(ROOT / "build/cpu/pixvqa"),
                        "PIXVQA_SWDEC": str(ROOT / "build/swdec/pixvqa"),
                    },
                )
                for name, script in [
                    ("cuda-numeric", "scripts/test_cuda_numeric.py"),
                    ("cuda-psnr-ssim", "scripts/test_cuda_psnr_ssim_8bit.py"),
                    ("cuda-pu21", "scripts/test_pu21_cuda.py"),
                    ("cuda-xpsnr-wsse", "scripts/test_xpsnr_cuda_wsse.py"),
                ]:
                    if not stage(
                        name,
                        [python, script],
                        {
                            "PIXVQA_BIN": str(ROOT / "build/cpu/pixvqa"),
                            "PIXVQA_CUDA_BIN": str(ROOT / "build/full/pixvqa"),
                        },
                    ):
                        break
                else:
                    if stage(
                        "prepare-uhd-fixtures",
                        [python, "scripts/prepare_xpsnr_fixtures.py"],
                    ):
                        stage(
                            "cuda-xpsnr-jobs",
                            [python, "scripts/test_xpsnr_cuda_jobs.py"],
                        )

    finally:
        report["finished"] = datetime.now(timezone.utc).isoformat()
        report["qa"] = None
        report["gpu"] = None
        report_errors = []

        try:
            if any(s["name"] == "qa-full" for s in report["stages"]):
                report["qa"] = read_qa_results()

            if any(s["name"] == "gpu-qualification" for s in report["stages"]):
                report["gpu"] = read_gpu_results()

        except (OSError, ValueError, KeyError, TypeError) as exc:
            report_errors.append(str(exc))

        report["report_errors"] = report_errors

        expected_stages = [
            "build-cpu",
            "build-swdec",
        ]

        if args.profile == "full":
            expected_stages.append("build-full")

        expected_stages.append("qa-full")

        if args.profile == "full":
            expected_stages.extend([
                "gpu-qualification",
                "cuda-numeric",
                "cuda-psnr-ssim",
                "cuda-pu21",
                "cuda-xpsnr-wsse",
                "prepare-uhd-fixtures",
                "cuda-xpsnr-jobs",
            ])

        completed_stages = {
            stage["name"]
            for stage in report["stages"]
            if stage["status"] == "PASS"
        }

        stages_passed = (
            len(report["stages"]) == len(expected_stages)
            and completed_stages == set(expected_stages)
        )

        qa_complete = (
            report["qa"] is not None
            and report["qa"]["status"] == "PASS"
            and report["qa"]["failed"] == 0
            and report["qa"]["skipped"] == 0
            and report["qa"]["executed"] == report["qa"]["selected"]
        )

        gpu_complete = (
            args.profile == "cpu"
            or (
                report["gpu"] is not None
                and report["gpu"]["status"] == "PASS"
            )
        )

        if not stages_passed or report_errors:
            report["status"] = "FAIL"
        elif not qa_complete or not gpu_complete:
            report["status"] = "INCOMPLETE"
        else:
            report["status"] = "PASS"

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / "summary.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\nPixVQA Release Qualification")
        print("=" * 35)
        for item in report["stages"]:
            print(f"{item['status']:4} {item['name']}")

        print(f"\nFINAL RESULT: {report['status']}")
        print(f"Report: {report_path.relative_to(ROOT)}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
