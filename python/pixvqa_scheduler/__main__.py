"""Opt-in batch scheduler CLI — does not replace ``pixvqa`` C++ orchestration."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .commands import cvqa_hdr_nr_command, simplevqa_command
from .scheduler import JobSpec, MetricScheduler, SchedulerConfig, scheduler_enabled


def _result_to_dict(r) -> dict:
    return {
        "job_id": r.spec.job_id,
        "metric": r.spec.metric,
        "state": r.state.value,
        "returncode": r.returncode,
        "wall_seconds": round(r.wall_seconds, 3),
        "admission_wait_seconds": round(r.admission_wait_seconds, 3),
        "error": r.error,
        "shared_gpu": r.spec.shared_gpu,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="PixVQA resource-aware metric batch scheduler (opt-in)."
    )
    ap.add_argument(
        "--enable",
        action="store_true",
        help="Force scheduler on for this invocation (or set PIXVQA_SCHEDULER=1).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print admission plan only.")
    sub = ap.add_subparsers(dest="cmd")

    mix = sub.add_parser("mix-example", help="Example job mix from env (for integration tests).")
    mix.add_argument("--spec-json", type=Path, help="JSON list of {metric, command,...}")

    svqa = sub.add_parser("simplevqa-batch")
    svqa.add_argument("--repo", required=True, type=Path)
    svqa.add_argument("--checkpoint", required=True, type=Path)
    svqa.add_argument("--test", required=True, type=Path)
    svqa.add_argument("--count", type=int, default=1)
    svqa.add_argument("--shared-gpu", action="store_true")
    svqa.add_argument("--timeout", type=float, default=3600)

    daemon_p = sub.add_parser("daemon", help="Run the cross-process admission daemon (foreground).")
    daemon_p.add_argument("--status", action="store_true", help="Print daemon socket path and exit.")

    cvqa = sub.add_parser("cvqa-batch")
    cvqa.add_argument("--repo", required=True, type=Path)
    cvqa.add_argument("--checkpoint", required=True, type=Path)
    cvqa.add_argument("--profile", required=True, type=Path)
    cvqa.add_argument("--test", required=True, type=Path)
    cvqa.add_argument("--count", type=int, default=1)
    cvqa.add_argument("--timeout", type=float, default=3600)

    args = ap.parse_args(argv)
    if args.enable:
        os.environ["PIXVQA_SCHEDULER"] = "1"

    if not args.cmd:
        ap.print_help()
        return 2

    if args.cmd == "daemon":
        from .ipc import socket_address

        if args.status:
            print(json.dumps({"socket": socket_address()}, indent=2))
            return 0
        from .daemon import main as daemon_main

        return daemon_main()

    specs: list[JobSpec] = []
    if args.cmd == "mix-example":
        if not args.spec_json or not args.spec_json.is_file():
            print("mix-example requires --spec-json", file=sys.stderr)
            return 2
        raw = json.loads(args.spec_json.read_text(encoding="utf-8"))
        for entry in raw:
            specs.append(
                JobSpec(
                    metric=entry["metric"],
                    command=entry["command"],
                    shared_gpu=bool(entry.get("shared_gpu")),
                    timeout_s=entry.get("timeout_s"),
                )
            )
    elif args.cmd == "simplevqa-batch":
        for i in range(args.count):
            cmd = simplevqa_command(
                repo=args.repo,
                checkpoint=args.checkpoint,
                test=args.test,
                use_shared_gpu_worker=args.shared_gpu,
            )
            specs.append(
                JobSpec(
                    metric="simplevqa",
                    command=cmd,
                    shared_gpu=args.shared_gpu,
                    timeout_s=args.timeout,
                    metadata={"index": i},
                )
            )
    elif args.cmd == "cvqa-batch":
        for i in range(args.count):
            cmd = cvqa_hdr_nr_command(
                test=args.test,
                repo=args.repo,
                checkpoint=args.checkpoint,
                profile=args.profile,
            )
            specs.append(
                JobSpec(
                    metric="cvqa-hdr-nr",
                    command=cmd,
                    timeout_s=args.timeout,
                    metadata={"index": i},
                )
            )

    if args.dry_run:
        cfg = SchedulerConfig(enabled=scheduler_enabled() or args.enable)
        print(json.dumps({
            "scheduler_enabled": cfg.enabled,
            "jobs": len(specs),
            "warnings": [cfg.cross_invocation_warning],
        }, indent=2))
        return 0

    sched = MetricScheduler(SchedulerConfig(enabled=True))
    report = sched.run_batch(specs)
    out = {
        "scheduler_enabled": True,
        "jobs": [_result_to_dict(r) for r in report.jobs],
        "telemetry": report.telemetry,
        "warnings": report.warnings,
        "total_wall_seconds": round(report.total_wall_seconds, 3),
        "snapshot": sched.snapshot(),
    }
    print(json.dumps(out, indent=2))
    ok = all(j.state.value == "completed" for j in report.jobs)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
