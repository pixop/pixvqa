#!/usr/bin/env python3
"""Run a PixVQA QA profile.

    python3 scripts/run_qa.py --profile smoke
    python3 scripts/run_qa.py --profile full
    python3 scripts/run_qa.py --profile hardware

Profiles write reports/qa/<profile>.json and reports/qa/<profile>.txt.
A skipped hardware test is not a pass. An optional missing tool is recorded
and does not by itself fail the profile.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from qa.cases_cli import cases as cli_cases
from qa.cases_media import cases as media_cases
from qa.harness import ROOT, World, find_ffmpeg, find_vmaf_cli, find_vvdec, run_cmd
from qa.inventory import build_inventory


def selected(case, profile: str) -> bool:
    if profile == "smoke":
        return "smoke" in case.profiles
    if profile == "full":
        return "smoke" in case.profiles or "full" in case.profiles
    if profile == "hardware":
        return "hardware" in case.profiles
    raise SystemExit(f"unknown profile {profile}")


def version_line(path: Path) -> str:
    if not path.is_file():
        return "absent"
    try:
        proc = run_cmd([str(path), "--version"], timeout=15)
    except Exception as exc:
        return str(exc)
    return (proc.stdout or proc.stderr).strip().replace("\n", " | ")


def environment(world: World, profile: str) -> dict:
    ff = "absent"
    if world.ffmpeg:
        try:
            proc = run_cmd([world.ffmpeg, "-version"], timeout=20)
            ff = proc.stdout.splitlines()[0] if proc.stdout else proc.stderr[:200]
        except Exception as exc:
            ff = str(exc)
    gpu = "not queried"
    if profile == "hardware":
        try:
            proc = run_cmd(["nvidia-smi", "-L"], timeout=20)
            gpu = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
        except Exception as exc:
            gpu = str(exc)
    rev = run_cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"], timeout=15)
    return {
        "profile": profile,
        "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": rev.stdout.strip() if rev.returncode == 0 else "",
        "cpu_binary": version_line(world.cpu),
        "swdec_binary": version_line(world.swdec),
        "full_binary": version_line(world.full),
        "ffmpeg": ff,
        "vvdec": world.vvdec or "absent",
        "vmaf_cli": str(world.vmaf_cli) if world.vmaf_cli else "absent",
        "model": str(world.model),
        "gpu": gpu,
    }


def coverage(results: list) -> dict:
    executed = {}
    for row in results:
        if row["status"] != "PASS":
            continue
        for tag in row["tags"]:
            executed.setdefault(tag.split(":", 1)[0] if False else tag, 0)
            executed[tag] = executed.get(tag, 0) + 1
    return executed


def select_qa_executable(root: Path, override: str = "") -> Path:
    """Choose the executable a QA profile scores.

    Priority: explicit override, then the installed bin/pixvqa, then the
    development build/cpu/pixvqa. An installed tree has no build/ directory.
    """
    chosen = override.strip()
    if chosen:
        path = Path(chosen)
        if not path.is_file():
            raise FileNotFoundError(f"QA executable override is not a file: {path}")
        return path
    installed = root / "bin" / "pixvqa"
    if installed.is_file():
        return installed
    developed = root / "build" / "cpu" / "pixvqa"
    if developed.is_file():
        return developed
    raise FileNotFoundError(
        "No PixVQA executable found. Expected bin/pixvqa after installation "
        "or build/cpu/pixvqa during development. Pass --cpu or set PIXVQA_QA_CPU."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PixVQA QA profiles")
    parser.add_argument("--profile", required=True, choices=("smoke", "full", "hardware"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "qa")
    parser.add_argument(
        "--cpu",
        type=Path,
        help="Executable to test. Overrides PIXVQA_QA_CPU. "
        "Default is bin/pixvqa, then build/cpu/pixvqa.",
    )
    args = parser.parse_args()

    inventory = build_inventory(ROOT)
    if inventory["missing_needles"]:
        print("inventory needles missing:", ", ".join(inventory["missing_needles"]), file=sys.stderr)
        return 2

    all_cases = cli_cases() + media_cases()
    chosen = [case for case in all_cases if selected(case, args.profile)]
    ids = [case.id for case in chosen]
    if len(ids) != len(set(ids)):
        print("duplicate case id", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix=f"pixvqa_qa_{args.profile}_") as tmp:
        override = str(args.cpu) if args.cpu else os.environ.get("PIXVQA_QA_CPU", "")
        try:
            cpu_binary = select_qa_executable(ROOT, override)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"QA executable: {cpu_binary}", flush=True)
        world = World(
            root=ROOT,
            tmp=Path(tmp),
            cpu=cpu_binary,
            swdec=ROOT / "build" / "swdec" / "pixvqa",
            full=ROOT / "build" / "full" / "pixvqa",
            ffmpeg=find_ffmpeg(),
            vvdec=find_vvdec(),
            model=ROOT / "model" / "vmaf_v1.0.16_3d0h.json",
            vmaf_cli=find_vmaf_cli(ROOT),
        )
        env = environment(world, args.profile)
        if args.profile == "hardware" and ("NVIDIA" not in env["gpu"] and "GPU" not in env["gpu"]):
            results.append({
                "id": "hardware-host",
                "group": "hardware",
                "status": "SKIP",
                "optional": False,
                "detail": env["gpu"],
                "command": "nvidia-smi -L",
                "tags": ["hardware-host"],
            })
        else:
            for index, case in enumerate(chosen, 1):
                print(f"[{index}/{len(chosen)}] {case.id}", flush=True)
                try:
                    result = case.func(world)
                except Exception as exc:
                    result_status = "FAIL"
                    detail = f"{type(exc).__name__}: {exc}"
                    if "Timeout" in type(exc).__name__:
                        detail = "timeout"
                    else:
                        detail = traceback.format_exc(limit=4)
                    from qa.harness import Result
                    result = Result(result_status, detail, case.id)
                results.append({
                    "id": case.id,
                    "group": case.group,
                    "status": result.status,
                    "optional": case.optional,
                    "detail": result.detail,
                    "command": result.command,
                    "tags": list(case.tags),
                })
                print(f"    {result.status} {result.detail[:180]}", flush=True)

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    mandatory_skip = [row for row in results if row["status"] == "SKIP" and not row["optional"]]
    optional_skip = [row for row in results if row["status"] == "SKIP" and row["optional"]]
    failures = [row for row in results if row["status"] == "FAIL"]
    if failures:
        verdict = "FAIL"
    elif mandatory_skip:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"

    gaps = [
        "VVC coverage is the tested 10-bit 4:2:0 clips, not a conformance suite.",
        "AV1 fixtures do not carry a film-grain header.",
        "A mid-stream resolution change is not generated.",
        "Concurrent GPU processes are not launched.",
        "The default profiles do not include a long stress run.",
        "8-bit AV1 NVDEC is compared after the host NV12 deinterleave.",
    ]
    for row in optional_skip:
        gaps.append(f"optional skip {row['id']}: {row['detail']}")

    report = {
        "verdict": verdict,
        "counts": counts,
        "selected": len(chosen) if not (args.profile == "hardware" and results and results[0]["id"] == "hardware-host") else len(results),
        "executed": len(results),
        "environment": env,
        "coverage": coverage(results),
        "gaps": gaps,
        "failures": [{"id": row["id"], "detail": row["detail"]} for row in failures],
        "skips": [{"id": row["id"], "optional": row["optional"], "detail": row["detail"]} for row in results if row["status"] == "SKIP"],
        "inventory": inventory,
        "cases": results,
    }
    json_path = args.output_dir / f"{args.profile}.json"
    text_path = args.output_dir / f"{args.profile}.txt"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"PixVQA QA {args.profile}: {verdict}",
        f"PASS {counts.get('PASS', 0)}  FAIL {counts.get('FAIL', 0)}  SKIP {counts.get('SKIP', 0)}",
        f"git {env.get('git', '')}",
        f"cpu {env['cpu_binary']}",
        f"swdec {env['swdec_binary']}",
        f"full {env['full_binary']}",
        f"ffmpeg {env['ffmpeg']}",
        f"gpu {env['gpu']}",
        "",
        "Failures:",
    ]
    lines.extend(f"- {row['id']}: {row['detail'][:400]}" for row in failures)
    if not failures:
        lines.append("- none")
    lines.append("")
    lines.append("Skips:")
    lines.extend(f"- {row['id']} optional={row['optional']}: {row['detail'][:300]}" for row in results if row["status"] == "SKIP")
    if not any(row["status"] == "SKIP" for row in results):
        lines.append("- none")
    lines.append("")
    lines.append("Coverage gaps:")
    lines.extend(f"- {gap}" for gap in gaps)
    lines.append("")
    lines.append("Reproduce:")
    lines.append(f"python3 scripts/run_qa.py --profile {args.profile}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[0])
    print(lines[1])
    print(f"wrote {json_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
