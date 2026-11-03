#!/usr/bin/env python3
"""Run declarative SDR/PQ/HLG CLI conformance matrix (negative + positive edge)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metric_cli_cases import CliConformanceCase, conformance_matrix  # noqa: E402
from metric_test_fixtures import FixtureContext, base_env, verify_cvvdp_mock_repeatable  # noqa: E402

DEFAULT_CPU = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
DEFAULT_SWDEC = Path(os.environ.get("PIXVQA_DECODE_BIN", ROOT / "build" / "swdec" / "pixvqa"))
FFMPEG = os.environ.get("PIXVQA_TEST_FFMPEG") or None


def git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resolve_ffmpeg() -> str | None:
    if FFMPEG:
        return FFMPEG
    import shutil

    return shutil.which("ffmpeg")


def bin_runnable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        proc = subprocess.run(
            [str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return proc.returncode in (0, 1) and ("Pix" in proc.stdout + proc.stderr or "Usage:" in proc.stdout)
    except (OSError, subprocess.TimeoutExpired):
        return False


def infrastructure_skip(case: CliConformanceCase, cpu: Path, swdec: Path, ffmpeg: str | None) -> str | None:
    bin_path = swdec if case.bin == "swdec" else cpu
    if not bin_path.is_file():
        return f"missing binary {bin_path}"
    if not bin_runnable(bin_path):
        return f"binary not runnable on this host ({bin_path})"
    if "ffmpeg" in case.needs and not ffmpeg:
        return "ffmpeg not found"
    return None


def stderr_matches(text: str, needles: tuple[str, ...]) -> str | None:
    for needle in needles:
        if needle not in text:
            return f"stderr missing {needle!r}"
    return None


def stdout_forbidden_hits(text: str, forbidden: tuple[str, ...]) -> str | None:
    for needle in forbidden:
        if needle in text:
            return f"stdout must not contain {needle!r}"
    return None


def evaluate_case(
    case: CliConformanceCase,
    ctx: FixtureContext,
    bin_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    cmd = [str(bin_path), *ctx.expand_args(case.args)]
    detail = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90)
    except subprocess.TimeoutExpired:
        return _result(case, cmd, "FAIL", -1, "process timed out", "hang")

    combined_err = proc.stderr + proc.stdout
    if case.outcome == "reject":
        if not case.rc_nonzero and proc.returncode == 0:
            pass
        elif case.rc_nonzero and proc.returncode == 0:
            detail = f"expected nonzero exit, got 0\nstdout={proc.stdout[-300:]}\nstderr={proc.stderr[-300:]}"
            return _result(case, cmd, "FAIL", proc.returncode, detail, "reject-nonzero")
        elif case.rc_nonzero and proc.returncode not in case.allowed_rc and proc.returncode != 0:
            detail = f"exit {proc.returncode} not in {case.allowed_rc}"
            return _result(case, cmd, "FAIL", proc.returncode, detail, "reject-rc")
        miss = stderr_matches(combined_err, case.stderr_contains)
        if miss:
            detail = f"{miss}\nstderr={proc.stderr[-400:]}"
            return _result(case, cmd, "FAIL", proc.returncode, detail, "reject-message")
        bad_out = stdout_forbidden_hits(proc.stdout, case.stdout_forbidden)
        if bad_out:
            detail = f"{bad_out}\nstdout={proc.stdout[-400:]}"
            return _result(case, cmd, "FAIL", proc.returncode, detail, "reject-clean-stdout")
        if "Traceback" in proc.stderr or "Traceback" in proc.stdout:
            detail = "Python traceback leaked to output"
            return _result(case, cmd, "FAIL", proc.returncode, detail, "traceback")
        return _result(case, cmd, "PASS", proc.returncode, case.summary, "reject")

    # accept
    if proc.returncode != 0:
        detail = f"expected success, rc={proc.returncode}\nstderr={proc.stderr[-400:]}"
        return _result(case, cmd, "FAIL", proc.returncode, detail, "accept-rc")
    miss = stderr_matches(combined_err, case.stderr_contains) if case.stderr_contains else None
    if miss:
        detail = f"{miss}\nstderr={proc.stderr[-400:]}"
        return _result(case, cmd, "FAIL", proc.returncode, detail, "accept-message")
    for needle in case.stdout_contains:
        if needle not in proc.stdout:
            detail = f"stdout missing {needle!r}\nstdout={proc.stdout[-400:]}"
            return _result(case, cmd, "FAIL", proc.returncode, detail, "accept-stdout")
    return _result(case, cmd, "PASS", proc.returncode, case.summary, "accept")


def _result(
    case: CliConformanceCase,
    cmd: list[str],
    status: str,
    rc: int,
    detail: str,
    phase: str,
) -> dict[str, Any]:
    return {
        "id": case.id,
        "category": case.category,
        "outcome": case.outcome,
        "status": status,
        "expected": case.summary,
        "phase": phase,
        "returncode": rc,
        "command": cmd,
        "detail": detail,
    }


def _fail_row(case: CliConformanceCase, detail: str, phase: str = "harness") -> dict[str, Any]:
    return {
        "id": case.id,
        "category": case.category,
        "outcome": case.outcome,
        "status": "FAIL",
        "expected": case.summary,
        "phase": phase,
        "returncode": -1,
        "command": [],
        "detail": detail,
    }


def run_cli_conformance_suite(
    *,
    cpu_bin: Path | None = None,
    swdec_bin: Path | None = None,
    json_report: Path | None = None,
    verbose: bool = True,
    strict: bool = False,
) -> int:
    cpu = cpu_bin or DEFAULT_CPU
    swdec = swdec_bin or DEFAULT_SWDEC
    ffmpeg = resolve_ffmpeg()

    results: list[dict[str, Any]] = []
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}

    try:
        verify_cvvdp_mock_repeatable()
        if verbose:
            print("OK   harness-cvvdp-mock-repeatable")
    except Exception as exc:
        if verbose:
            print(f"FAIL harness-cvvdp-mock-repeatable: {exc}", file=sys.stderr)
        counts["FAIL"] += 1
        results.append(
            {
                "id": "harness-cvvdp-mock-repeatable",
                "category": "harness",
                "outcome": "accept",
                "status": "FAIL",
                "expected": "CVVDP mock install is repeatable in one temp directory",
                "phase": "harness",
                "detail": f"{exc}\n{traceback.format_exc()}",
                "command": [],
            }
        )

    with tempfile.TemporaryDirectory(prefix="pixvqa_cli_conf_") as tmp:
        tmp_path = Path(tmp)
        try:
            ctx = FixtureContext(tmp_path, ffmpeg if ffmpeg else None)
        except Exception as exc:
            detail = f"fixture context: {exc}\n{traceback.format_exc()}"
            for case in conformance_matrix():
                results.append(_fail_row(case, detail, "fixture-context"))
                counts["FAIL"] += 1
                if verbose:
                    print(f"FAIL {case.id}: fixture context setup", file=sys.stderr)
        else:
            for case in conformance_matrix():
                row: dict[str, Any]
                try:
                    skip = infrastructure_skip(case, cpu, swdec, ffmpeg)
                    if skip:
                        counts["SKIP"] += 1
                        row = {
                            "id": case.id,
                            "category": case.category,
                            "outcome": case.outcome,
                            "status": "SKIP",
                            "expected": case.summary,
                            "detail": skip,
                            "command": [],
                        }
                        results.append(row)
                        if verbose:
                            print(f"SKIP {case.id} ({skip})")
                        continue

                    bin_path = swdec if case.bin == "swdec" else cpu
                    env = base_env(tmp_path, mock_cvvdp=case.mock_cvvdp)
                    row = evaluate_case(case, ctx, bin_path, env)
                except Exception as exc:
                    row = _fail_row(
                        case,
                        f"{exc}\n{traceback.format_exc()}",
                        "fixture-or-run",
                    )
                results.append(row)
                status = row.get("status", "FAIL")
                if status != "SKIP":
                    counts[status] = counts.get(status, 0) + 1
                if verbose:
                    if status == "PASS":
                        print(f"OK   {case.id}")
                    elif status == "FAIL":
                        print(f"FAIL {case.id}: {row.get('detail', '')[:200]}", file=sys.stderr)

    verdict = "PASS" if counts["FAIL"] == 0 and not (strict and counts["SKIP"]) else "FAIL"
    report = {
        "suite": "metric-cli-conformance",
        "verdict": verdict,
        "strict": strict,
        "commit": git_head(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "bins": {"cpu": str(cpu), "swdec": str(swdec), "ffmpeg": ffmpeg or ""},
        "counts": counts,
        "cases": results,
    }

    if json_report:
        json_report.parent.mkdir(parents=True, exist_ok=True)
        json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        print(
            f"\nCLI conformance: {verdict}  PASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}"
            + (" (strict)" if strict else "")
        )
        if json_report:
            print(f"JSON report: {json_report}")

    if counts["FAIL"]:
        return 1
    if strict and counts["SKIP"]:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PixVQA SDR/PQ/HLG CLI conformance matrix")
    parser.add_argument("--json", type=Path, help="Write JSON report path")
    parser.add_argument("--cpu-bin", type=Path, default=DEFAULT_CPU)
    parser.add_argument("--swdec-bin", type=Path, default=DEFAULT_SWDEC)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any case is skipped (release validation)",
    )
    args = parser.parse_args()
    strict = args.strict or os.environ.get("PIXVQA_CLI_CONFORMANCE_STRICT", "") in (
        "1",
        "true",
        "yes",
    )
    default_json = ROOT / "reports" / "metric_cli_conformance.json"
    report_path = args.json or default_json
    return run_cli_conformance_suite(
        cpu_bin=args.cpu_bin,
        swdec_bin=args.swdec_bin,
        json_report=report_path,
        verbose=not args.quiet,
        strict=strict,
    )


if __name__ == "__main__":
    raise SystemExit(main())
