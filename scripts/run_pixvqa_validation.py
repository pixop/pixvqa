#!/usr/bin/env python3
"""Run CLI conformance (phase 1) and numerical conformance (phase 2)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metric_cli_cases import conformance_matrix  # noqa: E402

CLI_CONFORMANCE_ENTRY = ROOT / "scripts" / "test_metric_cli_conformance.py"
PARAMETER_RESOLVER_ENTRY = ROOT / "scripts" / "test_metric_parameters.py"
NUMERICAL_ENTRY = ROOT / "scripts" / "test_metric_numerical_conformance.py"


def strict_enabled(flag: bool) -> bool:
    return flag or os.environ.get("PIXVQA_VALIDATION_STRICT", "") in ("1", "true", "yes")


def run_cli_conformance_phase(
    *,
    strict: bool,
    json_report: Path,
    quiet: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Run the 42-case CLI matrix via metric_cli_conformance (not parameter resolver tests)."""
    from metric_cli_conformance import run_cli_conformance_suite

    prev = os.environ.get("PIXVQA_CLI_CONFORMANCE_STRICT")
    os.environ["PIXVQA_CLI_CONFORMANCE_STRICT"] = "1" if strict else "0"
    try:
        rc = run_cli_conformance_suite(
            json_report=json_report,
            verbose=not quiet,
            strict=strict,
        )
    finally:
        if prev is None:
            os.environ.pop("PIXVQA_CLI_CONFORMANCE_STRICT", None)
        else:
            os.environ["PIXVQA_CLI_CONFORMANCE_STRICT"] = prev

    if json_report.is_file():
        report = json.loads(json_report.read_text(encoding="utf-8"))
    else:
        report = {"counts": {}, "verdict": "FAIL"}
    return rc, report


def summarize_cli_report(report: dict[str, Any]) -> dict[str, Any]:
    counts = report.get("counts") or {}
    matrix_cases = len(conformance_matrix())
    passed = int(counts.get("PASS", 0))
    failed = int(counts.get("FAIL", 0))
    skipped = int(counts.get("SKIP", 0))
    executed = passed + failed
    return {
        "matrix_cases": matrix_cases,
        "executed": executed,
        "pass": passed,
        "fail": failed,
        "skip": skipped,
        "verdict": report.get("verdict"),
        "prerequisites_unavailable": executed == 0 and matrix_cases > 0,
    }


def print_cli_phase_summary(summary: dict[str, Any], *, strict: bool) -> None:
    print(
        f"\nPhase 1 CLI conformance: {summary.get('verdict', 'UNKNOWN')}  "
        f"matrix={summary['matrix_cases']} executed={summary['executed']} "
        f"PASS={summary['pass']} FAIL={summary['fail']} SKIP={summary['skip']}"
        + (" (strict)" if strict else "")
    )
    if summary["prerequisites_unavailable"]:
        print(
            "Phase 1: no matrix cases executed — missing or unrunnable pixvqa binaries "
            "or ffmpeg (prerequisite failure, not successful CLI conformance).",
            file=sys.stderr,
        )


def run_parameter_resolver_phase(*, python: str) -> int:
    """Focused HDR/CVVDP resolver tests; separate from the 42-case CLI matrix."""
    proc = subprocess.run([python, str(PARAMETER_RESOLVER_ENTRY)], cwd=ROOT)
    return proc.returncode


def run_numerical_phase(
    *,
    python: str,
    profile: str,
    strict: bool,
) -> int:
    cmd = [
        python,
        str(NUMERICAL_ENTRY),
        "--profile",
        profile,
    ]
    if strict:
        cmd.append("--strict")
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def orchestrate(
    *,
    skip_cli: bool,
    skip_parameter_tests: bool,
    skip_numerical: bool,
    profile: str,
    strict: bool,
    cli_json: Path,
    quiet: bool,
) -> int:
    rc = 0

    if not skip_cli:
        if not CLI_CONFORMANCE_ENTRY.is_file():
            print(f"FAIL missing Phase 1 entry point: {CLI_CONFORMANCE_ENTRY}", file=sys.stderr)
            return 1
        print("=== Phase 1: CLI conformance (42-case matrix) ===")
        phase_rc, report = run_cli_conformance_phase(
            strict=strict,
            json_report=cli_json,
            quiet=quiet,
        )
        summary = summarize_cli_report(report)
        if not quiet:
            print_cli_phase_summary(summary, strict=strict)
            if cli_json.is_file():
                print(f"Phase 1 JSON report: {cli_json}")
        if phase_rc != 0:
            rc = 1
        if summary["prerequisites_unavailable"]:
            rc = 1

    if not skip_parameter_tests and rc == 0:
        if not PARAMETER_RESOLVER_ENTRY.is_file():
            print(f"FAIL missing parameter resolver entry: {PARAMETER_RESOLVER_ENTRY}", file=sys.stderr)
            return 1
        print("\n=== Phase 1b: parameter resolver tests (not the 42-case matrix) ===")
        if run_parameter_resolver_phase(python=sys.executable) != 0:
            rc = 1

    if not skip_numerical and rc == 0:
        print(f"\n=== Phase 2: numerical conformance ({profile}) ===")
        if run_numerical_phase(python=sys.executable, profile=profile, strict=strict) != 0:
            rc = 1

    if rc == 0:
        print("\nValidation orchestrator: PASS")
    else:
        print("\nValidation orchestrator: FAIL", file=sys.stderr)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="PixVQA validation: CLI + numerical")
    parser.add_argument("--skip-cli", action="store_true", help="Skip Phase 1 CLI conformance matrix")
    parser.add_argument(
        "--skip-parameter-tests",
        action="store_true",
        help="Skip Phase 1b parameter resolver tests (test_metric_parameters.py)",
    )
    parser.add_argument("--skip-numerical", action="store_true")
    parser.add_argument("--profile", choices=("smoke", "comprehensive"), default="smoke")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--cli-json",
        type=Path,
        default=ROOT / "reports" / "metric_cli_conformance.json",
        help="JSON report for Phase 1 CLI conformance",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    return orchestrate(
        skip_cli=args.skip_cli,
        skip_parameter_tests=args.skip_parameter_tests,
        skip_numerical=args.skip_numerical,
        profile=args.profile,
        strict=strict_enabled(args.strict),
        cli_json=args.cli_json,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())
