#!/usr/bin/env python3
"""Regression tests for run_pixvqa_validation orchestration."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pixvqa_validation as orchestrator  # noqa: E402


def test_phase1_entrypoint_is_cli_conformance() -> None:
    assert orchestrator.CLI_CONFORMANCE_ENTRY.name == "test_metric_cli_conformance.py"
    assert orchestrator.PARAMETER_RESOLVER_ENTRY.name == "test_metric_parameters.py"
    assert orchestrator.CLI_CONFORMANCE_ENTRY != orchestrator.PARAMETER_RESOLVER_ENTRY


def test_summarize_cli_report_executed_counts() -> None:
    report = {"counts": {"PASS": 40, "FAIL": 2, "SKIP": 0}, "verdict": "FAIL"}
    summary = orchestrator.summarize_cli_report(report)
    assert summary["matrix_cases"] == 42
    assert summary["executed"] == 42
    assert summary["pass"] == 40
    assert summary["fail"] == 2
    assert summary["skip"] == 0
    assert summary["prerequisites_unavailable"] is False


def test_summarize_prerequisites_unavailable() -> None:
    report = {"counts": {"PASS": 0, "FAIL": 0, "SKIP": 42}, "verdict": "FAIL"}
    summary = orchestrator.summarize_cli_report(report)
    assert summary["executed"] == 0
    assert summary["prerequisites_unavailable"] is True


def test_orchestrator_uses_cli_conformance_phase() -> None:
    calls: list[str] = []

    def fake_cli(**kwargs):
        calls.append("cli")
        return 0, {"counts": {"PASS": 42, "FAIL": 0, "SKIP": 0}, "verdict": "PASS"}

    with mock.patch.object(orchestrator, "run_cli_conformance_phase", side_effect=fake_cli), mock.patch.object(
        orchestrator, "run_parameter_resolver_phase", return_value=0
    ) as resolver, mock.patch.object(orchestrator, "run_numerical_phase", return_value=0) as numerical:
        rc = orchestrator.orchestrate(
            skip_cli=False,
            skip_parameter_tests=True,
            skip_numerical=True,
            profile="smoke",
            strict=True,
            cli_json=ROOT / "reports" / "metric_cli_conformance.json",
            quiet=True,
        )
    assert rc == 0
    assert calls == ["cli"]
    resolver.assert_not_called()
    numerical.assert_not_called()


def test_cli_failure_propagates() -> None:
    with mock.patch.object(
        orchestrator,
        "run_cli_conformance_phase",
        return_value=(1, {"counts": {"PASS": 41, "FAIL": 1, "SKIP": 0}, "verdict": "FAIL"}),
    ), mock.patch.object(orchestrator, "run_numerical_phase") as numerical:
        rc = orchestrator.orchestrate(
            skip_cli=False,
            skip_parameter_tests=True,
            skip_numerical=False,
            profile="comprehensive",
            strict=True,
            cli_json=ROOT / "reports" / "metric_cli_conformance.json",
            quiet=True,
        )
    assert rc == 1
    numerical.assert_not_called()


def test_zero_executed_fails_even_when_subprocess_rc_zero() -> None:
    with mock.patch.object(
        orchestrator,
        "run_cli_conformance_phase",
        return_value=(0, {"counts": {"PASS": 0, "FAIL": 0, "SKIP": 42}, "verdict": "FAIL"}),
    ), mock.patch.object(orchestrator, "run_numerical_phase") as numerical:
        rc = orchestrator.orchestrate(
            skip_cli=False,
            skip_parameter_tests=True,
            skip_numerical=False,
            profile="comprehensive",
            strict=False,
            cli_json=ROOT / "reports" / "metric_cli_conformance.json",
            quiet=True,
        )
    assert rc == 1
    numerical.assert_not_called()


def test_parameter_resolver_runs_after_cli_by_default() -> None:
    order: list[str] = []

    def fake_cli(**kwargs):
        order.append("cli")
        return 0, {"counts": {"PASS": 42, "FAIL": 0, "SKIP": 0}, "verdict": "PASS"}

    def fake_resolver(**kwargs):
        order.append("resolver")
        return 0

    with mock.patch.object(orchestrator, "run_cli_conformance_phase", side_effect=fake_cli), mock.patch.object(
        orchestrator, "run_parameter_resolver_phase", side_effect=fake_resolver
    ), mock.patch.object(orchestrator, "run_numerical_phase", return_value=0):
        orchestrator.orchestrate(
            skip_cli=False,
            skip_parameter_tests=False,
            skip_numerical=True,
            profile="smoke",
            strict=True,
            cli_json=ROOT / "reports" / "metric_cli_conformance.json",
            quiet=True,
        )
    assert order == ["cli", "resolver"]


def main() -> int:
    tests = [
        test_phase1_entrypoint_is_cli_conformance,
        test_summarize_cli_report_executed_counts,
        test_summarize_prerequisites_unavailable,
        test_orchestrator_uses_cli_conformance_phase,
        test_cli_failure_propagates,
        test_zero_executed_fails_even_when_subprocess_rc_zero,
        test_parameter_resolver_runs_after_cli_by_default,
    ]
    for fn in tests:
        fn()
        print(f"OK   {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
