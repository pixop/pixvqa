"""Execute numerical conformance matrix."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .baselines import compare_to_baseline, load_baseline, save_candidate
from .berlin import run_berlin_optional
from .cli_invoke import build_invoke_context, build_pixvqa_command, classify_failure
from .config import CHROMA, SIZE
from .environment import Environment, bin_runnable, discover
from .fixtures import ensure_all_families
from .metric_caps import GROUP_METRICS, expand_metrics, includes_learned
from .parse_metrics import parse_run
from .registry import Scenario, incompatible_reason, resolve_media_paths
from .tolerances import close

Profile = Literal["smoke", "comprehensive"]


@dataclass
class NumericalCase:
    id: str
    family: str
    scenario: Scenario
    metric: str
    mandatory: bool = True
    metric_device: str | None = None
    decode: str | None = None
    explicit_hdr_transfer: bool = False
    tags: tuple[str, ...] = ()


def build_cases(profile: Profile) -> list[NumericalCase]:
    families = ("sdr", "pq", "hlg")
    scenarios_smoke: tuple[Scenario, ...] = ("raw_raw", "raw_comp")
    scenarios_full: tuple[Scenario, ...] = ("raw_raw", "raw_comp", "comp_comp")
    scenarios = scenarios_full if profile == "comprehensive" else scenarios_smoke

    cases: list[NumericalCase] = []
    for family in families:
        for scenario in scenarios:
            if profile == "smoke":
                metrics = ("psnr", "ssim")
            else:
                metrics = tuple(GROUP_METRICS[family])
            for metric in metrics:
                cases.append(NumericalCase(f"{family}.{scenario}.{metric}", family, scenario, metric))
            if profile == "comprehensive":
                mandatory = not (scenario == "raw_raw" and family in ("sdr", "pq"))
                cases.append(
                    NumericalCase(
                        f"{family}.{scenario}.group_{family}",
                        family,
                        scenario,
                        f"group:{family}",
                        mandatory=mandatory,
                    )
                )
        cases.append(
            NumericalCase(
                f"{family}.consistency.psnr_raw_vs_comp",
                family,
                "raw_raw",
                "psnr",
                tags=("consistency",),
            )
        )
        if profile == "comprehensive":
            cases.append(
                NumericalCase(
                    f"{family}.hdr.inferred_vs_explicit",
                    family,
                    "raw_comp",
                    "psnr",
                    explicit_hdr_transfer=True,
                    tags=("hdr-equiv",),
                )
            )
            cases.append(
                NumericalCase(
                    f"{family}.backend.psnr_cpu",
                    family,
                    "raw_raw",
                    "psnr",
                    metric_device="cpu",
                    tags=("backend-cpu",),
                )
            )
            cases.append(
                NumericalCase(
                    f"{family}.backend.psnr_gpu",
                    family,
                    "raw_raw",
                    "psnr",
                    metric_device="gpu",
                    mandatory=False,
                    tags=("backend-gpu",),
                )
            )
            cases.append(
                NumericalCase(
                    f"{family}.decode.psnr_sw",
                    family,
                    "comp_comp",
                    "psnr",
                    decode="cpu",
                    tags=("decode-sw",),
                )
            )
            cases.append(
                NumericalCase(
                    f"{family}.decode.psnr_nvdec",
                    family,
                    "comp_comp",
                    "psnr",
                    decode="gpu",
                    mandatory=False,
                    tags=("decode-gpu",),
                )
            )
    return cases


def _validate_parsed(case: NumericalCase, parsed, invoke, proc) -> str | None:
    metric_name = case.metric.split(":", 1)[-1] if case.metric.startswith("group:") else case.metric
    names = expand_metrics(case.metric)

    if invoke.expected_frames is not None:
        if parsed.total_frames != invoke.expected_frames:
            return f"expected {invoke.expected_frames} frames, got {parsed.total_frames}"

    if "psnr" in names or case.metric.startswith("group:"):
        if parsed.psnr_avg is not None and not math.isfinite(parsed.psnr_avg):
            return "non-finite PSNR avg"
        if case.metric.startswith("group:") and parsed.psnr_avg is None:
            return "missing PSNR summary in group run"

    if metric_name == "psnr" and parsed.psnr_avg is None:
        return "missing PSNR summary"

    if metric_name == "cvvdp" and parsed.cvvdp_jod is None:
        return "missing CVVDP JOD"

    if metric_name == "cvvdp" and parsed.cvvdp_jod is not None and not math.isfinite(parsed.cvvdp_jod):
        return "non-finite CVVDP JOD"

    if "cvqa-hdr-nr" in names and proc.returncode == 0:
        if "cvqa-hdr-nr" not in proc.stdout and "quality_score" not in proc.stdout:
            return "missing cvqa-hdr-nr output"
    return None


def execute_case(
    case: NumericalCase,
    env: Environment,
    fixtures: dict,
    *,
    invoke_ctx,
) -> dict[str, Any]:
    fx = fixtures[case.family]
    metric_name = case.metric.split(":", 1)[-1] if case.metric.startswith("group:") else case.metric

    if case.metric.startswith("group:") and case.scenario == "raw_raw" and case.family in ("sdr", "pq"):
        return _row(case, "SKIP", "group includes cvqa-hdr-nr (compressed test only)", [])
    if not case.metric.startswith("group:"):
        if reason := incompatible_reason(metric_name, case.family, case.scenario):
            return _row(case, "SKIP", reason, [])

    if includes_learned(case.metric) and not invoke_ctx.learned_python:
        return _row(
            case,
            "FAIL",
            "learned metrics require PIXVQA_LEARNED_PYTHON or config/learned_metrics.local.toml [learned].python",
            [],
        )

    ref, test, _ref_raw, _test_raw = resolve_media_paths(fx, case.scenario, case.metric)
    no_ref = metric_name == "cvqa-hdr-nr"

    try:
        invoke = build_pixvqa_command(
            invoke_ctx,
            family=case.family,
            scenario=case.scenario,
            metric=case.metric,
            ref=ref,
            test=test,
            no_reference=no_ref,
            metric_device=case.metric_device,
            decode=case.decode,
            explicit_hdr_transfer=case.explicit_hdr_transfer,
        )
    except RuntimeError as exc:
        return _row(case, "FAIL", str(exc), [])

    try:
        proc = subprocess.run(
            invoke.argv,
            capture_output=True,
            text=True,
            env=invoke.env,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return _row(case, "FAIL", "timeout", [invoke.argv])
    except Exception as exc:
        return _row(case, "FAIL", f"{exc}\n{traceback.format_exc()}", [invoke.argv])

    cmd_repr = [str(x) for x in invoke.argv]
    if proc.returncode != 0:
        status, detail = classify_failure(proc.stderr + proc.stdout, proc.returncode, case.metric)
        return _row(case, status, f"rc={proc.returncode}\n{detail}", cmd_repr)

    parsed = parse_run(proc.stdout, proc.stderr, proc.returncode)
    problem = _validate_parsed(case, parsed, invoke, proc)
    if problem:
        return _row(case, "FAIL", problem, cmd_repr)

    return _row(
        case,
        "PASS",
        f"psnr_avg={parsed.psnr_avg} jod={parsed.cvvdp_jod}",
        cmd_repr,
        parsed=parsed,
    )


def _row(
    case: NumericalCase,
    status: str,
    detail: str,
    command: list[str],
    parsed=None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": case.id,
        "family": case.family,
        "scenario": case.scenario,
        "metric": case.metric,
        "status": status,
        "detail": detail,
        "command": command,
        "mandatory": case.mandatory,
    }
    if parsed:
        row["psnr_avg"] = parsed.psnr_avg
        row["cvvdp_jod"] = parsed.cvvdp_jod
    return row


def run_suite(
    *,
    profile: Profile = "smoke",
    strict: bool = False,
    json_report: Path | None = None,
    force_fixtures: bool = False,
) -> int:
    env = discover()
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}

    if not env.numpy_ok:
        print("FAIL numerical suite requires numpy (pip install numpy)", file=sys.stderr)
        return 1
    if not env.ffmpeg:
        print("FAIL numerical suite requires ffmpeg on PATH", file=sys.stderr)
        return 1
    if not bin_runnable(env.full_bin):
        msg = f"mandatory build/full/pixvqa not runnable ({env.full_bin})"
        if strict:
            print(f"FAIL {msg}", file=sys.stderr)
            return 1
        print(f"SKIP numerical suite: {msg}")
        return 0

    fixtures = ensure_all_families(
        env.ffmpeg,
        force=force_fixtures,
        include_nr=(profile == "comprehensive"),
    )
    invoke_ctx = build_invoke_context(env.full_bin)

    cases = build_cases(profile)
    results: list[dict[str, Any]] = []
    consistency_cache: dict[str, float] = {}
    baseline = load_baseline(profile)
    hdr_inferred_psnr: dict[str, float] = {}

    for case in cases:
        if case.id.endswith("consistency.psnr_raw_vs_comp"):
            continue
        row = execute_case(case, env, fixtures, invoke_ctx=invoke_ctx)
        if case.tags and "hdr-equiv" in case.tags:
            inferred = hdr_inferred_psnr.get(case.family)
            if row["status"] == "PASS" and inferred is not None and row.get("psnr_avg") is not None:
                if not close("psnr_avg", inferred, row["psnr_avg"]):
                    row = _row(
                        case,
                        "FAIL",
                        f"HDR explicit PSNR {row['psnr_avg']} vs inferred-path {inferred}",
                        row.get("command", []),
                    )
        bl = compare_to_baseline(row, baseline)
        if bl is not None:
            row = bl
        results.append(row)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        st = row["status"]
        if st == "PASS":
            print(f"OK   {case.id}")
        elif st == "SKIP":
            print(f"SKIP {case.id}: {row['detail'][:120]}")
        else:
            print(f"FAIL {case.id}: {row['detail'][:160]}", file=sys.stderr)
        if row.get("psnr_avg") is not None and case.scenario == "raw_raw" and case.metric == "psnr":
            consistency_cache[f"{case.family}.raw_raw"] = row["psnr_avg"]
        if (
            row.get("psnr_avg") is not None
            and case.scenario == "raw_comp"
            and case.metric == "psnr"
            and not case.explicit_hdr_transfer
        ):
            hdr_inferred_psnr[case.family] = row["psnr_avg"]

    for family in ("sdr", "pq", "hlg"):
        cid = f"{family}.consistency.psnr_raw_vs_comp"
        c1 = consistency_cache.get(f"{family}.raw_raw")
        ccase = NumericalCase(cid, family, "raw_comp", "psnr", tags=("consistency",))
        row2 = execute_case(ccase, env, fixtures, invoke_ctx=invoke_ctx)
        c2 = row2.get("psnr_avg")
        if row2["status"] == "PASS" and c1 is not None and c2 is not None:
            if not close("psnr_avg", c1, c2):
                row2 = _row(ccase, "FAIL", f"PSNR raw_raw {c1} vs raw_comp {c2}", row2.get("command", []))
        results.append(row2)
        counts[row2["status"]] = counts.get(row2["status"], 0) + 1

    if profile == "comprehensive":
        for family in ("sdr", "pq", "hlg"):
            cpu = next((r for r in results if r["id"] == f"{family}.backend.psnr_cpu"), None)
            gpu = next((r for r in results if r["id"] == f"{family}.backend.psnr_gpu"), None)
            if cpu and gpu and cpu["status"] == "PASS" and gpu["status"] == "PASS":
                ok = close("psnr_avg", cpu["psnr_avg"], gpu["psnr_avg"])
                row = {
                    "id": f"{family}.backend.psnr_cpu_vs_gpu",
                    "status": "PASS" if ok else "FAIL",
                    "detail": "cpu/gpu agree" if ok else f"cpu {cpu['psnr_avg']} vs gpu {gpu['psnr_avg']}",
                    "mandatory": True,
                }
                results.append(row)
                counts[row["status"]] += 1
            sw = next((r for r in results if r["id"] == f"{family}.decode.psnr_sw"), None)
            nv = next((r for r in results if r["id"] == f"{family}.decode.psnr_nvdec"), None)
            if sw and nv and sw["status"] == "PASS" and nv["status"] == "PASS":
                ok = close("psnr_avg", sw["psnr_avg"], nv["psnr_avg"])
                results.append(
                    {
                        "id": f"{family}.decode.sw_vs_nvdec",
                        "status": "PASS" if ok else "FAIL",
                        "detail": "decode paths agree" if ok else f"sw {sw['psnr_avg']} vs nvdec {nv['psnr_avg']}",
                        "mandatory": False,
                    }
                )
                counts["PASS" if ok else "FAIL"] += 1

    candidate = {
        r["id"]: {k: r[k] for k in r if k in ("psnr_avg", "cvvdp_jod", "status")}
        for r in results
        if r["status"] == "PASS"
    }
    save_candidate(
        profile,
        candidate,
        {
            "pixvqa": env.pixvqa_version,
            "learned_python": invoke_ctx.learned_python,
            "fixtures": {k: str(v.root) for k, v in fixtures.items()},
        },
    )

    verdict = "PASS" if counts["FAIL"] == 0 and not (strict and counts["SKIP"]) else "FAIL"
    report = {
        "suite": "numerical-conformance",
        "profile": profile,
        "strict": strict,
        "verdict": verdict,
        "finished": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "full_bin": str(env.full_bin),
            "ffmpeg": env.ffmpeg,
            "pixvqa_version": env.pixvqa_version,
            "cuda_available": env.cuda_available,
            "learned_python": invoke_ctx.learned_python,
            "learned_config": str(invoke_ctx.learned_config) if invoke_ctx.learned_config else None,
        },
        "counts": counts,
        "baseline_approved": baseline.get("approved", False),
        "cases": results,
    }
    berlin = run_berlin_optional(env.full_bin)
    if berlin:
        report["optional"] = berlin
        for row in berlin["berlin_cvvdp"]:
            if row["status"] == "FAIL":
                verdict = "FAIL"
                counts["FAIL"] = counts.get("FAIL", 0) + 1
    report["verdict"] = verdict
    if json_report:
        json_report.parent.mkdir(parents=True, exist_ok=True)
        json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nNumerical {profile}: {verdict}  PASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}")
    if json_report:
        print(f"JSON report: {json_report}")
    if counts["FAIL"]:
        return 1
    if strict and counts["SKIP"]:
        return 1
    return 0
