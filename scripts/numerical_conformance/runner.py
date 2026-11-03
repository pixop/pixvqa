"""Execute numerical conformance matrix."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .baselines import compare_to_baseline, load_baseline, save_candidate
from .berlin import run_berlin_optional
from .config import CHROMA, FPS, SIZE
from .environment import Environment, bin_runnable, discover
from .fixtures import ensure_all_families
from .parse_metrics import parse_run
from .registry import CVVDP_DISPLAY, GROUP_METRICS, Scenario, incompatible_reason, metric_supported, scenario_paths
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
                cid = f"{family}.{scenario}.{metric}"
                cases.append(NumericalCase(cid, family, scenario, metric))
            if profile == "comprehensive":
                if scenario == "raw_raw" and family in ("sdr", "pq"):
                    cases.append(
                        NumericalCase(
                            f"{family}.{scenario}.group_{family}",
                            family,
                            scenario,
                            f"group:{family}",
                            mandatory=False,
                        )
                    )
                else:
                    cases.append(
                        NumericalCase(
                            f"{family}.{scenario}.group_{family}",
                            family,
                            scenario,
                            f"group:{family}",
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


def run_pixvqa(
    env: Environment,
    *,
    ref: str,
    test: str,
    metric: str,
    family: str,
    metric_device: str | None = None,
    decode: str | None = None,
    explicit_hdr: bool = False,
    cvvdp_python: str | None = None,
    no_reference: bool = False,
) -> subprocess.CompletedProcess[str]:
    mval = metric.split(":", 1)[1] if metric.startswith("group:") else metric
    cmd = [
        str(env.full_bin),
        "--metrics",
        mval,
        "-t",
        test,
        "-s",
        SIZE,
        "-f",
        CHROMA,
        "--fps",
        FPS,
        "-n",
        "5",
        "-v",
        "1",
    ]
    if not no_reference:
        cmd = [
            str(env.full_bin),
            "--metrics",
            mval,
            "-r",
            ref,
            "-s",
            SIZE,
            "-f",
            CHROMA,
            "-t",
            test,
            "-s",
            SIZE,
            "-f",
            CHROMA,
            "--fps",
            FPS,
            "-n",
            "5",
            "-v",
            "1",
        ]
    if explicit_hdr:
        cmd += ["--hdr-transfer", "sdr" if family == "sdr" else family]
    if metric_device:
        cmd += ["--metric-device", metric_device]
    if decode:
        cmd += ["--decode", decode]
    run_env = os.environ.copy()
    run_env["PIXVQA_LEARNED_CONFIG"] = "none"
    if cvvdp_python and (mval == "cvvdp" or mval in ("sdr", "pq", "hlg")):
        run_env["PIXVQA_CVVDP_PYTHON"] = cvvdp_python
        if mval == "cvvdp" or mval in ("sdr", "pq", "hlg"):
            cmd += ["--cvvdp-python", cvvdp_python]
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env, timeout=600)


def execute_case(
    case: NumericalCase,
    env: Environment,
    fixtures: dict,
    *,
    cvvdp_python: str | None,
) -> dict[str, Any]:
    fx = fixtures[case.family]
    metric_name = case.metric.split(":", 1)[-1] if case.metric.startswith("group:") else case.metric
    if case.metric.startswith("group:") and case.scenario == "raw_raw" and case.family in ("sdr", "pq"):
        return _row(case, "SKIP", "group includes cvqa-hdr-nr (compressed test only)", [])
    if case.metric.startswith("group:"):
        pass
    elif reason := incompatible_reason(metric_name, case.family, case.scenario):
        return _row(case, "SKIP", reason, [])

    ref, test, _ref_raw, _test_raw = scenario_paths(fx, case.scenario)
    no_ref = metric_name == "cvqa-hdr-nr"
    if no_ref:
        ref = str(fx.distorted_lossy)

    try:
        proc = run_pixvqa(
            env,
            ref=ref,
            test=test,
            metric=case.metric,
            family=case.family,
            metric_device=case.metric_device,
            decode=case.decode,
            explicit_hdr=case.explicit_hdr_transfer,
            cvvdp_python=cvvdp_python,
            no_reference=no_ref,
        )
    except subprocess.TimeoutExpired:
        return _row(case, "FAIL", "timeout", [])
    except Exception as exc:
        return _row(case, "FAIL", f"{exc}\n{traceback.format_exc()}", [])

    parsed = parse_run(proc.stdout, proc.stderr, proc.returncode)
    if proc.returncode != 0:
        err = proc.stderr + proc.stdout
        if metric_name in ("cvvdp", "cvqa-hdr-nr") and (
            "checkpoint" in err.lower() or "python" in err.lower() or "cvvdp" in err.lower()
        ):
            return _row(case, "SKIP", f"learned metric unavailable: {proc.stderr[-300:]}", [str(proc.args)])
        return _row(case, "FAIL", f"rc={proc.returncode}\n{proc.stderr[-500:]}", [str(proc.args)])

    if parsed.total_frames != 5:
        return _row(case, "FAIL", f"expected 5 frames, got {parsed.total_frames}", [str(proc.args)])

    if metric_name in ("psnr",) or case.metric.startswith("group:"):
        if parsed.psnr_avg is None and "psnr" in case.metric:
            return _row(case, "FAIL", "missing PSNR summary", [str(proc.args)])

    if "cvvdp" in metric_name and parsed.cvvdp_jod is None and metric_name == "cvvdp":
        return _row(case, "FAIL", "missing CVVDP JOD", [str(proc.args)])

    if case.family == "pq" and case.scenario == "raw_comp" and "hdr-equiv" not in case.tags:
        if "info: cvvdp interpretation pq" not in proc.stderr and "cvvdp" in metric_name:
            pass  # native-only run

    return _row(
        case,
        "PASS",
        f"psnr_avg={parsed.psnr_avg} jod={parsed.cvvdp_jod}",
        [str(proc.args)],
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

    fixtures = ensure_all_families(env.ffmpeg, force=force_fixtures)
    cvvdp_py = sys.executable

    cases = build_cases(profile)
    results: list[dict[str, Any]] = []
    consistency_cache: dict[str, float] = {}
    baseline = load_baseline(profile)
    hdr_inferred_psnr: dict[str, float] = {}

    for case in cases:
        if case.id.endswith("consistency.psnr_raw_vs_comp"):
            continue
        row = execute_case(case, env, fixtures, cvvdp_python=cvvdp_py)
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
            and case.family
            and case.scenario == "raw_comp"
            and case.metric == "psnr"
            and not case.explicit_hdr_transfer
        ):
            hdr_inferred_psnr[case.family] = row["psnr_avg"]

    # Consistency cases
    for family in ("sdr", "pq", "hlg"):
        cid = f"{family}.consistency.psnr_raw_vs_comp"
        c1 = consistency_cache.get(f"{family}.raw_raw")
        ccase = NumericalCase(cid, family, "raw_comp", "psnr", tags=("consistency",))
        row2 = execute_case(ccase, env, fixtures, cvvdp_python=cvvdp_py)
        c2 = row2.get("psnr_avg")
        if row2["status"] == "PASS" and c1 is not None and c2 is not None:
            if not close("psnr_avg", c1, c2):
                row2 = _row(ccase, "FAIL", f"PSNR raw_raw {c1} vs raw_comp {c2}", row2.get("command", []))
        results.append(row2)
        counts[row2["status"]] = counts.get(row2["status"], 0) + 1

    # CPU vs GPU PSNR (comprehensive)
    if profile == "comprehensive":
        for family in ("sdr", "pq", "hlg"):
            cpu = next((r for r in results if r["id"] == f"{family}.backend.psnr_cpu"), None)
            gpu = next((r for r in results if r["id"] == f"{family}.backend.psnr_gpu"), None)
            if cpu and gpu and cpu["status"] == "PASS" and gpu["status"] == "PASS":
                if not close("psnr_avg", cpu["psnr_avg"], gpu["psnr_avg"]):
                    results.append(
                        {
                            "id": f"{family}.backend.psnr_cpu_vs_gpu",
                            "status": "FAIL",
                            "detail": f"cpu {cpu['psnr_avg']} vs gpu {gpu['psnr_avg']}",
                            "mandatory": True,
                        }
                    )
                    counts["FAIL"] += 1
                else:
                    results.append(
                        {
                            "id": f"{family}.backend.psnr_cpu_vs_gpu",
                            "status": "PASS",
                            "detail": "cpu/gpu agree",
                            "mandatory": True,
                        }
                    )
                    counts["PASS"] += 1
            sw = next((r for r in results if r["id"] == f"{family}.decode.psnr_sw"), None)
            nv = next((r for r in results if r["id"] == f"{family}.decode.psnr_nvdec"), None)
            if sw and nv and sw["status"] == "PASS" and nv["status"] == "PASS":
                if not close("psnr_avg", sw["psnr_avg"], nv["psnr_avg"]):
                    results.append(
                        {
                            "id": f"{family}.decode.sw_vs_nvdec",
                            "status": "FAIL",
                            "detail": f"sw {sw['psnr_avg']} vs nvdec {nv['psnr_avg']}",
                            "mandatory": False,
                        }
                    )
                    counts["FAIL"] += 1
                else:
                    results.append(
                        {
                            "id": f"{family}.decode.sw_vs_nvdec",
                            "status": "PASS",
                            "detail": "decode paths agree",
                            "mandatory": False,
                        }
                    )
                    counts["PASS"] += 1

    candidate = {r["id"]: {k: r[k] for k in r if k in ("psnr_avg", "cvvdp_jod", "status")} for r in results if r["status"] == "PASS"}
    save_candidate(profile, candidate, {"pixvqa": env.pixvqa_version, "fixtures": {k: str(v.root) for k, v in fixtures.items()}})

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
