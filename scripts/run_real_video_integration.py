#!/usr/bin/env python3
"""Reproducible, single-worker real-video integration campaign (146 planned cases).

Phase A is the original 48-pair PSNR campaign. B--G are explicit new *proposed*
coverage, not a recovered historical specification. No status is a PASS until a
PixVQA process has returned 0, processed the requested frame count, and emitted
its expected metrics; F requires a genuine CPU/GPU comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MODES = {"sdr": 60, "pq": 60, "hlg": 30}
FORMATS = ("yuv", "hevc", "av1", "vvc")
PU21 = "pu21-psnr,pu21-ssim,pu21-msssim"
VERSION = "pixvqa-integration-146-v1"
PSNR = re.compile(r"\[\s*PSNR\s+y:([0-9.+-]+).*?avg:([0-9.+-]+)")
PU21_PSNR = re.compile(r"\[ PU21-PSNR\s+r:([0-9.]+) g:([0-9.]+) b:([0-9.]+) avg:([0-9.]+) \]")
PU21_LUMA = {metric: re.compile(rf"\[ {metric}\s+y:([0-9.]+) u:n/a v:n/a avg:([0-9.]+) \]")
             for metric in ("PU21-SSIM", "PU21-MS-SSIM")}


@dataclass(frozen=True)
class Case:
    id: str
    phase: str
    mode: str
    ref_format: str
    test_format: str
    metrics: str
    backend: str = "cpu"  # cpu | parity
    model: str = ""
    peak: int = 0
    definition: str = "historical"  # historical | proposed


def input_path(root: Path, mode: str, fmt: str) -> Path:
    if fmt == "yuv":
        if mode == "hlg":
            return root / "hlg_ref_30s_2160p_420p10.yuv"
        return root / f"{mode}_ref_2160p_420p10.yuv"
    # Keep the historical Phase-A name priority. The fixture generator may
    # alternatively create uncorrected MP4s or raw VVC .266 elementary streams.
    if mode == "pq":
        candidates = [root / f"pq_{fmt}_qp32_bt2020nc_fixed.mp4",
                      root / f"pq_{fmt}_qp32.mp4"]
    elif mode == "hlg":
        candidates = [root / f"hlg_{fmt}_30s_qp32.mp4",
                      root / f"hlg_{fmt}_qp32.mp4"]
    else:
        candidates = [root / f"sdr_{fmt}_qp32.mp4"]
    if fmt == "vvc":
        candidates.append(root / f"{mode}_vvc_qp32.266")
    return next((p for p in candidates if p.is_file()), candidates[0])


def manifest() -> list[Case]:
    cases: list[Case] = []

    def add(phase: str, mode: str, ref: str, test: str, metrics: str,
            backend: str = "cpu", model: str = "", peak: int = 0) -> None:
        i = 1 + sum(c.phase == phase for c in cases)
        cases.append(Case(f"{phase}{i:03d}_{mode}_{ref}_to_{test}", phase,
                          mode, ref, test, metrics, backend, model, peak,
                          "historical" if phase == "A" else "proposed"))

    # A: original Phase A, same exact order and PSNR arguments as the archived runner.
    for mode in MODES:
        for ref in FORMATS:
            for test in FORMATS:
                add("A", mode, ref, test, "psnr")

    # B: same-file multi-metric codec sanity, 3 modes x 4 input formats = 12.
    for mode in MODES:
        for fmt in FORMATS:
            add("B", mode, fmt, fmt, "psnr,ssim,msssim,xpsnr")

    # C: 8 directed cross-format pairs per mode = 24.
    cross = (("yuv", "hevc"), ("hevc", "yuv"), ("yuv", "av1"),
             ("av1", "yuv"), ("yuv", "vvc"), ("vvc", "yuv"),
             ("hevc", "av1"), ("av1", "hevc"))
    for mode in MODES:
        for ref, test in cross:
            add("C", mode, ref, test, "psnr,ssim,msssim")

    # D: 9 pairs each for PQ/HLG display-domain PU21 = 18.
    pu_pairs = (("yuv", "yuv"), ("hevc", "hevc"), ("av1", "av1"),
                ("vvc", "vvc"), ("yuv", "hevc"), ("hevc", "yuv"),
                ("yuv", "av1"), ("av1", "yuv"), ("hevc", "av1"))
    for mode in ("pq", "hlg"):
        for ref, test in pu_pairs:
            add("D", mode, ref, test, PU21)

    # E: 6 compressed pairs x 3 modes: SDR VMAF, HDR PQ VMAF, HLG XPSNR/SSIM.
    codec_pairs = (("hevc", "hevc"), ("av1", "av1"), ("vvc", "vvc"),
                   ("hevc", "av1"), ("av1", "vvc"), ("vvc", "hevc"))
    for mode in MODES:
        for ref, test in codec_pairs:
            if mode == "sdr":
                add("E", mode, ref, test, "vmaf", model="model/vmaf_v1.0.16_3d0h.json")
            elif mode == "pq":
                add("E", mode, ref, test, "vmaf",
                    model="model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json")
            else:
                add("E", mode, ref, test, "xpsnr,ssim")

    # F: 6 compressed pairs x 2 HDR modes = 12 mandatory CPU/CUDA parity cases.
    for mode in ("pq", "hlg"):
        for ref, test in codec_pairs:
            add("F", mode, ref, test, PU21, backend="parity")

    # G: *newly proposed* 14 additional peak-conditioned UHD PU21 cases:
    # PQ clip=1000 nits, HLG display peak=4000 nits (different from 1000 default).
    # Historical handoff does not define the extra 14; these are explicit,
    # reviewable additions, not a claim about earlier approvals.
    peak_pairs = (("yuv", "yuv"), *codec_pairs)
    for mode in ("pq", "hlg"):
        for ref, test in peak_pairs:
            add("G", mode, ref, test, PU21, peak=1000 if mode == "pq" else 4000)

    counts = {phase: sum(c.phase == phase for c in cases) for phase in "ABCDEFG"}
    assert counts == dict(zip("ABCDEFG", (48, 12, 24, 18, 18, 12, 14))), counts
    assert len(cases) == 146 and len({c.id for c in cases}) == 146
    return cases


def command(case: Case, exe: Path, root: Path, frames: int, device: str,
            repository: Path) -> list[str]:
    ref, dist = (input_path(root, case.mode, fmt) for fmt in (case.ref_format, case.test_format))
    cmd = [str(exe), "--metrics", case.metrics, "-r", str(ref), "-t", str(dist),
           "-s", "3840x2160", "-f", "420p:10", "--fps", str(MODES[case.mode]),
           "--frames", str(frames), "--input", "auto", "--device", device,
           "--decode", "cpu", "--hdr-transfer", case.mode, "-v", "0"]
    if case.model:
        cmd.extend(["--model", str((repository / case.model).resolve())])
    if case.peak:
        cmd.extend(["--hdr-peak-nits", str(case.peak)])
    return cmd


def parse_metrics(stdout: str, metrics: str) -> dict[str, float] | None:
    result: dict[str, float] = {}
    for metric in metrics.split(","):
        if metric == "psnr":
            match = PSNR.search(stdout)
            if match:
                result.update(psnr_y=float(match.group(1)), psnr_avg=float(match.group(2)))
        elif metric == "pu21-psnr":
            match = PU21_PSNR.search(stdout)
            if match:
                result.update({"pu21_r": float(match.group(1)), "pu21_g": float(match.group(2)),
                               "pu21_b": float(match.group(3)), "pu21_avg": float(match.group(4))})
        elif metric in ("pu21-ssim", "pu21-msssim"):
            key = "PU21-SSIM" if metric == "pu21-ssim" else "PU21-MS-SSIM"
            match = PU21_LUMA[key].search(stdout)
            if match:
                result[metric] = float(match.group(1))
        elif metric == "vmaf":
            match = re.search(r"\[ VMAF[^\]]*?score:([0-9.]+)", stdout)
            if not match:
                match = re.search(r"\[ VMAF mean:([0-9.]+)", stdout)
            if match:
                result["vmaf"] = float(match.group(1))
        elif metric in ("ssim", "msssim", "xpsnr"):
            name = {"ssim": "SSIM", "msssim": "MS-SSIM", "xpsnr": "XPSNR"}[metric]
            match = re.search(rf"\[\s*{name}\s+[^\]]*?avg:([0-9.+-]+|inf)", stdout)
            if match:
                result[metric] = ("inf" if match.group(1) == "inf" else float(match.group(1)))
    expected = {"psnr": "psnr_y", "pu21-psnr": "pu21_avg", "vmaf": "vmaf"}
    if any((expected.get(k) or k) not in result for k in metrics.split(",")):
        return None
    return result


def file_stamp(path: Path) -> dict:
    try:
        st = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def case_signature(case: Case, exe: Path, root: Path, frames: int, repository: Path) -> str:
    paths = [exe, input_path(root, case.mode, case.ref_format),
             input_path(root, case.mode, case.test_format)]
    if case.model:
        paths.append(repository / case.model)
    payload = {"case": asdict(case), "frames": frames, "files": [file_stamp(p) for p in paths],
               "cpu_command": command(case, exe, root, frames, "cpu", repository),
               "gpu_command": command(case, exe, root, frames, "gpu", repository)
               if case.backend == "parity" else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def execute(cmd: list[str], frames: int, metrics: str, timeout: int) -> dict:
    start = time.monotonic()
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "FAIL", "error": repr(exc), "elapsed_s": round(time.monotonic()-start, 2)}
    out = run.stdout + "\n" + run.stderr
    count = re.search(r"Total frames:\s*(\d+)", out)
    parsed = parse_metrics(run.stdout, metrics)
    valid = run.returncode == 0 and count is not None and int(count.group(1)) == frames and parsed is not None
    return {"status": "PASS" if valid else "FAIL", "returncode": run.returncode,
            "frames": int(count.group(1)) if count else None, "metrics": parsed,
            "elapsed_s": round(time.monotonic()-start, 2), "output_tail": out[-4500:] if not valid else ""}


def report_write(path: Path, selected: list[Case], results: dict[str, dict], args: argparse.Namespace) -> None:
    rows = [results.get(case.id, {"id": case.id, "status": "NOT_RUN"}) for case in selected]
    counts = {status: sum(r["status"] == status for r in rows)
              for status in ("PASS", "FAIL", "BLOCKED", "NOT_RUN")}
    body = {"schema_version": VERSION, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "definition_note": "A is historical; B-G are explicitly proposed/reviewable coverage",
            "exe": str(args.exe), "data": str(args.data), "frames": args.frames,
            "planned_full_campaign": 146, "selected": len(selected), "counts": counts,
            "complete": counts["PASS"] == len(selected),
            "complete_full_campaign": len(selected) == 146 and counts["PASS"] == 146,
            "cases": rows}
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build/full/pixvqa"))
    parser.add_argument("--data", type=Path, default=Path("/mnt/d/tmp/2160p_420"))
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--phase", choices=[*"ABCDEFG", "all"], default="all")
    parser.add_argument("--case-id", action="append", default=[], help="Exact ID; repeatable")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="Reuse PASS only for matching file/command stamps")
    parser.add_argument("--list", action="store_true", help="Print proposed manifest and exit (no media required)")
    parser.add_argument("--check-fixtures", action="store_true", help="Report missing media/models without running")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/real_video_integration"))
    args = parser.parse_args(argv)
    if args.frames < 1 or args.frames > 60 or args.timeout < 1:
        parser.error("frames must be 1..60 and timeout must be positive")
    args.exe, args.data, args.repository = (p.resolve() for p in (args.exe, args.data, args.repository))
    cases = manifest()
    selected = [c for c in cases if (args.mode == "all" or c.mode == args.mode)
                and (args.phase == "all" or c.phase == args.phase)
                and (not args.case_id or c.id in args.case_id)]
    if args.case_id:
        unknown = set(args.case_id)-{c.id for c in cases}
        if unknown:
            parser.error("Unknown case IDs: " + ", ".join(sorted(unknown)))
    if not selected:
        parser.error("No cases match selection")
    if args.list:
        print(json.dumps({"version": VERSION, "count": len(selected),
                          "cases": [asdict(c) for c in selected]}, indent=2))
        return 0
    if args.check_fixtures:
        missing = sorted({str(p) for c in selected
                          for p in (input_path(args.data, c.mode, c.ref_format),
                                    input_path(args.data, c.mode, c.test_format),
                                    *((args.repository / c.model,) if c.model else ())) if not p.is_file()})
        print(json.dumps({"selected": len(selected), "missing": missing}, indent=2))
        return 0 if not missing else 2
    if not args.exe.is_file():
        parser.error(f"PixVQA executable not found: {args.exe}")
    args.output.mkdir(parents=True, exist_ok=True)
    # A filtered smoke run must never replace a previous complete 146-case
    # report. Resume is scoped to the identical selected-case set.
    if len(selected) == len(cases):
        name = "full_146_results.json"
    elif args.case_id:
        digest = hashlib.sha256("\n".join(c.id for c in selected).encode()).hexdigest()[:12]
        name = f"selected_{len(selected)}_{digest}_results.json"
    else:
        scope = f"phase_{args.phase.lower()}" if args.phase != "all" else "campaign"
        if args.mode != "all":
            scope += f"_mode_{args.mode}"
        name = scope + "_results.json"
    report = args.output / name
    old = {}
    if args.resume and report.is_file():
        prior = json.loads(report.read_text(encoding="utf-8"))
        if prior.get("schema_version") == VERSION:
            old = {r["id"]: r for r in prior["cases"] if "id" in r}
    results: dict[str, dict] = {}
    for index, case in enumerate(selected, 1):
        signature = case_signature(case, args.exe, args.data, args.frames, args.repository)
        previous = old.get(case.id, {})
        if previous.get("status") == "PASS" and previous.get("signature") == signature:
            results[case.id] = previous
            print(f"[{index}/{len(selected)}] {case.id}: PASS (resumed)", flush=True)
            continue
        files = [input_path(args.data, case.mode, case.ref_format),
                 input_path(args.data, case.mode, case.test_format)]
        if case.model:
            files.append(args.repository / case.model)
        missing = [str(path) for path in files if not path.is_file()]
        result: dict = {"id": case.id, "phase": case.phase, "definition": case.definition,
                        "mode": case.mode, "reference_format": case.ref_format,
                        "test_format": case.test_format, "metrics_requested": case.metrics,
                        "signature": signature}
        if missing:
            result.update(status="BLOCKED", reason="missing fixtures/models", missing=missing)
        else:
            cpu_cmd = command(case, args.exe, args.data, args.frames, "cpu", args.repository)
            result["commands"] = [cpu_cmd]
            cpu = execute(cpu_cmd, args.frames, case.metrics, args.timeout)
            result["cpu"] = cpu
            if case.backend == "parity" and cpu["status"] == "PASS":
                gpu_cmd = command(case, args.exe, args.data, args.frames, "gpu", args.repository)
                result["commands"].append(gpu_cmd)
                gpu = execute(gpu_cmd, args.frames, case.metrics, args.timeout)
                result["gpu"] = gpu
                if gpu["status"] == "PASS":
                    errors = {k: abs(value-gpu["metrics"][k]) for k, value in cpu["metrics"].items()}
                    result["parity_abs_error"] = errors
                    result["status"] = "PASS" if all(
                        d <= 1e-4 or d <= 1e-6*max(abs(cpu["metrics"][k]), 1.0)
                        for k, d in errors.items()) else "FAIL"
                    if result["status"] == "FAIL":
                        result["reason"] = "CPU/GPU metric discrepancy"
                else:
                    result["status"] = "FAIL"
                    result["reason"] = "mandatory GPU run failed"
            else:
                result["status"] = cpu["status"]
        results[case.id] = result
        print(f"[{index}/{len(selected)}] {case.id}: {result['status']}", flush=True)
        report_write(report, selected, results, args)
        if args.stop_on_fail and result["status"] != "PASS":
            break
    report_write(report, selected, results, args)
    statuses = [results.get(c.id, {}).get("status", "NOT_RUN") for c in selected]
    print(f"Campaign: {statuses.count('PASS')}/{len(selected)} PASS; "
          f"{statuses.count('FAIL')} FAIL; {statuses.count('BLOCKED')} BLOCKED; "
          f"{statuses.count('NOT_RUN')} NOT_RUN. Report: {report}")
    return 0 if all(s == "PASS" for s in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
