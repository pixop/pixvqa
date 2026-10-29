#!/usr/bin/env python3
"""Gate B: software HEVC / NVDEC / CUDA-metrics / FIFO + CPU VMAF operational tests.

Does not implement GPU VMAF. Distinguishes PASSED / FAILED / NOT EXECUTED.
Evidence: reports/vmaf_gate_b_results.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(os.environ.get("VMAF_GATE_B_DIR", ROOT / "testdata" / "vmaf_gate_b"))
MODEL = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"
PIX_CPU = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
PIX_SW = Path(os.environ.get("PIXVQA_SWDEC_BIN", ROOT / "build" / "swdec" / "pixvqa"))
PIX_CUDA = Path(os.environ.get("PIXVQA_CUDA_BIN", ROOT / "build" / "cuda" / "pixvqa"))
PIX_FULL = Path(os.environ.get("PIXVQA_FULL_BIN", ROOT / "build" / "full" / "pixvqa"))
TOL_SCORE = 1e-6
TOL_POOL = 1e-6
FIFO_TIMEOUT_S = float(os.environ.get("VMAF_GATE_B_FIFO_TIMEOUT", "60"))

# Documented CLI print precision (src/main.cpp). Coexistence compares at this precision.
PREC = {
    "psnr": {"decimals": 2, "fields": ("y", "u", "v", "avg"), "tol": 0.0},
    "mse": {"decimals": 1, "fields": ("y", "u", "v", "avg"), "tol": 0.0},
    "ssim": {"decimals": 6, "fields": ("y", "u", "v", "avg"), "tol": 0.0},
    "msssim": {"decimals": 6, "fields": ("y", "u", "v", "avg"), "tol": 0.0},
    "xpsnr": {"decimals": 4, "fields": ("y", "u", "v", "avg", "min"), "tol": 0.0},
}

RE_PSNR = re.compile(
    r"\[ PSNR\s+y:([0-9.]+)\s+u:([0-9.]+)\s+v:([0-9.]+)\s+avg:([0-9.]+) \]"
    r"\s*\[ MSE\s+y:([0-9.]+)\s+u:([0-9.]+)\s+v:([0-9.]+)\s+avg:([0-9.]+) \]"
)
RE_SSIM = re.compile(
    r"(?<![A-Z-])\[ SSIM\s+y:(-?[0-9.]+)\s+u:(-?[0-9.]+)\s+v:(-?[0-9.]+)\s+avg:(-?[0-9.]+) \]"
)
RE_MSSSIM = re.compile(
    r"\[ MS-SSIM\s+y:(-?[0-9.]+)\s+u:(-?[0-9.]+)\s+v:(-?[0-9.]+)\s+avg:(-?[0-9.]+) \]"
)
RE_XPSNR = re.compile(
    r"\[ XPSNR y:([0-9.inf]+)\s+u:([0-9.inf]+)\s+v:([0-9.inf]+)\s+avg:([0-9.inf]+)\s+min:([0-9.inf]+) \]"
)
RE_FRAMES = re.compile(r"Total frames:(\d+)")
RE_CUDA_PROF = re.compile(
    r"\[PIXVQA_CUDA_PROFILE\] h2d=([0-9.]+)ms kernel=([0-9.]+)ms d2h=([0-9.]+)ms "
    r"frames=(\d+).*?h2d_test=(\d+)"
)
RE_DEC_PROF = re.compile(
    r"pixvqa: decode_profile seconds=([0-9.]+) p010_seconds=([0-9.]+) "
    r"download_seconds=([0-9.]+) frames=(\d+) nvdec=(\d+)"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_vmaf(bin_path: Path) -> bool:
    if not bin_path.is_file():
        return False
    p = subprocess.run([str(bin_path), "--help"], capture_output=True, text=True)
    return "msssim, vmaf" in p.stdout and "--model PATH" in p.stdout


def supports_cuda_xpsnr(bin_path: Path) -> bool:
    ver = subprocess.run([str(bin_path), "--version"], capture_output=True, text=True)
    return "build: cuda" in ver.stdout


def parse_vmaf(out: str) -> tuple[list[float], dict[str, float], int]:
    per = []
    for line in out.splitlines():
        if line.startswith("vmaf_features "):
            kv = dict(re.findall(r"(\w+)=([-+0-9.eE]+)", line))
            if "score" in kv:
                per.append(float(kv["score"]))
    if not per:
        per = [float(x) for x in re.findall(r"\[ VMAF:([0-9.]+) \]", out)]
    m = re.search(
        r"\[ VMAF mean:([0-9.]+) harmonic:([0-9.]+) min:([0-9.]+) max:([0-9.]+) frames:(\d+) \]",
        out,
    )
    if not m:
        raise SystemExit("missing VMAF summary:\n" + out[-1500:])
    pools = {
        "mean": float(m.group(1)),
        "harmonic_mean": float(m.group(2)),
        "min": float(m.group(3)),
        "max": float(m.group(4)),
    }
    return per, pools, int(m.group(5))


def run_pix(
    bin_path: Path,
    args: list[str],
    env: dict | None = None,
    timeout: float | None = None,
) -> tuple[int, str]:
    e = os.environ.copy()
    if env:
        e.update(env)
    try:
        p = subprocess.run(
            [str(bin_path), *args],
            capture_output=True,
            text=True,
            env=e,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout or "") + "\n" + (ex.stderr or "")
        return 124, out + f"\nTIMEOUT after {timeout}s"
    return p.returncode, p.stdout + "\n" + p.stderr


def compare_scores(
    a: list[float], b: list[float], pa: dict[str, float], pb: dict[str, float], label: str
) -> dict:
    if len(a) != len(b):
        return {
            "label": label,
            "status": "FAILED",
            "detail": f"frame count {len(a)} vs {len(b)}",
            "max_abs_frame": None,
            "pool_abs": {},
            "frames": None,
        }
    max_frame = max(abs(x - y) for x, y in zip(a, b)) if a else 0.0
    pool_abs = {k: abs(pa[k] - pb[k]) for k in pa}
    ok = max_frame <= TOL_SCORE and all(v <= TOL_POOL for v in pool_abs.values())
    return {
        "label": label,
        "status": "PASSED" if ok else "FAILED",
        "max_abs_frame": max_frame,
        "pool_abs": pool_abs,
        "frames": len(a),
    }


def decode_identity_failures(
    *,
    dump_sha: str | None,
    oracle_sha: str,
    dump_size: int | None,
    expected_size: int,
    frame_count: int,
    expected_frames: int,
    also_match_sha: str | None = None,
) -> list[str]:
    """Return human-readable failure reasons; empty list means identity OK."""
    fails: list[str] = []
    if dump_sha is None:
        fails.append("decoded dump missing")
        return fails
    if dump_sha != oracle_sha:
        fails.append(f"dump sha {dump_sha} != oracle {oracle_sha}")
    if also_match_sha is not None and dump_sha != also_match_sha:
        fails.append(f"dump sha {dump_sha} != peer dump {also_match_sha}")
    if dump_size is None:
        fails.append("decoded dump size unknown")
    elif dump_size != expected_size:
        fails.append(f"dump size {dump_size} != expected {expected_size}")
    if frame_count != expected_frames:
        fails.append(f"frames {frame_count} != expected {expected_frames}")
    return fails


def parse_cuda_metrics(out: str, expect_xpsnr: bool) -> dict:
    """Parse per-frame and summary metric blocks from -v 1 output."""
    summary_idx = out.find("Summary:")
    body = out if summary_idx < 0 else out[:summary_idx]
    summary = "" if summary_idx < 0 else out[summary_idx:]

    def collect(rx: re.Pattern[str], text: str) -> list[tuple[float, ...]]:
        rows = []
        for m in rx.finditer(text):
            rows.append(tuple(float(g) if g != "inf" else float("inf") for g in m.groups()))
        return rows

    per = {
        "psnr": collect(RE_PSNR, body),
        "ssim": collect(RE_SSIM, body),
        "msssim": collect(RE_MSSSIM, body),
        "xpsnr": collect(RE_XPSNR, body),
    }
    # PSNR regex captures MSE too; split.
    psnr_per = [r[:4] for r in per["psnr"]]
    mse_per = [r[4:] for r in per["psnr"]]
    sum_psnr = collect(RE_PSNR, summary)
    sum_ssim = collect(RE_SSIM, summary)
    sum_msssim = collect(RE_MSSSIM, summary)
    sum_xpsnr = collect(RE_XPSNR, summary)
    fr = RE_FRAMES.search(out)
    prof = RE_CUDA_PROF.search(out)
    return {
        "frames_total": int(fr.group(1)) if fr else None,
        "per_frame": {
            "psnr": psnr_per,
            "mse": mse_per,
            "ssim": per["ssim"],
            "msssim": per["msssim"],
            "xpsnr": per["xpsnr"],
        },
        "summary": {
            "psnr": sum_psnr[-1][:4] if sum_psnr else None,
            "mse": sum_psnr[-1][4:] if sum_psnr else None,
            "ssim": sum_ssim[-1] if sum_ssim else None,
            "msssim": sum_msssim[-1] if sum_msssim else None,
            "xpsnr": sum_xpsnr[-1] if sum_xpsnr else None,
        },
        "cuda_profile": {
            "h2d_ms": float(prof.group(1)),
            "kernel_ms": float(prof.group(2)),
            "d2h_ms": float(prof.group(3)),
            "frames": int(prof.group(4)),
            "h2d_test": int(prof.group(5)),
        }
        if prof
        else None,
        "has_vmaf_summary": "[ VMAF mean:" in out,
        "has_vmaf_frame": bool(re.search(r"\[ VMAF:[0-9.]+ \]", out)),
        "msssim_fp64_warning": "warning: CUDA MS-SSIM is using" in out,
        "expect_xpsnr": expect_xpsnr,
    }


def compare_metric_pair(
    name: str,
    a_per: list[tuple],
    b_per: list[tuple],
    a_sum: tuple | None,
    b_sum: tuple | None,
    nframes: int,
) -> dict:
    meta = PREC[name]
    tol = meta["tol"]
    detail = {
        "metric": name,
        "print_decimals": meta["decimals"],
        "fields": list(meta["fields"]),
        "tolerance": tol,
        "agreement_class": f"equality at documented CLI print precision (%.{meta['decimals']}f tokens)",
        "status": "PASSED",
    }
    if len(a_per) != nframes or len(b_per) != nframes:
        detail["status"] = "FAILED"
        detail["detail"] = f"per-frame count a={len(a_per)} b={len(b_per)} expected={nframes}"
        return detail
    if a_sum is None or b_sum is None:
        detail["status"] = "FAILED"
        detail["detail"] = "missing summary"
        return detail
    max_pf = 0.0
    for i, (ra, rb) in enumerate(zip(a_per, b_per)):
        if len(ra) != len(rb):
            detail["status"] = "FAILED"
            detail["detail"] = f"field count mismatch frame {i}"
            return detail
        for xa, xb in zip(ra, rb):
            max_pf = max(max_pf, abs(xa - xb))
    max_sum = max(abs(xa - xb) for xa, xb in zip(a_sum, b_sum))
    detail["max_abs_per_frame"] = max_pf
    detail["max_abs_summary"] = max_sum
    detail["summary_without_vmaf"] = list(a_sum)
    detail["summary_with_vmaf"] = list(b_sum)
    if max_pf > tol + 1e-15 or max_sum > tol + 1e-15:
        detail["status"] = "FAILED"
        detail["detail"] = "printed values differ beyond tolerance"
    return detail


def cuda_metric_flags(include_xpsnr: bool) -> list[str]:
    flags = [
        "--metric-device", "gpu",
        "--metrics", "psnr,ssim,msssim",
        "-v",
        "1"]
    if include_xpsnr:
        flags += ["--metrics", "xpsnr",  "--fps", "30"]
    return flags


def validate_cuda_run(parsed: dict, nframes: int, vmaf_expected: bool, label: str) -> list[str]:
    fails: list[str] = []
    if parsed["frames_total"] != nframes:
        fails.append(f"{label}: Total frames {parsed['frames_total']} != {nframes}")
    for key in ("psnr", "ssim", "msssim"):
        if len(parsed["per_frame"][key]) != nframes:
            fails.append(f"{label}: missing per-frame {key} ({len(parsed['per_frame'][key])})")
        if parsed["summary"][key] is None:
            fails.append(f"{label}: missing summary {key}")
    if parsed["expect_xpsnr"]:
        if len(parsed["per_frame"]["xpsnr"]) != nframes:
            fails.append(f"{label}: missing per-frame xpsnr")
        if parsed["summary"]["xpsnr"] is None:
            fails.append(f"{label}: missing summary xpsnr")
    if parsed["cuda_profile"] is None:
        fails.append(f"{label}: missing [PIXVQA_CUDA_PROFILE] (set PIXVQA_CUDA_PROFILE=1)")
    else:
        if parsed["cuda_profile"]["frames"] != nframes:
            fails.append(f"{label}: CUDA profile frames mismatch")
        if parsed["cuda_profile"]["kernel_ms"] <= 0.0:
            fails.append(f"{label}: CUDA kernel_ms <= 0 (possible CPU fallback)")
    if parsed["msssim_fp64_warning"]:
        fails.append(f"{label}: unexpected non-FP64 MS-SSIM warning")
    if vmaf_expected:
        if not parsed["has_vmaf_summary"] or not parsed["has_vmaf_frame"]:
            fails.append(f"{label}: CPU VMAF output missing")
    else:
        if parsed["has_vmaf_summary"] or parsed["has_vmaf_frame"]:
            fails.append(f"{label}: VMAF unexpectedly present")
    return fails


def compare_cuda_pair(d0: dict, d1: dict, nframes: int, include_xpsnr: bool) -> tuple[list[dict], list[str]]:
    metrics = ["psnr", "mse", "ssim", "msssim"]
    if include_xpsnr:
        metrics.append("xpsnr")
    rows = []
    fails = []
    for name in metrics:
        row = compare_metric_pair(
            name,
            d0["per_frame"][name],
            d1["per_frame"][name],
            d0["summary"][name],
            d1["summary"][name],
            nframes,
        )
        rows.append(row)
        if row["status"] != "PASSED":
            fails.append(f"{name}: {row.get('detail', 'mismatch')}")
    return rows, fails


def main() -> int:
    results: dict = {
        "model": str(MODEL),
        "model_sha256": sha256_file(MODEL) if MODEL.is_file() else None,
        "fixture_dir": str(FIXTURE),
        "cases": [],
    }
    failures = 0

    def add(case: dict) -> None:
        nonlocal failures
        results["cases"].append(case)
        extra = case.get("detail", "")
        if isinstance(extra, str) and extra:
            print(f"{case['status']} {case['name']} — {extra[:200]}")
        else:
            print(f"{case['status']} {case['name']}")
        if case["status"] == "FAILED":
            failures += 1

    man = FIXTURE / "manifest.json"
    if not man.is_file():
        print("generating fixtures ...", flush=True)
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_vmaf_hevc_fixture.py")])
        if rc.returncode != 0:
            add({"name": "fixture_generation", "status": "FAILED", "detail": "gen script failed"})
            Path(ROOT / "reports" / "vmaf_gate_b_results.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            return 1
    manifest = json.loads(man.read_text(encoding="utf-8"))
    W, H, N = manifest["width"], manifest["height"], manifest["frames"]
    expected_size = int(manifest.get("expected_yuv_bytes", N * (W * H * 2 + 2 * ((W // 2) * (H // 2) * 2))))
    ref = FIXTURE / "ref.yuv"
    oracle = FIXTURE / "dist_oracle.yuv"
    hevc = FIXTURE / "dist.hevc.mp4"
    results["fixture"] = manifest

    # B-frame evidence from manifest (regenerated by gen script).
    hevc_stream = manifest.get("hevc_stream") or {}
    if hevc_stream.get("has_b_pictures") and hevc_stream.get("b_picture_count", 0) >= 1:
        add(
            {
                "name": "fixture_bframe_evidence",
                "status": "PASSED",
                "b_picture_count": hevc_stream.get("b_picture_count"),
                "pict_type_counts": hevc_stream.get("pict_type_counts"),
                "profile": hevc_stream.get("profile"),
                "pix_fmt": hevc_stream.get("pix_fmt"),
                "detail": f"ffprobe found {hevc_stream.get('b_picture_count')} B-pictures",
            }
        )
    else:
        add(
            {
                "name": "fixture_bframe_evidence",
                "status": "FAILED",
                "detail": "manifest lacks proven B-pictures; regenerate with gen_vmaf_hevc_fixture.py",
                "hevc_stream": hevc_stream,
            }
        )

    model_args = ["--metrics", "vmaf", "--model", str(MODEL)]
    size_fmt = ["-s", f"{W}x{H}", "-f", "420p:10"]

    # Negative control for digest assertion machinery.
    neg_fails = decode_identity_failures(
        dump_sha="0" * 64,
        oracle_sha="1" * 64,
        dump_size=expected_size - 1,
        expected_size=expected_size,
        frame_count=N - 1,
        expected_frames=N,
        also_match_sha="2" * 64,
    )
    add(
        {
            "name": "digest_assertion_negative_control",
            "status": "PASSED" if len(neg_fails) >= 3 else "FAILED",
            "detected_failures": neg_fails,
            "detail": "deliberately mismatched digests/size/frames must be reported",
        }
    )

    # --- Path A: raw oracle ---
    a_scores: list[float] = []
    a_pools: dict[str, float] = {}
    if not (PIX_CPU.is_file() and has_vmaf(PIX_CPU)):
        add({"name": "path_a_raw_oracle_vmaf", "status": "NOT EXECUTED", "detail": "cpu binary without VMAF"})
    else:
        rc, out = run_pix(
            PIX_CPU,
            [
                *model_args,
                "-v",
                "1",
                "-n",
                str(N),
                "-r",
                str(ref),
                *size_fmt,
                "-t",
                str(oracle),
                *size_fmt],
            {"PIXVQA_VMAF_FEATURES": "1"},
        )
        if rc != 0:
            add({"name": "path_a_raw_oracle_vmaf", "status": "FAILED", "detail": out[-800:]})
        else:
            a_scores, a_pools, nf = parse_vmaf(out)
            add(
                {
                    "name": "path_a_raw_oracle_vmaf",
                    "status": "PASSED" if nf == N else "FAILED",
                    "backend": "raw",
                    "binary": str(PIX_CPU),
                    "frames": nf,
                    "pools": a_pools,
                    "ref_sha256": sha256_file(ref),
                    "dist_sha256": sha256_file(oracle),
                    "detail": "" if nf == N else f"frames {nf} != {N}",
                }
            )

    # --- Path B: software HEVC ---
    sw_dump_sha = None
    if not (PIX_SW.is_file() and has_vmaf(PIX_SW)):
        add(
            {
                "name": "path_b_software_hevc_vmaf",
                "status": "NOT EXECUTED",
                "detail": "swdec binary missing or without VMAF",
            }
        )
    elif not a_scores:
        add(
            {
                "name": "path_b_software_hevc_vmaf",
                "status": "NOT EXECUTED",
                "detail": "path A baseline unavailable",
            }
        )
    else:
        dump = FIXTURE / "dist_swdec_dump.yuv"
        if dump.is_file():
            dump.unlink()
        rc, out = run_pix(
            PIX_SW,
            [
                *model_args,
                "--decode", "cpu",
                "-v",
                "1",
                "-n",
                str(N),
                "-r",
                str(ref),
                *size_fmt,
                "-t",
                str(hevc),
                *size_fmt],
            {"PIXVQA_VMAF_FEATURES": "1", "PIXVQA_DECODE_YUV": str(dump)},
        )
        if rc != 0:
            add({"name": "path_b_software_hevc_vmaf", "status": "FAILED", "detail": out[-1200:]})
        else:
            b_scores, b_pools, nf = parse_vmaf(out)
            cmp = compare_scores(a_scores, b_scores, a_pools, b_pools, "B_vs_A")
            dump_sha = sha256_file(dump) if dump.is_file() else None
            dump_size = dump.stat().st_size if dump.is_file() else None
            oracle_sha = sha256_file(oracle)
            id_fails = decode_identity_failures(
                dump_sha=dump_sha,
                oracle_sha=oracle_sha,
                dump_size=dump_size,
                expected_size=expected_size,
                frame_count=nf,
                expected_frames=N,
            )
            sw_dump_sha = dump_sha
            status = "PASSED" if cmp["status"] == "PASSED" and not id_fails else "FAILED"
            add(
                {
                    "name": "path_b_software_hevc_vmaf",
                    "status": status,
                    "backend": "software",
                    "binary": str(PIX_SW),
                    "hevc_sha256": sha256_file(hevc),
                    "decoded_dump_sha256": dump_sha,
                    "oracle_sha256": oracle_sha,
                    "decoded_dump_bytes": dump_size,
                    "expected_yuv_bytes": expected_size,
                    "byte_identity_vs_oracle": not id_fails,
                    "byte_identity_failures": id_fails,
                    "vmaf_score_parity_vs_path_a": cmp,
                    "pools": b_pools,
                    "detail": "; ".join(id_fails) if id_fails else "",
                }
            )

    # --- Path C: NVDEC ---
    if not (PIX_FULL.is_file() and has_vmaf(PIX_FULL)):
        add(
            {
                "name": "path_c_nvdec_host_vmaf",
                "status": "NOT EXECUTED",
                "detail": "full binary missing or without VMAF",
            }
        )
    elif not a_scores:
        add(
            {
                "name": "path_c_nvdec_host_vmaf",
                "status": "NOT EXECUTED",
                "detail": "path A baseline unavailable",
            }
        )
    else:
        dump = FIXTURE / "dist_nvdec_dump.yuv"
        if dump.is_file():
            dump.unlink()
        # VMAF-only --metric-device gpu uses CPU libvmaf (see vmaf_gpu_uses_cpu).
        # This path is NVDEC decode with host VMAF, so the metric device stays CPU.
        rc, out = run_pix(
            PIX_FULL,
            [
                *model_args,
                "--decode", "gpu",
                "-v",
                "1",
                "-n",
                str(N),
                "-r",
                str(ref),
                *size_fmt,
                "-t",
                str(hevc),
                *size_fmt],
            {
                "PIXVQA_VMAF_FEATURES": "1",
                "PIXVQA_DECODE_YUV": str(dump),
                "PIXVQA_DECODE_PROFILE": "1",
            },
        )
        if rc != 0:
            add({"name": "path_c_nvdec_host_vmaf", "status": "FAILED", "detail": out[-1500:]})
        else:
            c_scores, c_pools, nf = parse_vmaf(out)
            cmp_a = compare_scores(a_scores, c_scores, a_pools, c_pools, "C_vs_A")
            dump_sha = sha256_file(dump) if dump.is_file() else None
            dump_size = dump.stat().st_size if dump.is_file() else None
            oracle_sha = sha256_file(oracle)
            id_fails = decode_identity_failures(
                dump_sha=dump_sha,
                oracle_sha=oracle_sha,
                dump_size=dump_size,
                expected_size=expected_size,
                frame_count=nf,
                expected_frames=N,
                also_match_sha=sw_dump_sha,
            )
            dec = RE_DEC_PROF.search(out)
            status = "PASSED" if cmp_a["status"] == "PASSED" and not id_fails else "FAILED"
            add(
                {
                    "name": "path_c_nvdec_host_vmaf",
                    "status": status,
                    "backend": "nvdec_host_download",
                    "binary": str(PIX_FULL),
                    "cpu_vmaf": True,
                    "decoded_dump_sha256": dump_sha,
                    "oracle_sha256": oracle_sha,
                    "software_dump_sha256": sw_dump_sha,
                    "decoded_dump_bytes": dump_size,
                    "expected_yuv_bytes": expected_size,
                    "byte_identity_vs_oracle": dump_sha == oracle_sha if dump_sha else False,
                    "byte_identity_vs_software_dump": dump_sha == sw_dump_sha if dump_sha and sw_dump_sha else False,
                    "byte_identity_failures": id_fails,
                    "vmaf_score_parity_vs_path_a": cmp_a,
                    "pools": c_pools,
                    "decode_profile": {
                        "nvdec": int(dec.group(5)),
                        "frames": int(dec.group(4)),
                        "download_seconds": float(dec.group(3)),
                    }
                    if dec
                    else None,
                    "note": "VMAF remains CPU libvmaf after NVDEC→host download",
                    "detail": "; ".join(id_fails) if id_fails else "",
                }
            )

        rc_fb, out_fb = run_pix(
            PIX_SW,
            [
                *model_args,
                "--decode", "gpu",
                "-v",
                "0",
                "-n",
                "1",
                "-r",
                str(ref),
                *size_fmt,
                "-t",
                str(hevc),
                *size_fmt],
        )
        add(
            {
                "name": "nvdec_no_silent_fallback_on_swdec",
                "status": "PASSED" if rc_fb != 0 else "FAILED",
                "detail": "swdec correctly rejects --decode nvdec" if rc_fb != 0 else out_fb[-400:],
            }
        )

    # --- Path D: CUDA metrics ± VMAF on raw YUV (full binary preferred) ---
    cuda_bin = PIX_FULL if (PIX_FULL.is_file() and has_vmaf(PIX_FULL)) else None
    include_xpsnr = bool(cuda_bin and supports_cuda_xpsnr(cuda_bin))
    if cuda_bin is None:
        add(
            {
                "name": "path_d_cuda_metrics_plus_cpu_vmaf_raw",
                "status": "NOT EXECUTED",
                "detail": "no full/CUDA binary with VMAF for Path D",
            }
        )
    else:
        base = [
            *cuda_metric_flags(include_xpsnr),
            "-n",
            str(N),
            "-r",
            str(ref),
            *size_fmt,
            "-t",
            str(oracle),
            *size_fmt]
        env = {"PIXVQA_CUDA_PROFILE": "1"}
        rc0, out0 = run_pix(cuda_bin, base, env)
        rc1, out1 = run_pix(cuda_bin, base + model_args, env)
        if rc0 != 0 or rc1 != 0:
            add(
                {
                    "name": "path_d_cuda_metrics_plus_cpu_vmaf_raw",
                    "status": "FAILED",
                    "detail": f"rc0={rc0} rc1={rc1}\n" + (out0[-600:] + out1[-800:]),
                    "xpsnr_included": include_xpsnr,
                }
            )
        else:
            d0 = parse_cuda_metrics(out0, include_xpsnr)
            d1 = parse_cuda_metrics(out1, include_xpsnr)
            fails = validate_cuda_run(d0, N, False, "D0") + validate_cuda_run(d1, N, True, "D1")
            rows, cmp_fails = compare_cuda_pair(d0, d1, N, include_xpsnr)
            fails.extend(cmp_fails)
            # Ordering: on VMAF runs, VMAF token follows metrics on the same line.
            order_ok = True
            for line in out1.splitlines():
                if re.match(r"^\d+\s*:", line) and "[ VMAF:" in line:
                    if line.find("[ VMAF:") < line.find("[ PSNR"):
                        order_ok = False
            if not order_ok:
                fails.append("VMAF appeared before PSNR on a frame line")
            add(
                {
                    "name": "path_d_cuda_metrics_plus_cpu_vmaf_raw",
                    "status": "PASSED" if not fails else "FAILED",
                    "binary": str(cuda_bin),
                    "input_backend": "raw",
                    "cuda_configuration": {
                        "kernel": "cuda",
                        "cuda_prec": "fp64",
                        "xpsnr_backend": "cuda" if include_xpsnr else None,
                    },
                    "model_path": str(MODEL),
                    "ref_sha256": sha256_file(ref),
                    "dist_sha256": sha256_file(oracle),
                    "frames": N,
                    "xpsnr_included": include_xpsnr,
                    "xpsnr_exclusion_reason": None
                    if include_xpsnr
                    else "build help did not advertise CUDA XPSNR",
                    "cuda_profile_without_vmaf": d0["cuda_profile"],
                    "cuda_profile_with_vmaf": d1["cuda_profile"],
                    "metric_comparisons": rows,
                    "detail": "; ".join(fails),
                }
            )

        # Reject cuda VMAF backend
        rc_c, out_c = run_pix(
            cuda_bin,
            [
                "--metrics", "vmaf",
                "--metric-device",
                "gpu",
                "--model",
                str(MODEL),
                "-v",
                "0",
                "-r",
                str(ref),
                *size_fmt,
                "-t",
                str(oracle),
                *size_fmt],
        )
        add(
            {
                "name": "vmaf_gpu_uses_cpu",
                "status": "PASSED"
                if rc_c == 0 and "using CPU libvmaf" in out_c and "backend=cpu" in out_c
                else "FAILED",
                "detail": out_c[-300:],
            }
        )

    # --- Path D2/D3: NVDEC + CUDA metrics ± VMAF ---
    if not (PIX_FULL.is_file() and has_vmaf(PIX_FULL)):
        add(
            {
                "name": "path_d_nvdec_cuda_metrics_plus_cpu_vmaf",
                "status": "NOT EXECUTED",
                "detail": "full binary missing",
            }
        )
    elif not a_scores:
        add(
            {
                "name": "path_d_nvdec_cuda_metrics_plus_cpu_vmaf",
                "status": "NOT EXECUTED",
                "detail": "path A baseline unavailable",
            }
        )
    else:
        include_xpsnr_nv = supports_cuda_xpsnr(PIX_FULL)
        base_nv = [
            *cuda_metric_flags(include_xpsnr_nv),
            "--decode", "gpu",
            "-n",
            str(N),
            "-r",
            str(ref),
            *size_fmt,
            "-t",
            str(hevc),
            *size_fmt]
        env_d2 = {"PIXVQA_CUDA_PROFILE": "1", "PIXVQA_DECODE_PROFILE": "1"}
        env_d3 = {
            "PIXVQA_CUDA_PROFILE": "1",
            "PIXVQA_DECODE_PROFILE": "1",
            "PIXVQA_VMAF_FEATURES": "1",
            "PIXVQA_DECODE_YUV": str(FIXTURE / "dist_nvdec_d3_dump.yuv"),
        }
        rc2, out2 = run_pix(PIX_FULL, base_nv, env_d2)
        rc3, out3 = run_pix(PIX_FULL, base_nv + model_args, env_d3)
        if rc2 != 0 or rc3 != 0:
            add(
                {
                    "name": "path_d_nvdec_cuda_metrics_plus_cpu_vmaf",
                    "status": "FAILED",
                    "detail": f"rc_d2={rc2} rc_d3={rc3}\n" + (out2[-700:] + out3[-900:]),
                }
            )
        else:
            d2 = parse_cuda_metrics(out2, include_xpsnr_nv)
            d3 = parse_cuda_metrics(out3, include_xpsnr_nv)
            fails = validate_cuda_run(d2, N, False, "D2") + validate_cuda_run(d3, N, True, "D3")
            rows, cmp_fails = compare_cuda_pair(d2, d3, N, include_xpsnr_nv)
            fails.extend(cmp_fails)
            dec2 = RE_DEC_PROF.search(out2)
            dec3 = RE_DEC_PROF.search(out3)
            if not dec2 or int(dec2.group(5)) != 1:
                fails.append("D2: nvdec!=1 in decode_profile")
            if not dec3 or int(dec3.group(5)) != 1:
                fails.append("D3: nvdec!=1 in decode_profile")
            if dec3 and float(dec3.group(3)) <= 0.0:
                fails.append("D3: download_seconds <= 0 (VMAF should force host download)")
            # D2 is device-resident: dump path is unavailable; D3 dump must match oracle.
            dump3 = FIXTURE / "dist_nvdec_d3_dump.yuv"
            d3_sha = sha256_file(dump3) if dump3.is_file() else None
            oracle_sha = sha256_file(oracle)
            if d3_sha != oracle_sha:
                fails.append("D3 dump sha != oracle")
            vmaf_scores, vmaf_pools, nf = parse_vmaf(out3)
            cmp_a = compare_scores(a_scores, vmaf_scores, a_pools, vmaf_pools, "D3_vmaf_vs_A")
            if cmp_a["status"] != "PASSED":
                fails.append("D3 VMAF parity vs A failed")
            add(
                {
                    "name": "path_d_nvdec_cuda_metrics_plus_cpu_vmaf",
                    "status": "PASSED" if not fails else "FAILED",
                    "binary": str(PIX_FULL),
                    "input_backend": "nvdec",
                    "cuda_configuration": {
                        "kernel": "cuda",
                        "cuda_prec": "fp64",
                        "xpsnr_backend": "cuda" if include_xpsnr_nv else None,
                    },
                    "model_path": str(MODEL),
                    "hevc_sha256": sha256_file(hevc),
                    "frames": N,
                    "xpsnr_included": include_xpsnr_nv,
                    "d2_decode_profile": {
                        "nvdec": int(dec2.group(5)),
                        "download_seconds": float(dec2.group(3)),
                        "frames": int(dec2.group(4)),
                    }
                    if dec2
                    else None,
                    "d3_decode_profile": {
                        "nvdec": int(dec3.group(5)),
                        "download_seconds": float(dec3.group(3)),
                        "frames": int(dec3.group(4)),
                    }
                    if dec3
                    else None,
                    "cuda_profile_d2": d2["cuda_profile"],
                    "cuda_profile_d3": d3["cuda_profile"],
                    "metric_comparisons": rows,
                    "d3_dump_sha256": d3_sha,
                    "oracle_sha256": oracle_sha,
                    "vmaf_score_parity_vs_path_a": cmp_a,
                    "note": (
                        "D2 keeps NVDEC device-resident CUDA scoring; D3 forces host download "
                        "for CPU VMAF while CUDA metrics still score the device planes."
                    ),
                    "detail": "; ".join(fails),
                }
            )

    # --- CUDA-only binary ---
    if not PIX_CUDA.is_file():
        add({"name": "path_cuda_only_raw_vmaf", "status": "NOT EXECUTED", "detail": "build/cuda missing"})
    elif not has_vmaf(PIX_CUDA):
        add(
            {
                "name": "path_cuda_only_raw_vmaf",
                "status": "FAILED",
                "detail": "build/full/pixvqa is not linked against libvmaf (help lacks --vmaf)",
            }
        )
    elif not a_scores:
        add(
            {
                "name": "path_cuda_only_raw_vmaf",
                "status": "NOT EXECUTED",
                "detail": "path A baseline unavailable",
            }
        )
    else:
        include_x_cu = supports_cuda_xpsnr(PIX_CUDA)
        # Focused: at least one CUDA metric + CPU VMAF; also compare PSNR coexistence.
        base = [
            "--metric-device", "gpu",
            "--metrics", "psnr",
            "-v",
            "1",
            "-n",
            str(N),
            "-r",
            str(ref),
            *size_fmt,
            "-t",
            str(oracle),
            *size_fmt]
        env = {"PIXVQA_CUDA_PROFILE": "1"}
        rc0, out0 = run_pix(PIX_CUDA, base, env)
        rc1, out1 = run_pix(PIX_CUDA, base + model_args, env)
        fails = []
        if rc0 != 0 or rc1 != 0:
            fails.append(f"rc0={rc0} rc1={rc1}")
            add(
                {
                    "name": "path_cuda_only_raw_vmaf",
                    "status": "FAILED",
                    "binary": str(PIX_CUDA),
                    "detail": "; ".join(fails) + "\n" + (out0[-500:] + out1[-700:]),
                }
            )
        else:
            d0 = parse_cuda_metrics(out0, False)
            d1 = parse_cuda_metrics(out1, False)
            # Only PSNR/MSE required for this focused case.
            for label, parsed, want_vmaf in (("CU0", d0, False), ("CU1", d1, True)):
                if parsed["frames_total"] != N:
                    fails.append(f"{label}: frames")
                if len(parsed["per_frame"]["psnr"]) != N or parsed["summary"]["psnr"] is None:
                    fails.append(f"{label}: missing PSNR")
                if parsed["cuda_profile"] is None or parsed["cuda_profile"]["kernel_ms"] <= 0:
                    fails.append(f"{label}: CUDA profile/kernel")
                if want_vmaf and not parsed["has_vmaf_summary"]:
                    fails.append(f"{label}: VMAF missing")
                if not want_vmaf and parsed["has_vmaf_summary"]:
                    fails.append(f"{label}: unexpected VMAF")
            row = compare_metric_pair(
                "psnr",
                d0["per_frame"]["psnr"],
                d1["per_frame"]["psnr"],
                d0["summary"]["psnr"],
                d1["summary"]["psnr"],
                N,
            )
            if row["status"] != "PASSED":
                fails.append("PSNR coexistence failed")
            vmaf_scores, vmaf_pools, nf = parse_vmaf(out1)
            cmp_a = compare_scores(a_scores, vmaf_scores, a_pools, vmaf_pools, "cuda_only_vs_A")
            if cmp_a["status"] != "PASSED":
                fails.append("VMAF vs Path A failed")
            add(
                {
                    "name": "path_cuda_only_raw_vmaf",
                    "status": "PASSED" if not fails else "FAILED",
                    "binary": str(PIX_CUDA),
                    "input_backend": "raw",
                    "decode_support": "none",
                    "cuda_configuration": {"kernel": "cuda", "cuda_prec": "fp64"},
                    "model_path": str(MODEL),
                    "libvmaf_linked": True,
                    "xpsnr_supported_in_binary": include_x_cu,
                    "metric_comparisons": [row],
                    "vmaf_score_parity_vs_path_a": cmp_a,
                    "cuda_profile_without_vmaf": d0["cuda_profile"],
                    "cuda_profile_with_vmaf": d1["cuda_profile"],
                    "detail": "; ".join(fails),
                }
            )

    # --- Path E: FIFO with timeout / producer failure handling ---
    if not (PIX_CPU.is_file() and has_vmaf(PIX_CPU) and a_scores):
        add({"name": "path_e_fifo_raw_vmaf", "status": "NOT EXECUTED", "detail": "cpu/VMAF/baseline missing"})
    else:
        with tempfile.TemporaryDirectory(prefix="vmaf_fifo_") as td:
            td_path = Path(td)
            fifo = td_path / "dist.fifo"
            os.mkfifo(fifo)
            producer_err: list[str] = []

            def producer() -> None:
                try:
                    data = oracle.read_bytes()
                    with fifo.open("wb") as f:
                        f.write(data)
                except Exception as ex:  # noqa: BLE001 — surface in evidence
                    producer_err.append(str(ex))

            th = threading.Thread(target=producer, daemon=True)
            th.start()
            t0 = time.perf_counter()
            rc, out = run_pix(
                PIX_CPU,
                [
                    *model_args,
                    "-v",
                    "1",
                    "-n",
                    str(N),
                    "-r",
                    str(ref),
                    *size_fmt,
                    "-t",
                    str(fifo),
                    *size_fmt],
                {"PIXVQA_VMAF_FEATURES": "1"},
                timeout=FIFO_TIMEOUT_S,
            )
            t1 = time.perf_counter()
            th.join(timeout=5)
            if th.is_alive():
                # Reader side ended; producer should not hang. Force FAIL.
                add(
                    {
                        "name": "path_e_fifo_raw_vmaf",
                        "status": "FAILED",
                        "detail": "producer thread still alive after consumer exit",
                        "wall_s": t1 - t0,
                    }
                )
            elif rc == 124:
                add(
                    {
                        "name": "path_e_fifo_raw_vmaf",
                        "status": "FAILED",
                        "detail": f"consumer timed out after {FIFO_TIMEOUT_S}s",
                        "wall_s": t1 - t0,
                    }
                )
            elif producer_err:
                add(
                    {
                        "name": "path_e_fifo_raw_vmaf",
                        "status": "FAILED",
                        "detail": "producer error: " + "; ".join(producer_err),
                        "wall_s": t1 - t0,
                    }
                )
            elif rc != 0:
                add({"name": "path_e_fifo_raw_vmaf", "status": "FAILED", "detail": out[-800:]})
            else:
                e_scores, e_pools, nf = parse_vmaf(out)
                cmp = compare_scores(a_scores, e_scores, a_pools, e_pools, "E_vs_A")
                add(
                    {
                        "name": "path_e_fifo_raw_vmaf",
                        "status": cmp["status"],
                        "frames": nf,
                        "wall_s": t1 - t0,
                        "consumer_timeout_s": FIFO_TIMEOUT_S,
                        "vmaf_vs_path_a": cmp,
                        "note": "raw YUV over FIFO; finite consumer timeout; temporal flush after EOF",
                    }
                )

        # Stalled-producer negative: open FIFO with no writer → consumer must time out / fail, not hang forever.
        with tempfile.TemporaryDirectory(prefix="vmaf_fifo_stall_") as td:
            fifo = Path(td) / "stall.fifo"
            os.mkfifo(fifo)
            # No producer. Short timeout.
            rc, out = run_pix(
                PIX_CPU,
                [
                    *model_args,
                    "-v",
                    "0",
                    "-n",
                    str(N),
                    "-r",
                    str(ref),
                    *size_fmt,
                    "-t",
                    str(fifo),
                    *size_fmt],
                timeout=min(5.0, FIFO_TIMEOUT_S),
            )
            add(
                {
                    "name": "path_e_fifo_stalled_producer_timeout",
                    "status": "PASSED" if rc != 0 else "FAILED",
                    "detail": (
                        f"rc={rc} (expect non-zero / timeout when no producer writes)"
                        if rc != 0
                        else "consumer succeeded with no producer — unexpected"
                    ),
                    "consumer_timeout_s": min(5.0, FIFO_TIMEOUT_S),
                }
            )

    out_path = ROOT / "reports" / "vmaf_gate_b_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    passed = sum(1 for c in results["cases"] if c["status"] == "PASSED")
    failed = sum(1 for c in results["cases"] if c["status"] == "FAILED")
    skipped = sum(1 for c in results["cases"] if c["status"] == "NOT EXECUTED")
    print(f"summary PASSED={passed} FAILED={failed} NOT_EXECUTED={skipped}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
