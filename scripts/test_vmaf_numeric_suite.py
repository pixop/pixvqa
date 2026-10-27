#!/usr/bin/env python3
"""Representative VMAF numerical suite: temporal coverage + unclipped scores.

Oracle A: official `vmaf` CLI JSON (6 decimal digits) — TOL_SCORE 1e-6, TOL_FEAT 5e-7.
Oracle B: `vmaf_ref_oracle` (libvmaf C API, %.17g) — same absolute tolerances.

Retains serialized-output acceptance; does not claim bit-exact internal equality.
Evidence: reports/vmaf_numeric_suite_results.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXVQA = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
VMAF_CLI = Path(os.environ.get("VMAF_CLI", ROOT / "deps" / "libvmaf-74f930d" / "bin" / "vmaf"))
ORACLE = Path(os.environ.get("VMAF_REF_ORACLE", ROOT / "build" / "cpu" / "vmaf_ref_oracle"))
MODEL_DIR = ROOT / "model"

MODELS = {
    "vmaf_v1.0.16_3d0h.json": {
        "clip_hi": 100.0,
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02",
        "motion3": "integer_motion3_mmxv_18",
    },
    "vmaf_v1.0.16_5d0h.json": {
        "clip_hi": 100.0,
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5",
        "motion3": "integer_motion3_mmxv_18",
    },
    "vmaf_v1.0.16_1d5h_2160.json": {
        "clip_hi": 100.0,
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160",
        "motion3": "integer_motion3_mmxv_18",
    },
    "vmaf_v1.0.16_3d0h_2160.json": {
        "clip_hi": 110.0,
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160",
        "motion3": "integer_motion3_mmxv_18",
    },
    "vmaf_v1.0.16_hfr_3d0h.json": {
        "clip_hi": 100.0,
        "hfr": True,
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02",
        "motion3": "integer_motion3_mffw_mmxv_18_mma",
    },
    "vmaf_v1.0.16_hfr_5d0h.json": {
        "clip_hi": 100.0,
        "hfr": True,
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5",
        "motion3": "integer_motion3_mffw_mmxv_18_mma",
    },
    "vmaf_v1.0.16_hfr_1d5h_2160.json": {
        "clip_hi": 100.0,
        "hfr": True,
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160",
        "motion3": "integer_motion3_mffw_mmxv_18_mma",
    },
    "vmaf_v1.0.16_hfr_3d0h_2160.json": {
        "clip_hi": 110.0,
        "hfr": True,
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160",
        "motion3": "integer_motion3_mffw_mmxv_18_mma",
    },
}

W, H = 1920, 1080
TOL_SCORE = 1e-6
TOL_FEAT = 5e-7
MARGIN = 2.0  # unclipped: away from 0 and clip_hi by this margin


def run(cmd: list[str], env: dict[str, str] | None = None) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write("FAIL CMD " + " ".join(cmd) + "\n" + p.stderr + p.stdout)
        raise SystemExit(p.returncode)
    return p.stdout + "\n" + p.stderr


def write_pair(ref: Path, dist: Path, n: int, kind: str) -> tuple[str, str]:
    """Generate 10-bit 420 sequences with controlled degradation.

    kind:
      noise     — strong spatial luma/chroma noise (compression-like)
      blocky    — 16×16 quantization (spatial blockiness)
      temporal  — scene change + motion (for HFR / motion3)
    """
    rh = hashlib.sha256()
    dh = hashlib.sha256()
    yw, yh = W, H
    cw, ch = W // 2, H // 2
    y_count = yw * yh
    c_count = cw * ch

    with ref.open("wb") as fr, dist.open("wb") as fd:
        for fi in range(n):
            scene = 1 if (kind == "temporal" and fi >= max(1, n // 2)) else 0
            y_ref = bytearray(y_count * 2)
            y_dist = bytearray(y_count * 2)
            # Fill 16×16 blocks to keep generation fast while remaining structured.
            for by in range(0, yh, 16):
                for bx in range(0, yw, 16):
                    base = (180 + scene * 400 + (by * 3 + bx * 5 + fi * 17) % 400) & 0x3FF
                    if kind == "noise":
                        dlt = ((by * 13 + bx * 7 + fi * 31) % 161) - 80
                        dv = max(0, min(1023, base + dlt))
                    elif kind == "blocky":
                        dv = ((base // 64) * 64) & 0x3FF
                    else:
                        # Strong temporal churn: large shift, scene cut, residual noise.
                        shift = (fi * 37 + scene * 320) % 220
                        dlt = ((by * 5 + bx * 3 + fi * 11) % 121) - 60
                        dv = max(0, min(1023, base + shift + dlt))
                    rb = struct.pack("<H", base)
                    db = struct.pack("<H", dv)
                    for row in range(by, min(by + 16, yh)):
                        off = (row * yw + bx) * 2
                        npx = min(16, yw - bx)
                        y_ref[off : off + npx * 2] = rb * npx
                        y_dist[off : off + npx * 2] = db * npx
            fr.write(y_ref)
            fd.write(y_dist)
            rh.update(y_ref)
            dh.update(y_dist)
            for plane_seed in (512, 520):
                c_ref = bytearray(c_count * 2)
                c_dist = bytearray(c_count * 2)
                for by in range(0, ch, 8):
                    for bx in range(0, cw, 8):
                        u = (plane_seed + scene * 40 + (by + bx + fi) % 40) & 0x3FF
                        if kind == "noise":
                            du = max(0, min(1023, u + ((by * 3 + fi) % 41) - 20))
                        elif kind == "blocky":
                            du = ((u // 32) * 32) & 0x3FF
                        else:
                            du = max(0, min(1023, u + ((fi + by) % 15) - 7))
                        rb = struct.pack("<H", u)
                        db = struct.pack("<H", du)
                        for row in range(by, min(by + 8, ch)):
                            off = (row * cw + bx) * 2
                            npx = min(8, cw - bx)
                            c_ref[off : off + npx * 2] = rb * npx
                            c_dist[off : off + npx * 2] = db * npx
                fr.write(c_ref)
                fd.write(c_dist)
                rh.update(c_ref)
                dh.update(c_dist)
    return rh.hexdigest(), dh.hexdigest()


def parse_pixvqa(
    out: str, n: int
) -> tuple[list[float], dict[str, float], int, dict[str, list[float]]]:
    # Prefer %.17g scores from PIXVQA_VMAF_FEATURES; fall back to %.6f banners.
    feats: dict[str, list[float]] = {k: [] for k in ("cambi", "speed_chroma", "adm3", "motion3")}
    per: list[float] = []
    for line in out.splitlines():
        if not line.startswith("vmaf_features "):
            continue
        kv = dict(re.findall(r"(\w+)=([-+0-9.eE]+)", line))
        if "score" in kv:
            per.append(float(kv["score"]))
        for k in feats:
            feats[k].append(float(kv[k]))
    if len(per) != n or any(len(v) != n for v in feats.values()):
        per = [float(x) for x in re.findall(r"\[ VMAF:([0-9.]+) \]", out)]
        if len(per) != n:
            raise SystemExit(f"bad pixvqa output (n={n}):\n{out}")
    m = re.search(
        r"\[ VMAF mean:([0-9.]+) harmonic:([0-9.]+) min:([0-9.]+) max:([0-9.]+) frames:(\d+) \]",
        out,
    )
    if not m:
        raise SystemExit(f"missing pooled VMAF:\n{out}")
    pools = {
        "mean": float(m.group(1)),
        "harmonic_mean": float(m.group(2)),
        "min": float(m.group(3)),
        "max": float(m.group(4)),
    }
    frames_reported = int(m.group(5))
    return per, pools, frames_reported, feats


def harmonic_sanity(scores: list[float]) -> float:
    """libvmaf 3.2.1 formula: N / sum(1/(score[i]+1)) - 1 (sanity only)."""
    n = len(scores)
    return n / sum(1.0 / (s + 1.0) for s in scores) - 1.0


def model_sha256(model: str) -> str:
    return hashlib.sha256((MODEL_DIR / model).read_bytes()).hexdigest()


def compare_cli(model: str, keys: dict, ref: Path, dist: Path, n: int, td: Path) -> dict:
    oracle = td / f"cli_{model}_{n}.json"
    run(
        [
            str(VMAF_CLI),
            "-r",
            str(ref),
            "-d",
            str(dist),
            "-w",
            str(W),
            "-h",
            str(H),
            "-p",
            "420",
            "-b",
            "10",
            "--model",
            f"path={MODEL_DIR / model}",
            "--json",
            "-o",
            str(oracle),
            "--threads",
            "8",
            "--quiet"]
    )
    data = json.loads(oracle.read_text(encoding="utf-8"))
    o_scores = [float(fr["metrics"]["vmaf"]) for fr in data["frames"]]
    o_feats = {nme: [float(fr["metrics"][keys[nme]]) for fr in data["frames"]] for nme in (
        "cambi", "speed_chroma", "adm3", "motion3"
    )}
    o_pooled = float(data["pooled_metrics"]["vmaf"]["mean"])

    env = os.environ.copy()
    env["PIXVQA_VMAF_FEATURES"] = "1"
    out = run(
        [
            str(PIXVQA),
            "--metrics", "vmaf",
            "--model",
            str(MODEL_DIR / model),
            
            "--threads",
            "8",
            "-v",
            "1",
            "-n",
            str(n),
            "-r",
            str(ref),
            "-s",
            f"{W}x{H}",
            "-f",
            "420p:10",
            "-t",
            str(dist),
            "-s",
            f"{W}x{H}",
            "-f",
            "420p:10"],
        env=env,
    )
    p_scores, p_pools, frames_n, p_feats = parse_pixvqa(out, n)
    if frames_n != n:
        raise SystemExit(f"{model} n={n}: CLI frames mismatch summary={frames_n}")

    max_score = max(abs(a - b) for a, b in zip(p_scores, o_scores))
    max_score = max(max_score, abs(p_pools["mean"] - o_pooled))
    max_feat = 0.0
    for name in p_feats:
        for a, b in zip(p_feats[name], o_feats[name]):
            max_feat = max(max_feat, abs(a - b))
    if max_score > TOL_SCORE or max_feat > TOL_FEAT:
        raise SystemExit(
            f"{model} n={n}: CLI FAIL max_score={max_score:.3e} max_feat={max_feat:.3e}"
        )
    return {
        "oracle": "vmaf_cli_json_6dec",
        "scores_pixvqa": p_scores,
        "scores_oracle": o_scores,
        "pooled_pixvqa": p_pools,
        "pooled_oracle_mean": o_pooled,
        "max_abs_score": max_score,
        "max_abs_feature": max_feat,
        "features_pixvqa": p_feats,
        "features_oracle": o_feats,
        "frames": frames_n,
    }


def compare_c_api(model: str, keys: dict, ref: Path, dist: Path, n: int) -> dict:
    raw = run(
        [
            str(ORACLE),
            "--model",
            str(MODEL_DIR / model),
            "--ref",
            str(ref),
            "--dist",
            str(dist),
            "--width",
            str(W),
            "--height",
            str(H),
            "--bitdepth",
            "10",
            "--frames",
            str(n),
            "--threads",
            "8",
            "--feat-cambi",
            keys["cambi"],
            "--feat-speed",
            keys["speed_chroma"],
            "--feat-adm3",
            keys["adm3"],
            "--feat-motion3",
            keys["motion3"]]
    )
    # Oracle prints JSON on stdout; ignore any stray stderr.
    start = raw.find("{")
    end = raw.rfind("}")
    data = json.loads(raw[start : end + 1])
    o_scores = [float(fr["vmaf"]) for fr in data["frames"]]
    o_feats = {
        k: [float(fr[k]) for fr in data["frames"]]
        for k in ("cambi", "speed_chroma", "adm3", "motion3")
    }
    if "pooled" in data:
        o_pools = {k: float(data["pooled"][k]) for k in ("mean", "harmonic_mean", "min", "max")}
    else:
        # Backward-compatible: older oracle only emitted pooled_mean.
        o_pools = {
            "mean": float(data["pooled_mean"]),
            "harmonic_mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    if any(v != v for v in o_pools.values()):  # NaN check
        raise SystemExit(f"{model}: oracle missing pooled.{{mean,harmonic_mean,min,max}}")

    env = os.environ.copy()
    env["PIXVQA_VMAF_FEATURES"] = "1"
    out = run(
        [
            str(PIXVQA),
            "--metrics", "vmaf",
            "--model",
            str(MODEL_DIR / model),
            
            "--threads",
            "8",
            "-v",
            "1",
            "-n",
            str(n),
            "-r",
            str(ref),
            "-s",
            f"{W}x{H}",
            "-f",
            "420p:10",
            "-t",
            str(dist),
            "-s",
            f"{W}x{H}",
            "-f",
            "420p:10"],
        env=env,
    )
    p_scores, p_pools, frames_n, p_feats = parse_pixvqa(out, n)
    if frames_n != n:
        raise SystemExit(f"{model} n={n}: frames summary={frames_n} expected={n}")

    # Per-frame + four pooled methods (PixVQA summary is %.6f vs oracle %.17g).
    max_score = max(abs(a - b) for a, b in zip(p_scores, o_scores))
    pool_abs = {k: abs(p_pools[k] - o_pools[k]) for k in o_pools}
    for k, d in pool_abs.items():
        if d > TOL_SCORE:
            raise SystemExit(
                f"{model} n={n}: C-API pool {k} FAIL "
                f"pixvqa={p_pools[k]} oracle={o_pools[k]} abs={d:.3e}"
            )
        max_score = max(max_score, d)

    # Min/max must match extrema of the per-frame vector (high-precision scores).
    if abs(min(p_scores) - p_pools["min"]) > TOL_SCORE:
        raise SystemExit(f"{model}: pixvqa min {p_pools['min']} != min(scores) {min(p_scores)}")
    if abs(max(p_scores) - p_pools["max"]) > TOL_SCORE:
        raise SystemExit(f"{model}: pixvqa max {p_pools['max']} != max(scores) {max(p_scores)}")
    if abs(min(o_scores) - o_pools["min"]) > 1e-12:
        raise SystemExit(f"{model}: oracle min inconsistent with scores")
    if abs(max(o_scores) - o_pools["max"]) > 1e-12:
        raise SystemExit(f"{model}: oracle max inconsistent with scores")

    # Harmonic sanity vs libvmaf formula (not a PixVQA implementation).
    h_check = harmonic_sanity(o_scores)
    if abs(h_check - o_pools["harmonic_mean"]) > 1e-9:
        raise SystemExit(
            f"{model}: harmonic sanity {h_check} vs oracle {o_pools['harmonic_mean']}"
        )
    if n == 1 and abs(o_pools["harmonic_mean"] - o_scores[0]) > 1e-9:
        raise SystemExit(f"{model}: 1-frame harmonic != frame score")

    max_feat = 0.0
    for name in p_feats:
        for a, b in zip(p_feats[name], o_feats[name]):
            max_feat = max(max_feat, abs(a - b))
    if max_score > TOL_SCORE or max_feat > TOL_FEAT:
        raise SystemExit(
            f"{model} n={n}: C-API FAIL max_score={max_score:.3e} max_feat={max_feat:.3e}"
        )
    return {
        "oracle": "libvmaf_c_api_double",
        "libvmaf": data.get("libvmaf"),
        "scores_pixvqa": p_scores,
        "scores_oracle": o_scores,
        "pooled_pixvqa": p_pools,
        "pooled_oracle": o_pools,
        "pool_abs": pool_abs,
        "max_abs_score": max_score,
        "max_abs_feature": max_feat,
        "max_abs_pool": {k: pool_abs[k] for k in pool_abs},
        "frames": frames_n,
        "status": "PASSED",
    }


def assert_unclipped(scores: list[float], clip_hi: float, label: str) -> None:
    for i, s in enumerate(scores):
        if s <= MARGIN or s >= clip_hi - MARGIN:
            raise SystemExit(
                f"{label}: frame {i} score {s} not unclipped vs [0,{clip_hi}] "
                f"(need margin {MARGIN})"
            )


def main() -> int:
    if not PIXVQA.is_file() or not VMAF_CLI.is_file():
        print("missing pixvqa or vmaf cli", file=sys.stderr)
        return 2
    if not ORACLE.is_file():
        print(f"missing {ORACLE}; run scripts/build_vmaf_ref_oracle.sh", file=sys.stderr)
        return 2

    evidence: dict = {
        "methodology": {
            "cli_tolerance_score": TOL_SCORE,
            "cli_tolerance_feature": TOL_FEAT,
            "pool_tolerance_score": TOL_SCORE,
            "pool_tolerance_note": (
                "PixVQA summary prints %.6f; oracle uses %.17g. "
                "Comparisons are serialized-output acceptance, not bit-exact."
            ),
            "cli_precision_note": "Official vmaf --json prints 6 decimal digits",
            "c_api_oracle": str(ORACLE),
            "size": f"{W}x{H}",
            "bitdepth": 10,
            "chroma": "420",
            "pools": ["mean", "harmonic_mean", "min", "max"],
        },
        "cases": [],
        "pool_abs_max_global": {
            "mean": 0.0,
            "harmonic_mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
    }

    def record(case: dict, capi: dict) -> None:
        for k, d in capi["max_abs_pool"].items():
            evidence["pool_abs_max_global"][k] = max(evidence["pool_abs_max_global"][k], d)
        evidence["cases"].append(case)

    with tempfile.TemporaryDirectory(prefix="vmaf_suite_") as td:
        td_path = Path(td)

        # --- Temporal coverage (3d0h + HFR 3d0h) ---
        for n in (1, 2, 3, 5, 30):
            kind = "temporal" if n >= 5 else "noise"
            ref, dist = td_path / f"t{n}_ref.yuv", td_path / f"t{n}_dist.yuv"
            rh, dh = write_pair(ref, dist, n, kind)
            for model in ("vmaf_v1.0.16_3d0h.json", "vmaf_v1.0.16_hfr_3d0h.json"):
                print(f"temporal n={n} model={model} kind={kind} ...", flush=True)
                cli = compare_cli(model, MODELS[model], ref, dist, n, td_path)
                capi = compare_c_api(model, MODELS[model], ref, dist, n)
                record(
                    {
                        "case": "temporal",
                        "status": "PASSED",
                        "frames": n,
                        "kind": kind,
                        "model": model,
                        "model_sha256": model_sha256(model),
                        "ref_sha256": rh,
                        "dist_sha256": dh,
                        "scores_pixvqa": capi["scores_pixvqa"],
                        "scores_oracle": capi["scores_oracle"],
                        "pooled_pixvqa": capi["pooled_pixvqa"],
                        "pooled_oracle": capi["pooled_oracle"],
                        "pool_abs": capi["max_abs_pool"],
                        "cli_max_abs_score": cli["max_abs_score"],
                        "cli_max_abs_feature": cli["max_abs_feature"],
                        "c_api_max_abs_feature": capi["max_abs_feature"],
                    },
                    capi,
                )
                print(
                    f"  OK pools mean/h/min/max Δ="
                    f"{capi['max_abs_pool']['mean']:.3e}/"
                    f"{capi['max_abs_pool']['harmonic_mean']:.3e}/"
                    f"{capi['max_abs_pool']['min']:.3e}/"
                    f"{capi['max_abs_pool']['max']:.3e}"
                )

        # --- Unclipped coverage: all eight models, three distortion kinds ---
        for kind in ("noise", "blocky", "temporal"):
            n = 5 if kind != "temporal" else 8
            ref, dist = td_path / f"u_{kind}_ref.yuv", td_path / f"u_{kind}_dist.yuv"
            rh, dh = write_pair(ref, dist, n, kind)
            for model, meta in MODELS.items():
                print(f"unclipped kind={kind} model={model} ...", flush=True)
                cli = compare_cli(model, meta, ref, dist, n, td_path)
                assert_unclipped(cli["scores_pixvqa"], meta["clip_hi"], f"{model}/{kind}")
                assert_unclipped(
                    [cli["pooled_pixvqa"]["mean"]], meta["clip_hi"], f"{model}/{kind}/mean"
                )
                capi = compare_c_api(model, meta, ref, dist, n)
                record(
                    {
                        "case": "unclipped",
                        "status": "PASSED",
                        "frames": n,
                        "kind": kind,
                        "model": model,
                        "model_sha256": model_sha256(model),
                        "clip_hi": meta["clip_hi"],
                        "ref_sha256": rh,
                        "dist_sha256": dh,
                        "scores_pixvqa": capi["scores_pixvqa"],
                        "scores_oracle": capi["scores_oracle"],
                        "pooled_pixvqa": capi["pooled_pixvqa"],
                        "pooled_oracle": capi["pooled_oracle"],
                        "pool_abs": capi["max_abs_pool"],
                        "cli_max_abs_score": cli["max_abs_score"],
                        "cli_max_abs_feature": cli["max_abs_feature"],
                        "c_api_max_abs_feature": capi["max_abs_feature"],
                    },
                    capi,
                )
                print(
                    f"  OK unclipped mean={capi['pooled_pixvqa']['mean']:.6f} "
                    f"poolΔmax={max(capi['max_abs_pool'].values()):.3e}"
                )

        # --- Near-zero / zero-compatible harmonic path (severe noise) ---
        # Uses extreme degradation; exact zeros are not fabricated if scores stay > 0.
        n = 5
        ref, dist = td_path / "z_ref.yuv", td_path / "z_dist.yuv"
        # Overwrite write_pair with maximal luma destruction for near-zero scores.
        rh = hashlib.sha256()
        dh = hashlib.sha256()
        with ref.open("wb") as fr, dist.open("wb") as fd:
            for fi in range(n):
                y_ref = bytearray(W * H * 2)
                y_dist = bytearray(W * H * 2)
                for i in range(0, W * H * 2, 2):
                    # Structured reference; distorted plane forced near black.
                    v = (300 + (i // 2 + fi * 17) % 400) & 0x3FF
                    y_ref[i] = v & 0xFF
                    y_ref[i + 1] = (v >> 8) & 0xFF
                    y_dist[i] = 0
                    y_dist[i + 1] = 0
                fr.write(y_ref)
                fd.write(y_dist)
                rh.update(y_ref)
                dh.update(y_dist)
                c = bytes(W * H // 2)  # U+V zeros for both
                # Proper 10-bit mid chroma for ref, zero for dist.
                cu = bytearray((W // 2) * (H // 2) * 2)
                for i in range(0, len(cu), 2):
                    cu[i], cu[i + 1] = 0x00, 0x02  # 512
                cz = bytes(len(cu))
                for _ in range(2):
                    fr.write(cu)
                    fd.write(cz)
                    rh.update(cu)
                    dh.update(cz)
        for model in ("vmaf_v1.0.16_3d0h.json", "vmaf_v1.0.16_hfr_3d0h.json"):
            print(f"near_zero n={n} model={model} ...", flush=True)
            capi = compare_c_api(model, MODELS[model], ref, dist, n)
            scores = capi["scores_oracle"]
            exact_zero = all(abs(s) < 1e-12 for s in scores)
            near_zero = all(s < 5.0 for s in scores)
            if not near_zero:
                raise SystemExit(
                    f"{model}: expected near-zero scores, got {scores} "
                    "(cannot validate low-score harmonic path)"
                )
            record(
                {
                    "case": "near_zero",
                    "status": "PASSED",
                    "frames": n,
                    "model": model,
                    "model_sha256": model_sha256(model),
                    "ref_sha256": rh.hexdigest(),
                    "dist_sha256": dh.hexdigest(),
                    "exact_zero_scores": exact_zero,
                    "scores_oracle": scores,
                    "pooled_pixvqa": capi["pooled_pixvqa"],
                    "pooled_oracle": capi["pooled_oracle"],
                    "pool_abs": capi["max_abs_pool"],
                    "note": (
                        "Exact zeros observed"
                        if exact_zero
                        else "Near-zero scores (exact zero not required)"
                    ),
                },
                capi,
            )
            print(
                f"  OK near_zero exact_zero={exact_zero} "
                f"harmonic={capi['pooled_oracle']['harmonic_mean']:.6f}"
            )

    out_path = ROOT / "reports" / "vmaf_numeric_suite_results.json"
    out_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print("pool_abs_max_global:", evidence["pool_abs_max_global"])
    print("OK VMAF numeric suite (temporal + unclipped + pools + near-zero)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
