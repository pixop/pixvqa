#!/usr/bin/env python3
"""Level 1–3 VMAF parity vs official CLI for all eight vmaf_v1.0.16 models."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXVQA = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
VMAF_CLI = Path(os.environ.get("VMAF_CLI", ROOT / "deps" / "libvmaf-74f930d" / "bin" / "vmaf"))

MODEL_DIR = ROOT / "model"
MODELS = {
    "vmaf_v1.0.16_3d0h.json": {
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02",
        "motion3": "integer_motion3_mmxv_18",
    },
    "vmaf_v1.0.16_5d0h.json": {
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5",
        "motion3": "integer_motion3_mmxv_18",
    },
    "vmaf_v1.0.16_1d5h_2160.json": {
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160",
        "motion3": "integer_motion3_mmxv_18",
    },
    "vmaf_v1.0.16_3d0h_2160.json": {
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160",
        "motion3": "integer_motion3_mmxv_18",
    },
    "vmaf_v1.0.16_hfr_3d0h.json": {
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02",
        "motion3": "integer_motion3_mffw_mmxv_18_mma",
    },
    "vmaf_v1.0.16_hfr_5d0h.json": {
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5",
        "motion3": "integer_motion3_mffw_mmxv_18_mma",
    },
    "vmaf_v1.0.16_hfr_1d5h_2160.json": {
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160",
        "motion3": "integer_motion3_mffw_mmxv_18_mma",
    },
    "vmaf_v1.0.16_hfr_3d0h_2160.json": {
        "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
        "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5",
        "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160",
        "motion3": "integer_motion3_mffw_mmxv_18_mma",
    },
}

W, H, N = 1920, 1080, 3
TOL_SCORE = 1e-6
TOL_FEAT = 5e-7  # see test_vmaf_smoke.py — CLI prints 6 decimals


def write_pair(ref: Path, dist: Path) -> tuple[str, str]:
    """Reference plus mild luma noise for non-trivial VMAF scores."""
    rh = hashlib.sha256()
    dh = hashlib.sha256()
    with ref.open("wb") as fr, dist.open("wb") as fd:
        for i in range(N):
            for row in range(H):
                for col in range(W):
                    v = (400 + i * 13 + (row * 7 + col * 3) % 200) & 0x3FF
                    fr.write(v.to_bytes(2, "little"))
                    rh.update(v.to_bytes(2, "little"))
                    # Spatially varying distortion.
                    d = min(1023, max(0, v + ((row + col + i) % 7) - 3))
                    fd.write(d.to_bytes(2, "little"))
                    dh.update(d.to_bytes(2, "little"))
            for plane_seed in (512, 520):
                for row in range(H // 2):
                    for col in range(W // 2):
                        u = (plane_seed + (row + col + i) % 16) & 0x3FF
                        fr.write(u.to_bytes(2, "little"))
                        rh.update(u.to_bytes(2, "little"))
                        du = min(1023, max(0, u + ((row + i) % 3) - 1))
                        fd.write(du.to_bytes(2, "little"))
                        dh.update(du.to_bytes(2, "little"))
    return rh.hexdigest(), dh.hexdigest()


def run(cmd: list[str], env: dict[str, str] | None = None) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write("FAIL CMD " + " ".join(cmd) + "\n" + p.stderr + p.stdout)
        raise SystemExit(p.returncode)
    return p.stdout + "\n" + p.stderr


def compare_model(model_name: str, feat_keys: dict[str, str], ref: Path, dist: Path, td: Path) -> dict:
    model_path = MODEL_DIR / model_name
    if not model_path.is_file():
        raise SystemExit(f"missing model JSON: {model_path}")
    oracle = td / f"{model_name}.oracle.json"
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
            f"path={model_path}",
            "--json",
            "-o",
            str(oracle),
            "--threads",
            "8",
            "--quiet"]
    )
    data = json.loads(oracle.read_text(encoding="utf-8"))
    o_scores = [float(fr["metrics"]["vmaf"]) for fr in data["frames"]]
    o_feats = {
        n: [float(fr["metrics"][k]) for fr in data["frames"]] for n, k in feat_keys.items()
    }
    o_pooled = float(data["pooled_metrics"]["vmaf"]["mean"])

    env = os.environ.copy()
    env["PIXVQA_VMAF_FEATURES"] = "1"
    out = run(
        [
            str(PIXVQA),
            "--metrics", "vmaf",
            "--model",
            str(model_path),
            
            "--threads",
            "8",
            "-v",
            "1",
            "-n",
            str(N),
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
    p_scores = [float(x) for x in re.findall(r"\[ VMAF:([0-9.]+) \]", out)]
    m = re.search(
        r"\[ VMAF mean:([0-9.]+) harmonic:([0-9.]+) min:([0-9.]+) max:([0-9.]+) frames:(\d+) \]",
        out,
    )
    if not m or len(p_scores) != N:
        raise SystemExit(f"{model_name}: bad pixvqa output\n{out}")
    p_pooled = float(m.group(1))
    p_feats: dict[str, list[float]] = {n: [] for n in feat_keys}
    for line in out.splitlines():
        if not line.startswith("vmaf_features "):
            continue
        kv = dict(re.findall(r"(\w+)=([-+0-9.eE]+)", line))
        for n in feat_keys:
            p_feats[n].append(float(kv[n if n != "speed_chroma" else "speed_chroma"]))

    max_score = max(abs(a - b) for a, b in zip(p_scores, o_scores))
    max_score = max(max_score, abs(p_pooled - o_pooled))
    max_feat = 0.0
    for n in feat_keys:
        for a, b in zip(p_feats[n], o_feats[n]):
            max_feat = max(max_feat, abs(a - b))
    if max_score > TOL_SCORE or max_feat > TOL_FEAT:
        raise SystemExit(
            f"{model_name}: FAIL max_score={max_score:.3e} max_feat={max_feat:.3e} "
            f"scores={p_scores} oracle={o_scores}"
        )
    return {
        "model": model_name,
        "model_path": str(model_path),
        "max_abs_score": max_score,
        "max_abs_feature": max_feat,
        "pooled_pixvqa": p_pooled,
        "pooled_oracle": o_pooled,
        "per_frame_pixvqa": p_scores,
    }


def main() -> int:
    if not PIXVQA.is_file() or not VMAF_CLI.is_file():
        print("missing binary", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="vmaf_num_") as td:
        td_path = Path(td)
        ref, dist = td_path / "ref.yuv", td_path / "dist.yuv"
        ref_h, dist_h = write_pair(ref, dist)
        print(f"ref_sha256={ref_h}")
        print(f"dist_sha256={dist_h}")
        print(f"size={W}x{H} frames={N} bitdepth=10")
        results = []
        for model, keys in MODELS.items():
            print(f"model {model} ...", flush=True)
            results.append(compare_model(model, keys, ref, dist, td_path))
            r = results[-1]
            print(
                f"  OK max_score={r['max_abs_score']:.3e} max_feat={r['max_abs_feature']:.3e} "
                f"pooled={r['pooled_pixvqa']:.6f}"
            )
        out = ROOT / "reports" / "vmaf_numeric_results.json"
        out.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        print("OK all eight VMAF v1 models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
