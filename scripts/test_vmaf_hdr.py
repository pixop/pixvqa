#!/usr/bin/env python3
"""Public HDR VMAF v1.1.8 parity against the official CLI from the same build.

Uses the public CLI (--metrics, --model, --hdr-*). Does not use --vmaf-backend.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIX = Path(os.environ.get("PIXVQA", ROOT / "build" / "cpu" / "pixvqa"))
CLI = Path(os.environ.get("VMAF_CLI", ROOT / "deps" / "libvmaf-74f930d" / "bin" / "vmaf"))


def resolve_ffmpeg() -> str:
    explicit = os.environ.get("FFMPEG", "").strip()
    if explicit:
        return explicit
    return shutil.which("ffmpeg") or ""


FFMPEG = resolve_ffmpeg()
WARN = "pixvqa: warning: VMAF CUDA computation is unavailable; using CPU libvmaf."
TOL = 1e-4
HDR = [
    "--hdr-transfer", "pq",
    "--hdr-primaries", "bt2020",
    "--hdr-matrix", "bt2020nc",
    "--hdr-range", "limited",
]
MODELS = {
    "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json": ("cambi_ceot_pq_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_psm_bilinear_snn_0.19_wvm_5", "integer_adm3_csf_3_dlmw_0.7_egl_1_min_0.5_nw_0.02_ssz", "integer_motion3_mmxv_18", 100),
    "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json": ("cambi_ceot_pq_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5", "integer_adm3_csf_3_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5_ssz", "integer_motion3_mmxv_18", 100),
    "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json": ("cambi_ceot_pq_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_psm_bilinear_snn_0.19_wvm_5", "integer_adm3_csf_3_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160_ssz", "integer_motion3_mmxv_18", 100),
    "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json": ("cambi_ceot_pq_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5", "integer_adm3_csf_3_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160_ssz", "integer_motion3_mmxv_18", 110),
    "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json": ("cambi_ceot_pq_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_psm_bilinear_snn_0.19_wvm_5", "integer_adm3_csf_3_dlmw_0.7_egl_1_min_0.5_nw_0.02_ssz", "integer_motion3_mffw_mmxv_18_mma", 100),
    "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json": ("cambi_ceot_pq_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5", "integer_adm3_csf_3_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5_ssz", "integer_motion3_mffw_mmxv_18_mma", 100),
    "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json": ("cambi_ceot_pq_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_psm_bilinear_snn_0.19_wvm_5", "integer_adm3_csf_3_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160_ssz", "integer_motion3_mffw_mmxv_18_mma", 100),
    "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json": ("cambi_ceot_pq_hrs_1080_cmxv_17_vlt_0.06", "speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5", "integer_adm3_csf_3_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160_ssz", "integer_motion3_mffw_mmxv_18_mma", 110),
}


def run(cmd, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=merged)


def write_ramp(path: Path, w: int, h: int, n: int, kind: str) -> None:
    cw, ch = w // 2, h // 2
    with path.open("wb") as out:
        for i in range(n):
            y = bytearray()
            for row in range(h):
                for col in range(w):
                    sample = 64 + ((col + i * 17 + row) * 877 // max(w, 1)) % 877
                    if kind == "quantize":
                        sample &= ~15
                    elif kind == "blur":
                        sample = 64 + (((col + i * 17 + row) * 877 // max(w, 1)) % 877)
                        sample = (sample + 64 + (((col + 2 + i * 17 + row) * 877 // max(w, 1)) % 877)) // 2
                    elif kind == "resample":
                        sample = 64 + (((col // 2 * 2 + i * 17 + row // 2 * 2) * 877 // max(w, 1)) % 877)
                    y += struct.pack("<H", sample)
            chroma = struct.pack("<H", 512) * (cw * ch)
            out.write(y)
            out.write(chroma)
            out.write(chroma)


def pix_cmd(model: Path, ref: Path, dist: Path, w: int, h: int, extra=None):
    cmd = [str(PIX), "--metrics", "vmaf", "--model", str(model), "--threads", "8",
           *(extra if extra is not None else HDR),
           "-r", str(ref), "-s", f"{w}x{h}", "-f", "420p:10",
           "-t", str(dist), "-s", f"{w}x{h}", "-f", "420p:10"]
    return cmd


def cli_cmd(model: Path, ref: Path, dist: Path, w: int, h: int, out: Path):
    side = []
    for s in ("ref", "dist"):
        side += [f"--color_range_{s}", "limited", f"--color_primaries_{s}", "bt2020",
                 f"--color_trc_{s}", "pq", f"--color_matrix_{s}", "bt2020nc"]
    return [str(CLI), "-r", str(ref), "-d", str(dist), "-w", str(w), "-h", str(h), "-p", "420", "-b", "10",
            *side, "--model", f"path={model}", "--json", "-o", str(out), "--threads", "8", "--quiet"]


def compare(model_rel: str, ref: Path, dist: Path, w: int, h: int, label: str) -> dict:
    model = ROOT / "model" / model_rel
    keys = MODELS[model_rel]
    with tempfile.TemporaryDirectory() as td:
        js = Path(td) / "cli.json"
        cli = run(cli_cmd(model, ref, dist, w, h, js))
        if cli.returncode != 0:
            raise SystemExit(f"CLI failed {label}\n{cli.stderr}")
        data = json.loads(js.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PIXVQA_VMAF_FEATURES"] = "1"
    pix = run(pix_cmd(model, ref, dist, w, h), env)
    if pix.returncode != 0:
        raise SystemExit(f"pixvqa failed {label}\n{pix.stderr}")
    text = pix.stdout + "\n" + pix.stderr
    scores = [float(x) for x in re.findall(r"\[ VMAF:([0-9.eE+-]+) \]", text)]
    summary = re.search(r"\[ VMAF mean:([0-9.eE+-]+) harmonic:([0-9.eE+-]+) min:([0-9.eE+-]+) max:([0-9.eE+-]+) frames:(\d+) \]", text)
    if not summary:
        raise SystemExit(f"no summary {label}\n{text[-500:]}")
    frames = data["frames"]
    pooled = data["pooled_metrics"]["vmaf"]
    cli_scores = [float(fr["metrics"]["vmaf"]) for fr in frames]
    if len(scores) != len(cli_scores):
        raise SystemExit(f"frame count {label}: pix {len(scores)} cli {len(cli_scores)}")
    score_abs = max(abs(a - b) for a, b in zip(scores, cli_scores))
    pools = {
        "mean": abs(float(summary.group(1)) - float(pooled["mean"])),
        "harmonic": abs(float(summary.group(2)) - float(pooled["harmonic_mean"])),
        "min": abs(float(summary.group(3)) - float(pooled["min"])),
        "max": abs(float(summary.group(4)) - float(pooled["max"])),
    }
    feat_names = ("cambi", "speed_chroma", "adm3", "motion3")
    feat_abs = {}
    for logical, name in zip(feat_names, keys[:4]):
        cs = [float(fr["metrics"][name]) for fr in frames]
        ps = [float(m) for m in re.findall(rf"{logical}=([0-9.eE+-]+)", text)]
        if len(ps) != len(cs):
            raise SystemExit(f"missing {logical} {label}: {len(ps)} vs {len(cs)}")
        feat_abs[logical] = max(abs(a - b) for a, b in zip(ps, cs))
    clip = keys[4]
    clip_ok = f"clip:[0,{clip}]" in text
    within = all(0.0 <= s <= clip for s in cli_scores)
    color_ok = "color_ref=limited:bt2020:pq:bt2020nc" in text and "color_dist=limited:bt2020:pq:bt2020nc" in text
    commit_ok = "commit=74f930d17de8c58e7c2ce55e2b897b0030ebd8ef" in text and "zimg=3.0.6" in text
    worst = max(score_abs, *pools.values(), *feat_abs.values())
    passed = worst <= TOL and clip_ok and within and color_ok and commit_ok and int(summary.group(5)) == len(frames)
    print(f"{'OK  ' if passed else 'FAIL'} {label} mean={summary.group(1)} worst={worst:.3e} clip={clip}")
    return {"label": label, "model": model_rel, "mean": float(summary.group(1)), "worst": worst,
            "score_abs": score_abs, "pools": pools, "features": feat_abs, "pass": passed,
            "clip": clip, "frames": len(frames)}


def expect_gpu_matches_cpu(ref: Path, dist: Path) -> bool:
    model = ROOT / "model" / "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json"
    base = pix_cmd(model, ref, dist, 1920, 1080) + ["-v", "0"]
    cpu = run(base + ["--metric-device", "cpu"])
    gpu = run(base + ["--metric-device", "gpu"])
    pattern = r"\[ VMAF mean:[0-9.eE+-]+ harmonic:[0-9.eE+-]+ min:[0-9.eE+-]+ max:[0-9.eE+-]+ frames:\d+ \]"
    cpu_sum = re.search(pattern, cpu.stdout)
    gpu_sum = re.search(pattern, gpu.stdout)
    ok = (
        cpu.returncode == 0
        and gpu.returncode == 0
        and cpu_sum is not None
        and gpu_sum is not None
        and cpu_sum.group(0) == gpu_sum.group(0)
        and WARN in gpu.stderr
        and WARN not in cpu.stderr
        and "backend=cpu" in gpu.stdout
        and "backend=cuda" not in gpu.stdout
        and "VMAF: cpu libvmaf" in gpu.stdout
        and "color_ref=limited:bt2020:pq:bt2020nc" in gpu.stdout
        and "CUDA HDR" not in gpu.stderr
    )
    print(f"{'OK  ' if ok else 'FAIL'} gpu-vmaf-matches-cpu rc={gpu.returncode}")
    if not ok:
        print((gpu.stderr + gpu.stdout)[-500:])
    return ok


def expect_fail(label, args, needle):
    pix = run([str(PIX), *args])
    text = pix.stderr + pix.stdout
    ok = pix.returncode != 0 and needle in text
    print(f"{'OK  ' if ok else 'FAIL'} {label} rc={pix.returncode}")
    if not ok:
        print(text[-400:])
    return ok


def main() -> int:
    if not PIX.is_file() or not CLI.is_file():
        raise SystemExit(f"missing binary pix={PIX} cli={CLI}")
    results = []
    fails = []
    with tempfile.TemporaryDirectory(prefix="pixvqa_hdr_") as td:
        td = Path(td)
        pairs = {}
        for w, h, tag in ((1920, 1080, "1080"), (3840, 2160, "2160")):
            ref = td / f"ref_{tag}.yuv"
            write_ramp(ref, w, h, 8, "identical")
            variants = {"identical": ref}
            for kind in ("blur", "resample"):
                path = td / f"{kind}_{tag}.yuv"
                write_ramp(path, w, h, 8, kind)
                variants[kind] = path
            pairs[tag] = (w, h, ref, variants)
        plan = {
            "1080": [
                "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json",
                "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_5d0h.json",
                "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json",
                "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_5d0h.json",
            ],
            "2160": [
                "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_1d5h_2160.json",
                "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json",
                "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_1d5h_2160.json",
                "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json",
            ],
        }
        for tag, models in plan.items():
            w, h, ref, variants = pairs[tag]
            for model in models:
                for kind in ("identical", "blur", "resample"):
                    row = compare(model, ref, variants[kind], w, h, f"{tag} {kind} {Path(model).stem}")
                    results.append(row)
                    if not row["pass"]:
                        fails.append(row["label"])
        base = ["-r", str(pairs["1080"][2]), "-s", "1920x1080", "-f", "420p:10",
                "-t", str(pairs["1080"][2]), "-s", "1920x1080", "-f", "420p:10"]
        hdr_model = str(ROOT / "model" / "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json")
        sdr_model = str(ROOT / "model" / "vmaf_v1.0.16_3d0h.json")
        checks = [
            expect_fail("missing-color", ["--metrics", "vmaf", "--model", hdr_model, *base], "explicit PQ transfer"),
            expect_fail("partial-color", ["--metrics", "vmaf", "--model", hdr_model, "--hdr-transfer", "pq", *base], "explicit complete colorimetry"),
            expect_fail("hlg", ["--metrics", "vmaf", "--model", hdr_model, "--hdr-transfer", "hlg", "--hdr-primaries", "bt2020", "--hdr-matrix", "bt2020nc", "--hdr-range", "limited", *base], "unsupported and unvalidated"),
            expect_fail("full-range", ["--metrics", "vmaf", "--model", hdr_model, "--hdr-transfer", "pq", "--hdr-primaries", "bt2020", "--hdr-matrix", "bt2020nc", "--hdr-range", "full", *base], "full-range PQ"),
            expect_fail("sdr-model-pq", ["--metrics", "vmaf", "--model", sdr_model, "--hdr-transfer", "pq", *base], "rejects explicit HDR"),
            expect_gpu_matches_cpu(pairs["1080"][2], pairs["1080"][2]),
        ]
        if not all(checks):
            fails.append("error-cases")
        real = real_hevc(td)
        results.extend(real["cases"])
        fails.extend(real["fails"])
        perf = real["perf"]
    worst = max((r["worst"] for r in results), default=0.0)
    report = {"pass": not fails, "cases": len(results), "worst_abs": worst, "fails": fails,
              "results": results, "perf": perf, "pixvqa": str(PIX), "cli": str(CLI)}
    out = ROOT / "reports" / "hdr_vmaf_public_parity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} cases={len(results)} worst={worst:.3e} pass={not fails}")
    return 0 if not fails else 1


def real_hevc(td: Path) -> dict:
    out = {"cases": [], "fails": [], "perf": []}
    master = Path("/mnt/f/videos/2160p/BasketballGame_3840x2160_5994_420_10b_pq_bt2020l_29m25s_3600fr.yuv")
    if not master.is_file() or not Path(FFMPEG).is_file():
        print("SKIP real HEVC (master or ffmpeg unavailable)")
        out["skipped"] = True
        return out
    frame = 3840 * 2160 * 3 // 2 * 2
    ref = td / "real4k.yuv"
    with master.open("rb") as src, ref.open("wb") as dst:
        src.seek(1800 * frame)
        dst.write(src.read(60 * frame))
    enc = run([FFMPEG, "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
               "-s", "3840x2160", "-r", "60000/1001", "-i", str(ref), "-frames:v", "60",
               "-c:v", "libx265", "-preset", "fast", "-crf", "28", "-pix_fmt", "yuv420p10le",
               "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
               "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
               "-color_range", "tv", str(td / "real4k.mp4")])
    if enc.returncode != 0:
        raise SystemExit(f"encode failed\n{enc.stderr}")
    dec = run([FFMPEG, "-hide_banner", "-v", "error", "-y", "-i", str(td / "real4k.mp4"),
               "-f", "rawvideo", "-pix_fmt", "yuv420p10le", str(td / "dist4k.yuv")])
    if dec.returncode != 0 or (td / "dist4k.yuv").stat().st_size != 60 * frame:
        raise SystemExit("decode size mismatch")
    for model in (
        "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json",
        "hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h_2160.json",
    ):
        row = compare(model, ref, td / "dist4k.yuv", 3840, 2160, f"real 2160 crf28 {Path(model).stem}")
        out["cases"].append(row)
        if not row["pass"]:
            out["fails"].append(row["label"])
    down_ref = td / "real1080.yuv"
    down_dist = td / "dist1080.yuv"
    for src, dst in ((ref, down_ref), (td / "dist4k.yuv", down_dist)):
        got = run([FFMPEG, "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
                   "-s", "3840x2160", "-i", str(src), "-vf", "scale=1920:1080:flags=lanczos",
                   "-pix_fmt", "yuv420p10le", "-f", "rawvideo", str(dst)])
        if got.returncode != 0:
            raise SystemExit(got.stderr)
    row = compare("hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json", down_ref, down_dist, 1920, 1080,
                  "real 1080 derived crf28 hdr_vmaf_v1.1.8_hfr_3d0h")
    out["cases"].append(row)
    if not row["pass"]:
        out["fails"].append(row["label"])
    for label, model, w, h, r, d, n in (
        ("4k", "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h_2160.json", 3840, 2160, ref, td / "dist4k.yuv", 60),
        ("1080", "hdr_vmaf_v1.1.8_hfr/hdr_vmaf_v1.1.8_hfr_3d0h.json", 1920, 1080, down_ref, down_dist, 60),
    ):
        model_path = ROOT / "model" / model
        walls = []
        ios = []
        computes = []
        for _ in range(2):
            env = os.environ.copy()
            env["PIXVQA_PROFILE"] = "1"
            t0 = time.perf_counter()
            pix = run(pix_cmd(model_path, r, d, w, h) + ["-v", "0"], env)
            walls.append(time.perf_counter() - t0)
            if pix.returncode != 0:
                raise SystemExit(pix.stderr[-300:])
            prof = re.search(r"io=([0-9.]+)ms compute=([0-9.]+)ms", pix.stderr)
            if not prof:
                raise SystemExit(f"no profile {label}\n{pix.stderr[-300:]}")
            ios.append(float(prof.group(1)) / 1000.0)
            computes.append(float(prof.group(2)) / 1000.0)
        out["perf"].append({
            "label": label, "model": model, "frames": n, "threads": 8,
            "wall_s": walls, "io_s": ios, "compute_s": computes,
            "median_fps": n / sorted(walls)[0],
            "median_compute_fps": n / sorted(computes)[0],
        })
        print(f"PERF {label} wall_fps={n / sorted(walls)[0]:.2f} compute_fps={n / sorted(computes)[0]:.2f} io={sorted(ios)[0]:.2f}s")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
