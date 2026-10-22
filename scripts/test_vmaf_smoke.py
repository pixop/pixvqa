#!/usr/bin/env python3
"""Low-resolution VMAF smoke: PixVQA vs official libvmaf CLI on identical YUV."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXVQA = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
VMAF_CLI = Path(
    os.environ.get("VMAF_CLI", ROOT / "deps" / "libvmaf-3.2.1" / "bin" / "vmaf")
)
MODEL_PATH = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"
# 320x240 is large enough for the 3d0h SpEED path; 176x144 is not.
W, H, N = 320, 240, 3
TOL_SCORE = 1e-6
# Official vmaf CLI JSON/XML prints features with 6 decimal digits. Against that
# oracle the observed max abs error on motion3 is ~3.8e-7 (half-ulps of 1e-6).
# The provisional Architect target of 1e-8 is tighter than the printed oracle.
TOL_FEAT = 5e-7

# Exported collector keys for vmaf_v1.0.16_3d0h in libvmaf 3.2.1.
FEAT = {
    "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
    "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
    "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02",
    "motion3": "integer_motion3_mmxv_18",
}


def write_yuv420p10(path: Path, frames: int, seed: int) -> str:
    yw, yh = W, H
    cw, ch = W // 2, H // 2
    h = hashlib.sha256()
    with path.open("wb") as f:
        for i in range(frames):
            y = bytearray()
            for row in range(yh):
                for col in range(yw):
                    v = (seed + i * 17 + row * 3 + col * 5) & 0x3FF
                    y += v.to_bytes(2, "little")
            u = bytearray()
            vpl = bytearray()
            for row in range(ch):
                for col in range(cw):
                    uv = (512 + ((seed + i + row + col) & 31)) & 0x3FF
                    u += uv.to_bytes(2, "little")
                    vpl += ((512 - ((seed + i + row) & 31)) & 0x3FF).to_bytes(2, "little")
            blob = bytes(y) + bytes(u) + bytes(vpl)
            f.write(blob)
            h.update(blob)
    return h.hexdigest()


def run(cmd: list[str], env: dict[str, str] | None = None) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write("CMD FAIL: " + " ".join(cmd) + "\n")
        sys.stderr.write(p.stderr)
        sys.stderr.write(p.stdout)
        raise SystemExit(p.returncode)
    return p.stdout + "\n" + p.stderr


def parse_pixvqa(out: str) -> tuple[list[float], float, dict[str, list[float]]]:
    per = [float(x) for x in re.findall(r"\[ VMAF:([0-9.]+) \]", out)]
    m = re.search(
        r"\[ VMAF mean:([0-9.]+) harmonic:([0-9.]+) min:([0-9.]+) max:([0-9.]+) frames:(\d+) \]",
        out,
    )
    if not m:
        raise SystemExit("missing pooled VMAF in pixvqa output:\n" + out)
    pooled = float(m.group(1))
    if len(per) != N:
        raise SystemExit(f"expected {N} per-frame scores, got {len(per)}:\n{out}")
    feats: dict[str, list[float]] = {k: [] for k in FEAT}
    for line in out.splitlines():
        if not line.startswith("vmaf_features "):
            continue
        # vmaf_features i score=.. cambi=.. speed_chroma=.. adm3=.. motion3=..
        kv = dict(re.findall(r"(\w+)=([-+0-9.eE]+)", line))
        feats["cambi"].append(float(kv["cambi"]))
        feats["speed_chroma"].append(float(kv["speed_chroma"]))
        feats["adm3"].append(float(kv["adm3"]))
        feats["motion3"].append(float(kv["motion3"]))
    if any(len(v) != N for v in feats.values()):
        raise SystemExit(f"missing PIXVQA_VMAF_FEATURES dump:\n{out}")
    return per, pooled, feats


def parse_vmaf_json(path: Path) -> tuple[list[float], dict[str, list[float]], float]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    scores = [float(fr["metrics"]["vmaf"]) for fr in data["frames"]]
    feats = {
        name: [float(fr["metrics"][key]) for fr in data["frames"]] for name, key in FEAT.items()
    }
    pooled = float(data["pooled_metrics"]["vmaf"]["mean"])
    if len(scores) != N:
        raise SystemExit(f"oracle frames {len(scores)} != {N}")
    return scores, feats, pooled


def main() -> int:
    if not PIXVQA.is_file():
        print(f"missing pixvqa: {PIXVQA}", file=sys.stderr)
        return 2
    if not VMAF_CLI.is_file():
        print(f"missing vmaf cli: {VMAF_CLI}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="pixvqa_vmaf_") as td:
        td_path = Path(td)
        ref = td_path / "ref.yuv"
        dist = td_path / "dist.yuv"
        ref_hash = write_yuv420p10(ref, N, seed=11)
        dist_hash = write_yuv420p10(dist, N, seed=29)
        xml = td_path / "oracle.xml"

        oracle_cmd = [
            str(VMAF_CLI),
            "--reference",
            str(ref),
            "--distorted",
            str(dist),
            "--width",
            str(W),
            "--height",
            str(H),
            "--pixel_format",
            "420",
            "--bitdepth",
            "10",
            "--model",
            f"path={MODEL_PATH}",
            "--json",
            "--output",
            str(xml.with_suffix(".json")),
            "--threads",
            "4"]
        run(oracle_cmd)
        o_scores, o_feats, o_pooled = parse_vmaf_json(xml.with_suffix(".json"))

        env = os.environ.copy()
        env["PIXVQA_VMAF_FEATURES"] = "1"
        pix_cmd = [
            str(PIXVQA),
            "--metrics", "vmaf",
            "--model",
            str(MODEL_PATH),
            
            "--threads",
            "4",
            "-v",
            "1",
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
            "420p:10"]
        out = run(pix_cmd, env=env)
        p_scores, p_pooled, p_feats = parse_pixvqa(out)

        print(f"ref_sha256={ref_hash}")
        print(f"dist_sha256={dist_hash}")
        print(f"model={MODEL_PATH}")
        print(f"frames={N} size={W}x{H} bitdepth=10")

        max_abs = 0.0
        for i, (a, b) in enumerate(zip(p_scores, o_scores)):
            d = abs(a - b)
            max_abs = max(max_abs, d)
            print(f"frame {i}: pixvqa={a:.12f} oracle={b:.12f} abs={d:.3e}")
            if d > TOL_SCORE:
                print(f"FAIL score frame {i}", file=sys.stderr)
                return 1
        dp = abs(p_pooled - o_pooled)
        print(f"pooled: pixvqa={p_pooled:.12f} oracle={o_pooled:.12f} abs={dp:.3e}")
        if dp > TOL_SCORE:
            print("FAIL pooled", file=sys.stderr)
            return 1

        max_feat = 0.0
        for name in FEAT:
            for i, (a, b) in enumerate(zip(p_feats[name], o_feats[name])):
                d = abs(a - b)
                max_feat = max(max_feat, d)
                if d > TOL_FEAT:
                    print(
                        f"FAIL feature {name} frame {i}: {a} vs {b} abs={d}",
                        file=sys.stderr,
                    )
                    return 1
            print(f"feature {name}: ok (n={N})")

        print(f"OK vmaf smoke max_abs_score={max_abs:.3e} max_abs_feat={max_feat:.3e}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
