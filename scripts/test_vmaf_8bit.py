#!/usr/bin/env python3
"""8-bit YUV420 numerical comparison vs official CLI + C-API oracle."""
from __future__ import annotations

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
VMAF_CLI = Path(os.environ.get("VMAF_CLI", ROOT / "deps" / "libvmaf-3.2.1" / "bin" / "vmaf"))
ORACLE = Path(os.environ.get("VMAF_REF_ORACLE", ROOT / "build" / "cpu" / "vmaf_ref_oracle"))
MODEL = ROOT / "model" / "vmaf_v1.0.16_3d0h.json"
FEAT = {
    "cambi": "cambi_hrs_1080_cmxv_17_vlt_0.06",
    "speed_chroma": "speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5",
    "adm3": "integer_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02",
    "motion3": "integer_motion3_mmxv_18",
}
W, H, N = 320, 240, 5
TOL_SCORE = 1e-6
TOL_FEAT = 5e-7


def write_pair(ref: Path, dist: Path) -> None:
    with ref.open("wb") as fr, dist.open("wb") as fd:
        for fi in range(N):
            for row in range(H):
                for col in range(W):
                    v = (40 + fi * 3 + (row + col) % 80) & 0xFF
                    fr.write(struct.pack("B", v))
                    d = max(0, min(255, v + ((row * 3 + col + fi) % 41) - 20))
                    fd.write(struct.pack("B", d))
            for seed in (128, 140):
                for row in range(H // 2):
                    for col in range(W // 2):
                        u = (seed + (row + col + fi) % 16) & 0xFF
                        fr.write(struct.pack("B", u))
                        fd.write(struct.pack("B", max(0, min(255, u + (fi % 7) - 3))))


def main() -> int:
    if not PIXVQA.is_file() or not VMAF_CLI.is_file() or not ORACLE.is_file():
        print("missing binary", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="vmaf8_") as td:
        td_path = Path(td)
        ref, dist = td_path / "r.yuv", td_path / "d.yuv"
        write_pair(ref, dist)
        oracle = td_path / "cli.json"
        subprocess.run(
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
                "8",
                "--model",
                f"path={MODEL}",
                "--json",
                "-o",
                str(oracle),
                "--threads",
                "4",
                "--quiet"],
            check=True,
        )
        data = json.loads(oracle.read_text(encoding="utf-8"))
        o_scores = [float(fr["metrics"]["vmaf"]) for fr in data["frames"]]
        env = os.environ.copy()
        env["PIXVQA_VMAF_FEATURES"] = "1"
        p = subprocess.run(
            [
                str(PIXVQA),
                "--metrics", "vmaf",
                "--model",
                str(MODEL),
                
                "--threads",
                "4",
                "-v",
                "1",
                "-n",
                str(N),
                "-r",
                str(ref),
                "-s",
                f"{W}x{H}",
                "-f",
                "420p:8",
                "-t",
                str(dist),
                "-s",
                f"{W}x{H}",
                "-f",
                "420p:8"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        out = p.stdout + "\n" + p.stderr
        per = []
        for line in out.splitlines():
            if line.startswith("vmaf_features "):
                kv = dict(re.findall(r"(\w+)=([-+0-9.eE]+)", line))
                per.append(float(kv["score"]))
        max_score = max(abs(a - b) for a, b in zip(per, o_scores))
        if max_score > TOL_SCORE:
            raise SystemExit(f"8-bit CLI FAIL {max_score}")
        # C-API double oracle
        raw = subprocess.run(
            [
                str(ORACLE),
                "--model",
                str(MODEL),
                "--ref",
                str(ref),
                "--dist",
                str(dist),
                "--width",
                str(W),
                "--height",
                str(H),
                "--bitdepth",
                "8",
                "--frames",
                str(N),
                "--feat-cambi",
                FEAT["cambi"],
                "--feat-speed",
                FEAT["speed_chroma"],
                "--feat-adm3",
                FEAT["adm3"],
                "--feat-motion3",
                FEAT["motion3"]],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        start, end = raw.find("{"), raw.rfind("}")
        cdata = json.loads(raw[start : end + 1])
        c_scores = [float(fr["vmaf"]) for fr in cdata["frames"]]
        max_c = max(abs(a - b) for a, b in zip(per, c_scores))
        if max_c > TOL_SCORE:
            raise SystemExit(f"8-bit C-API FAIL {max_c}")
        print(f"OK 8-bit 420p scores={per}")
        print(f"OK max_abs_cli={max_score:.3e} max_abs_c_api={max_c:.3e}")
        evidence = {
            "format": "420p:8",
            "size": f"{W}x{H}",
            "frames": N,
            "model": str(MODEL),
            "scores_pixvqa": per,
            "scores_cli": o_scores,
            "scores_c_api": c_scores,
            "max_abs_cli": max_score,
            "max_abs_c_api": max_c,
        }
        outp = ROOT / "reports" / "vmaf_8bit_results.json"
        outp.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
