#!/usr/bin/env python3
"""Clipping-range metadata must come from JSON, including unknown-checksum models."""
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
SRC = ROOT / "model" / "vmaf_v1.0.16_3d0h_2160.json"
OFFICIAL_SHA = "77f552fd8d5b6c6a2166bfc4470c9368ec68d2425ed40f1cc235b0cc500f33c2"
W, H = 1920, 1080


def write_yuv(path: Path) -> None:
    with path.open("wb") as f:
        f.write(bytes(W * H * 2))
        f.write(bytes((W // 2) * (H // 2) * 2))
        f.write(bytes((W // 2) * (H // 2) * 2))


def run_model(model: Path, ref: Path, dist: Path) -> str:
    cmd = [
        str(PIXVQA),
        "--metrics", "vmaf",
        "--model",
        str(model),
        
        "-v",
        "0",
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
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + "\n" + p.stderr
    if p.returncode != 0:
        raise SystemExit(f"pixvqa failed:\n{out}")
    return out


def main() -> int:
    if not PIXVQA.is_file():
        print("missing pixvqa", file=sys.stderr)
        return 2
    if not SRC.is_file():
        print(f"missing {SRC}", file=sys.stderr)
        return 2
    official = hashlib.sha256(SRC.read_bytes()).hexdigest()
    if official != OFFICIAL_SHA:
        raise SystemExit(f"unexpected official digest {official}")

    with tempfile.TemporaryDirectory(prefix="vmaf_clip_") as td:
        td_path = Path(td)
        ref, dist = td_path / "r.yuv", td_path / "d.yuv"
        write_yuv(ref)
        write_yuv(dist)

        # Harmless JSON reformat that changes SHA-256 without changing SVM weights
        # or model_dict.score_clip semantics. sort_keys may move param_dict after
        # model_dict; PixVQA must still pick the array form (not null).
        data = json.loads(SRC.read_text(encoding="utf-8"))
        modified = td_path / "custom_4k_3h_reformatted.json"
        modified.write_text(json.dumps(data, indent=4, sort_keys=True) + "\n", encoding="utf-8")
        mod_sha = hashlib.sha256(modified.read_bytes()).hexdigest()
        if mod_sha == official:
            raise SystemExit("reformatting failed to change SHA-256")

        out = run_model(modified, ref, dist)
        m = re.search(r"clip:\[([0-9.]+),([0-9.]+)\]", out)
        if not m:
            raise SystemExit("missing clip metadata:\n" + out)
        lo, hi = float(m.group(1)), float(m.group(2))
        if abs(lo - 0.0) > 1e-9 or abs(hi - 110.0) > 1e-9:
            raise SystemExit(f"expected clip [0,110] from JSON, got [{lo},{hi}]\n{out}")
        sha = re.search(r"sha256=([0-9a-f]{64})", out)
        if not sha or sha.group(1) != mod_sha:
            raise SystemExit(f"sha256 meta mismatch: {sha} vs file {mod_sha}\n{out}")
        if sha.group(1) == official:
            raise SystemExit("still reporting official SHA — unknown-checksum path not exercised")
        if "clip:[0,100]" in out:
            raise SystemExit("invented clip max 100 for unknown checksum")
        print("OK   reformatted 4K 3H (unknown SHA) reports clip:[0,110] from JSON")
        print(f"OK   official_sha={official}")
        print(f"OK   modified_sha={mod_sha}")

        # Optional: strip score_clip array → expect clip:unknown
        no_clip = td_path / "custom_no_score_clip.json"
        data2 = json.loads(SRC.read_text(encoding="utf-8"))
        if isinstance(data2.get("model_dict"), dict) and "score_clip" in data2["model_dict"]:
            data2["model_dict"]["score_clip"] = None
        if isinstance(data2.get("param_dict"), dict):
            data2["param_dict"]["score_clip"] = None
        no_clip.write_text(json.dumps(data2, indent=2) + "\n", encoding="utf-8")
        # Loading may still work; metadata should not invent 100.
        p = subprocess.run(
            [
                str(PIXVQA),
                "--metrics", "vmaf",
                "--model",
                str(no_clip),
                
                "-v",
                "0",
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
            capture_output=True,
            text=True,
        )
        out2 = p.stdout + "\n" + p.stderr
        if p.returncode != 0:
            print("NOTE null score_clip load rejected by libvmaf (acceptable)")
            print(out2.splitlines()[-3:] if out2 else "")
        else:
            if "clip:unknown" not in out2:
                # If JSON parser still finds a clip elsewhere, fail loudly.
                if "clip:[0,100]" in out2:
                    raise SystemExit("invented [0,100] when score_clip null")
                raise SystemExit(f"expected clip:unknown when score_clip null:\n{out2}")
            print("OK   null score_clip reports clip:unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
