#!/usr/bin/env python3
"""ChipQA / HDR-ChipQA reference parity and guard tests (uses pinned goldens)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

BASELINE = json.loads((ROOT / "reports/chipqa-reference/REFERENCE_BASELINE.json").read_text())
SDR_CLIP = ROOT / BASELINE["sdr_chipqa"]["clip"]
HDR_YUV = ROOT / BASELINE["hdr_chipqa"]["derived_yuv"]
CHIPQA_REPO = ROOT / "deps/ChipQA"
HDR_REPO = ROOT / "deps/hdr_chipqa"


def fail(msg: str) -> None:
    print(f"FAIL {msg}", file=sys.stderr)
    raise SystemExit(1)


def ok(name: str) -> None:
    print(f"OK   {name}")


def resolve_test_python(env_var: str) -> str | None:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_dir():
        candidate = path / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
        fail(f"{env_var}={raw!r} is not a venv with bin/python")
    if path.is_file():
        return str(path)
    fail(f"{env_var}={raw!r} is not a Python interpreter")


def require_chipqa_env(feature_var: str, inference_var: str) -> tuple[str, str] | None:
    feat = resolve_test_python(feature_var)
    inf = resolve_test_python(inference_var)
    if feat is None or inf is None:
        missing = [v for v, x in ((feature_var, feat), (inference_var, inf)) if x is None]
        print(f"SKIP parity ({', '.join(missing)} unset; see docs/dev/CHIPQA_REFERENCE.md)")
        return None
    return feat, inf


def test_sdr_contract() -> None:
    from pixvqa_chipqa.adapter import SetupError, inspect_video, run

    raw = ROOT / "reports/chipqa-reference/derived_pq_distorted_960x540_60f.yuv"
    try:
        inspect_video(raw)
    except SetupError:
        ok("sdr-rejects-raw-yuv")
    else:
        fail("sdr accepted raw YUV")

    if not SDR_CLIP.is_file():
        print("SKIP sdr parity (sample clip missing)")
        return
    interpreters = require_chipqa_env("PIXVQA_CHIPQA_FEATURE_PYTHON", "PIXVQA_CHIPQA_INFERENCE_PYTHON")
    if interpreters is None:
        return
    feat_py, inf_py = interpreters
    result = run(
        CHIPQA_REPO,
        SDR_CLIP,
        feature_python=feat_py,
        inference_python=inf_py,
        timeout=7200,
    )
    want = BASELINE["sdr_chipqa"]["svr_score"]
    if abs(result["quality_score"] - want) > 1e-5:
        fail(f"sdr score {result['quality_score']} != {want}")
    if result["feature_sha256"] != BASELINE["sdr_chipqa"]["feature_sha256"]:
        fail("sdr feature sha mismatch")
    ok(f"sdr-parity score={want}")


def test_hdr_contract() -> None:
    from pixvqa_hdr_chipqa.adapter import SetupError, run, validate_yuv

    try:
        validate_yuv(HDR_YUV, 960, 540, transfer="hlg")
    except SetupError:
        ok("hdr-rejects-hlg-transfer-token")
    else:
        fail("hdr accepted HLG")

    if not HDR_YUV.is_file():
        print("SKIP hdr parity (derived YUV missing)")
        return
    interpreters = require_chipqa_env("PIXVQA_HDR_CHIPQA_FEATURE_PYTHON", "PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON")
    if interpreters is None:
        return
    feat_py, inf_py = interpreters
    result = run(
        HDR_REPO,
        HDR_YUV,
        960,
        540,
        transfer="pq",
        fps_num=60,
        fps_den=1,
        feature_python=feat_py,
        inference_python=inf_py,
        timeout=7200,
    )
    want = BASELINE["hdr_chipqa"]["svr_score"]
    if abs(result["quality_score"] - want) > 1e-5:
        fail(f"hdr score {result['quality_score']} != {want}")
    if result["feature_sha256"] != BASELINE["hdr_chipqa"]["feature_sha256"]:
        fail("hdr feature sha mismatch")
    ok(f"hdr-parity score={want}")


def test_model_hash_guard() -> None:
    from pixvqa_chipqa.inference import validate_repo, SetupError

    try:
        validate_repo(CHIPQA_REPO)
    except SetupError as exc:
        fail(f"chipqa repo validation failed unexpectedly: {exc}")
    ok("chipqa-bundled-model-hashes")


def main() -> None:
    test_model_hash_guard()
    test_sdr_contract()
    test_hdr_contract()
    print("chipqa reference tests passed")


if __name__ == "__main__":
    main()
