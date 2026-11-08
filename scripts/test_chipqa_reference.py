#!/usr/bin/env python3
"""HDR-ChipQA unified reference parity (SDR BT.709 + PQ HDR goldens; HLG experimental contract)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

BASELINE = json.loads((ROOT / "reports/chipqa-reference/REFERENCE_BASELINE.json").read_text())
SDR_CLIP = ROOT / BASELINE["sdr_chipqa"]["clip"]
HDR_YUV = ROOT / BASELINE["hdr_chipqa"]["derived_yuv"]
HDR_REPO = ROOT / "deps/hdr_chipqa"

HLG_WARNING = "WARNING: HDR-ChipQA on HLG is experimental"


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
        return str(raw)
    fail(f"{env_var}={raw!r} is not a Python interpreter")


def require_env(feature_var: str, inference_var: str) -> tuple[str, str] | None:
    feat = resolve_test_python(feature_var)
    inf = resolve_test_python(inference_var)
    if feat is None or inf is None:
        missing = [v for v, x in ((feature_var, feat), (inference_var, inf)) if x is None]
        print(f"SKIP parity ({', '.join(missing)} unset; see docs/dev/CHIPQA_REFERENCE.md)")
        return None
    return feat, inf


def test_sdr_compressed() -> None:
    from pixvqa_hdr_chipqa.adapter import run

    if not SDR_CLIP.is_file():
        print("SKIP sdr compressed parity (sample clip missing)")
        return
    interpreters = require_env("PIXVQA_HDR_CHIPQA_FEATURE_PYTHON", "PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON")
    if interpreters is None:
        return
    feat_py, inf_py = interpreters
    result = run(
        HDR_REPO,
        SDR_CLIP,
        feature_python=feat_py,
        inference_python=inf_py,
        timeout=7200,
    )
    from pixvqa_hdr_chipqa import SDR_GOLDEN_FEATURE_SHA256, SDR_GOLDEN_SCORE

    if result["pipeline"] != "sdr" or result["color_space"] != "BT709":
        fail(f"sdr pipeline mislabeled: {result.get('pipeline')} {result.get('color_space')}")
    if abs(result["quality_score"] - SDR_GOLDEN_SCORE) > 1e-5:
        fail(f"sdr score {result['quality_score']} != {SDR_GOLDEN_SCORE}")
    if result["feature_sha256"] != SDR_GOLDEN_FEATURE_SHA256:
        fail("sdr feature sha mismatch")
    ok(f"sdr-bt709-parity score={SDR_GOLDEN_SCORE}")


def test_transfer_tokens() -> None:
    from pixvqa_hdr_chipqa.adapter import SetupError, classify_transfer_token, resolve_pipeline_mode

    if classify_transfer_token("bt709") != "sdr":
        fail("bt709 token not sdr")
    if classify_transfer_token("smpte2084") != "pq":
        fail("pq token not pq")
    if classify_transfer_token("arib-std-b67") != "hlg":
        fail("arib-std-b67 token not hlg")
    if classify_transfer_token("hlg") != "hlg":
        fail("hlg token not hlg")
    ok("transfer-tokens")

    yuv = Path("/tmp/sdr_dist_960x540_60f.yuv")
    if not yuv.is_file():
        print("SKIP sdr matrix guard (decoded YUV missing)")
        return

    sdr_pipe = resolve_pipeline_mode(yuv, transfer="sdr", width=960, height=540, bit_depth=8)
    if sdr_pipe["color_space"] != "BT709":
        fail("SDR raw mislabeled matrix")

    hlg_pipe = resolve_pipeline_mode(yuv, transfer="hlg", width=960, height=540, bit_depth=10)
    if hlg_pipe["mode"] != "hlg" or hlg_pipe["color_space"] != "BT2020":
        fail(f"HLG raw pipeline wrong: {hlg_pipe}")
    ok("hlg-raw-pipeline-bt2020")

    try:
        resolve_pipeline_mode(yuv, transfer="hlg", width=960, height=540, bit_depth=8)
    except SetupError:
        ok("hlg-rejects-8bit-raw")
    else:
        fail("accepted 8-bit HLG raw")

    try:
        resolve_pipeline_mode(yuv, transfer="smpte428", width=960, height=540, bit_depth=10)
    except SetupError:
        ok("rejects-unsupported-hdr-transfer")
    else:
        fail("accepted smpte428")


def test_hdr_contract() -> None:
    from pixvqa_hdr_chipqa.adapter import SetupError, run, validate_yuv

    try:
        validate_yuv(HDR_YUV, 960, 540, transfer="hlg")
    except SetupError as exc:
        if "HLG requires" not in str(exc) and "10-bit" not in str(exc):
            fail(f"unexpected HLG validate error: {exc}")
    else:
        ok("hlg-accepts-raw-yuv-validation")

    try:
        validate_yuv(HDR_YUV, 960, 540, transfer="hlg", color_space="BT709")
    except SetupError:
        ok("hlg-rejects-bt709-matrix")
    else:
        fail("HLG accepted BT709 matrix")

    if not HDR_YUV.is_file():
        print("SKIP hdr parity (derived YUV missing)")
        return
    interpreters = require_env("PIXVQA_HDR_CHIPQA_FEATURE_PYTHON", "PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON")
    if interpreters is None:
        return
    feat_py, inf_py = interpreters
    result = run(
        HDR_REPO,
        HDR_YUV,
        960,
        540,
        bit_depth=10,
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
    if result["color_space"] != "BT2020" or result["pipeline"] != "pq":
        fail("hdr PQ path must report BT2020/pq")
    if result.get("experimental"):
        fail("PQ path must not be marked experimental")
    ok(f"hdr-pq-parity score={want}")


def test_hlg_raw_scoring() -> None:
    from pixvqa_hdr_chipqa.adapter import run

    if not HDR_YUV.is_file():
        print("SKIP hlg raw scoring (derived YUV missing)")
        return
    interpreters = require_env("PIXVQA_HDR_CHIPQA_FEATURE_PYTHON", "PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON")
    if interpreters is None:
        return
    feat_py, inf_py = interpreters
    import io
    from contextlib import redirect_stderr

    buf = io.StringIO()
    with redirect_stderr(buf):
        result = run(
            HDR_REPO,
            HDR_YUV,
            960,
            540,
            bit_depth=10,
            transfer="hlg",
            fps_num=60,
            fps_den=1,
            feature_python=feat_py,
            inference_python=inf_py,
            timeout=7200,
        )
    err = buf.getvalue()
    if HLG_WARNING not in err:
        fail("HLG experimental warning missing from stderr")
    if result["pipeline"] != "hlg" or not result.get("experimental"):
        fail(f"HLG run not labeled experimental: {result.get('pipeline')}")
    if result["color_space"] != "BT2020":
        fail("HLG must use BT2020 upstream matrix, not relabeled PQ")
    # Same YUV codes → same features as PQ path (upstream color_space=BT2020 only).
    if result["feature_sha256"] != BASELINE["hdr_chipqa"]["feature_sha256"]:
        fail("HLG BT2020 features must match PQ golden (same YUV domain, no EOTF)")
    ok("hlg-raw-experimental-bt2020-features")


def test_hlg_mismatch_and_compressed() -> None:
    from pixvqa_hdr_chipqa.adapter import SetupError, resolve_pipeline_mode

    ffmpeg = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    ffprobe = subprocess.run(["which", "ffprobe"], capture_output=True, text=True)
    if ffmpeg.returncode != 0 or ffprobe.returncode != 0:
        print("SKIP hlg compressed tests (ffmpeg/ffprobe missing)")
        return

    with tempfile.TemporaryDirectory(prefix="pixvqa-hlg-chipqa-") as tmp:
        tmp_path = Path(tmp)
        yuv = tmp_path / "hlg10.yuv"
        container = tmp_path / "hlg10.mkv"
        frame_bytes = 960 * 540 * 3
        if HDR_YUV.is_file():
            yuv.write_bytes(HDR_YUV.read_bytes()[: frame_bytes * 5])
        else:
            yuv.write_bytes(b"\x00\x80" * (frame_bytes * 5))
        enc = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "960x540", "-r", "60",
                "-i", str(yuv),
                "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                "-bsf:v",
                "hevc_metadata=colour_primaries=9:transfer_characteristics=18:"
                "matrix_coefficients=9:video_full_range_flag=0",
                "-frames:v", "5", str(container),
            ],
            capture_output=True,
            text=True,
        )
        if enc.returncode != 0:
            print(f"SKIP hlg compressed encode failed: {enc.stderr[-300:]}")
            return
        pipe = resolve_pipeline_mode(container)
        if pipe["mode"] != "hlg" or pipe["bit_depth"] != 10:
            fail(f"compressed HLG misrouted: {pipe}")
        ok("hlg-compressed-metadata-dispatch")
        try:
            resolve_pipeline_mode(container, transfer="pq")
        except SetupError:
            ok("hlg-compressed-rejects-pq-cli")
        else:
            fail("accepted PQ CLI on HLG compressed")

    if SDR_CLIP.is_file():
        try:
            resolve_pipeline_mode(SDR_CLIP, transfer="hlg")
        except SetupError:
            ok("sdr-compressed-rejects-hlg-cli")
        else:
            fail("accepted HLG CLI on SDR compressed")


def test_model_hash_guard() -> None:
    from pixvqa_hdr_chipqa.inference import validate_repo

    validate_repo(HDR_REPO)
    ok("hdr-chipqa-bundled-model-hashes")


def test_upstream_color_space_independent() -> None:
    repo_script = HDR_REPO / "hdr_chipqa.py"
    if not repo_script.is_file():
        print("SKIP upstream color_space check (deps/hdr_chipqa missing)")
        return
    text = repo_script.read_text(encoding="utf-8")
    if "color_transfer" in text or "HLG" in text or "PQ" in text.split("parser")[0]:
        pass  # transfer not in CLI — expected
    if "'BT2020','BT709'" not in text.replace(" ", "") and '{"BT2020","BT709"}' not in text:
        if "choices={'BT2020','BT709'}" not in text:
            fail("upstream hdr_chipqa.py color_space choices not found")
    ok("upstream-hdr-chipqa-color-space-bt2020-bt709-only")


def main() -> None:
    test_model_hash_guard()
    test_transfer_tokens()
    test_upstream_color_space_independent()
    test_sdr_compressed()
    test_hdr_contract()
    test_hlg_mismatch_and_compressed()
    test_hlg_raw_scoring()
    print("hdr-chipqa reference tests passed")


if __name__ == "__main__":
    main()
