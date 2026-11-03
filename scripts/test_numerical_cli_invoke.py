#!/usr/bin/env python3
"""Regression tests for numerical-suite pixvqa invocation generation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.cli_invoke import (  # noqa: E402
    build_invoke_context,
    build_pixvqa_command,
    classify_failure,
    frame_limit_for,
    needs_explicit_hdr_transfer,
    resolve_vmaf_model,
)
from numerical_conformance.metric_caps import GROUP_METRICS, includes_cvqa, includes_cvvdp  # noqa: E402
from numerical_conformance.registry import cvvdp_skip_reason  # noqa: E402


def test_vmaf_requires_model_flag() -> None:
    ctx = build_invoke_context(ROOT / "build" / "full" / "pixvqa")
    inv = build_pixvqa_command(
        ctx,
        family="sdr",
        scenario="raw_raw",
        metric="vmaf",
        ref="/ref.yuv",
        test="/test.yuv",
    )
    assert "--model" in inv.argv
    idx = inv.argv.index("--model")
    model = Path(inv.argv[idx + 1])
    assert "2160" in model.name
    assert "hdr_vmaf" not in model.name


def test_pq_vmaf_hdr_model() -> None:
    model = resolve_vmaf_model(ROOT / "build" / "full" / "pixvqa", "pq")
    assert model is not None
    assert "hdr_vmaf" in str(model)


def test_pu21_raw_explicit_transfer() -> None:
    assert needs_explicit_hdr_transfer("pq", "raw_raw", "pu21-psnr", force_explicit=False)
    assert not needs_explicit_hdr_transfer("pq", "comp_comp", "pu21-psnr", force_explicit=False)


def test_cvqa_omits_frames() -> None:
    assert frame_limit_for("cvqa-hdr-nr") is None
    ctx = build_invoke_context(ROOT / "build" / "full" / "pixvqa")
    ctx.learned_python = "/learned/bin/python"
    inv = build_pixvqa_command(
        ctx,
        family="pq",
        scenario="raw_comp",
        metric="cvqa-hdr-nr",
        ref="",
        test="/dist.mp4",
        no_reference=True,
    )
    assert "-n" not in inv.argv and "--frames" not in inv.argv


def test_group_sdr_display_and_no_frames_with_cvqa() -> None:
    ctx = build_invoke_context(ROOT / "build" / "full" / "pixvqa")
    ctx.learned_python = "/learned/bin/python"
    inv = build_pixvqa_command(
        ctx,
        family="sdr",
        scenario="comp_comp",
        metric="group:sdr",
        ref="/ref.mkv",
        test="/test.mp4",
    )
    assert "--cvvdp-display" in inv.argv
    assert inv.argv[inv.argv.index("--cvvdp-display") + 1] == "standard_4k"
    assert "-n" not in inv.argv
    assert includes_cvqa("group:sdr")
    assert includes_cvvdp("group:sdr")
    assert "cvqa-hdr-nr" in GROUP_METRICS["sdr"]
    assert "cvvdp" in GROUP_METRICS["sdr"]


def test_cvvdp_raw_transfer() -> None:
    assert needs_explicit_hdr_transfer("hlg", "raw_raw", "cvvdp", force_explicit=False)


def test_cvqa_only_no_cvvdp_cli_flags() -> None:
    ctx = build_invoke_context(ROOT / "build" / "full" / "pixvqa")
    ctx.learned_python = "/learned/bin/python"
    inv = build_pixvqa_command(
        ctx,
        family="pq",
        scenario="raw_comp",
        metric="cvqa-hdr-nr",
        ref="",
        test="/dist.mp4",
        no_reference=True,
    )
    assert "--cvvdp-python" not in inv.argv
    assert "--cvvdp-display" not in inv.argv
    assert "PIXVQA_CVVDP_PYTHON" not in inv.env
    assert "--cvqahdr-python" in inv.argv


def test_cvvdp_only_no_cvqahdr_cli_flags() -> None:
    ctx = build_invoke_context(ROOT / "build" / "full" / "pixvqa")
    ctx.learned_python = "/learned/bin/python"
    inv = build_pixvqa_command(
        ctx,
        family="pq",
        scenario="comp_comp",
        metric="cvvdp",
        ref="/ref.mkv",
        test="/test.mp4",
    )
    assert "--cvqahdr-python" not in inv.argv
    assert "PIXVQA_CVQAHDR_PYTHON" not in inv.env
    assert "--cvvdp-python" in inv.argv


def test_sdr_cvvdp_raw_skip_reason() -> None:
    assert cvvdp_skip_reason("cvvdp", "sdr", "raw_raw")
    assert cvvdp_skip_reason("group:sdr", "sdr", "raw_comp")
    assert cvvdp_skip_reason("cvvdp", "pq", "raw_raw") is None


def test_classify_checkpoint_skip() -> None:
    status, _ = classify_failure("pixvqa-cvqahdr: checkpoint not found", 2, "cvqa-hdr-nr")
    assert status == "SKIP"


def test_classify_python_module_as_fail() -> None:
    status, _ = classify_failure("No module named pixvqa_cvvdp", 2, "cvvdp")
    assert status == "FAIL"


def test_invocation_script_import_check() -> None:
    """Catch NameError-style regressions before any test body runs on RTX."""
    import py_compile

    py_compile.compile(__file__, doraise=True)
    assert includes_cvqa("group:pq")
    assert "cvvdp" in GROUP_METRICS["hlg"]


def main() -> int:
    test_invocation_script_import_check()
    print("OK   invocation-script-import-check")
    tests = [
        test_vmaf_requires_model_flag,
        test_pq_vmaf_hdr_model,
        test_pu21_raw_explicit_transfer,
        test_cvqa_omits_frames,
        test_group_sdr_display_and_no_frames_with_cvqa,
        test_cvqa_only_no_cvvdp_cli_flags,
        test_cvvdp_only_no_cvqahdr_cli_flags,
        test_cvvdp_raw_transfer,
        test_sdr_cvvdp_raw_skip_reason,
        test_classify_checkpoint_skip,
        test_classify_python_module_as_fail,
    ]
    for fn in tests:
        fn()
        print(f"OK   {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
