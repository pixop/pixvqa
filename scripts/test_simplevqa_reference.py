#!/usr/bin/env python3
"""Contract tests for the SimpleVQA reference wrapper (no PyTorch required)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pixvqa_simplevqa.adapter import (
    InferenceError, SetupError, parse_score, run, validate,
)

with tempfile.TemporaryDirectory(prefix="pixvqa_svqa_test_") as tmp:
    d = Path(tmp)
    repo = d / "upstream"
    repo.mkdir()
    (repo / "model").mkdir()
    (repo / "model" / "UGC_BVQA_model.py").write_text("# mock model\n")
    # Model demo simulates the official CLI arguments and writes the same
    # one-line CSV output. Tests subprocess staging and score parsing, not
    # official PyTorch inference.
    (repo / "test_demo.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--method_name');p.add_argument('--dist');"
        "p.add_argument('--output');p.add_argument('--is_gpu',action='store_true')\n"
        "a=p.parse_args()\n"
        "assert a.method_name == 'single-scale'\n"
        "assert Path('ckpts/UGC_BVQA_model.pth').is_file()\n"
        "assert Path('model/UGC_BVQA_model.py').is_file()\n"
        "Path(a.output).write_text(f'{a.dist},0.723456\\n')\n",
        encoding="utf-8",
    )
    checkpoint = d / "weights.pth"
    checkpoint.write_bytes(b"fake-weights-for-wrapper-contract-only")
    video = d / "tiny.mp4"
    video.write_bytes(b"fake-video-for-wrapper-contract-only")
    probe = {"codec": "hevc", "width": 1920, "height": 1080,
             "fps": 60.0, "frame_count": 600, "transfer": "bt709"}
    with patch("pixvqa_simplevqa.adapter.inspect_video", return_value=probe), patch(
        "pixvqa_simplevqa.adapter.importlib.util.find_spec", return_value=object()
    ):
        info = validate(repo, checkpoint, video)
        assert info["input"] == probe
        result = run(repo, checkpoint, video, device="cpu")
        assert result["quality_score"] == 0.723456
        assert result["temporal_semantics"] == "pooled-video"
        assert result["checkpoint_sha256"]
        assert result["backend"] == "upstream-test_demo"
        result = run(repo, checkpoint, video, device="cuda")
        assert result["device"] == "cuda"
        try:
            run(repo, checkpoint, video, device="gpu")
        except SetupError:
            pass
        else:
            raise AssertionError("unsupported device accepted")
        with patch("pixvqa_simplevqa.adapter.inspect_video", return_value={**probe, "transfer": "smpte2084"}):
            try:
                validate(repo, checkpoint, video)
            except SetupError:
                pass
            else:
                raise AssertionError("PQ accepted")
    for value in ("nan", "inf", "not-a-score", "1.0,extra"):
        try:
            parse_score(f"{video},{value}\n", video)
        except InferenceError:
            pass
        else:
            raise AssertionError(f"accepted invalid score: {value}")
    try:
        parse_score(f"{d / 'other.mp4'},0.4\n", video)
    except InferenceError:
        pass
    else:
        raise AssertionError("accepted mismatched video")
    raw = d / "raw.yuv"
    raw.write_bytes(b"raw")
    try:
        from pixvqa_simplevqa.adapter import inspect_video
        inspect_video(raw)
    except SetupError:
        pass
    else:
        raise AssertionError("raw YUV was accepted by upstream demo")
    checkpoint.unlink()
    try:
        validate(repo, checkpoint, video, check_imports=False, probe_video=False)
    except SetupError:
        pass
    else:
        raise AssertionError("missing checkpoint accepted")

print("PASS: SimpleVQA reference adapter setup, staging, output, unsupported transfer and errors")
