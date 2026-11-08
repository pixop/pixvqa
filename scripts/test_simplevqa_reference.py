#!/usr/bin/env python3
"""Contract tests for the SimpleVQA reference wrapper (no PyTorch required)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pixvqa_simplevqa.adapter import (
    InferenceError,
    SetupError,
    _CPU_OVERLAY_OLD_BLOCK,
    _CUDA_SINGLE_DECODE_MAIN_OLD,
    _CUDA_UNIFIED_FN_ANCHOR,
    cpu_overlay_test_demo,
    cuda_single_decode_overlay_test_demo,
    inspect_video,
    parse_score,
    run,
    runtime_stack_versions,
    validate,
)


def _completed(rc: int, stdout: str = "", stderr: str = "") -> CompletedProcess[str]:
    return CompletedProcess([], returncode=rc, stdout=stdout, stderr=stderr)


def _ffprobe_json_pair(stream_obj: dict, fmt_obj: dict) -> list[CompletedProcess[str]]:
    return [
        _completed(0, json.dumps({"streams": [stream_obj]})),
        _completed(0, json.dumps({"format": fmt_obj})),
    ]


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
    probe = {
        "codec": "hevc",
        "width": 1920,
        "height": 1080,
        "fps": 60.0,
        "frame_count": 600,
        "duration_seconds": 10.0,
        "transfer": "bt709",
    }
    def _identity_overlay(src: Path) -> str:
        return src.read_text(encoding="utf-8")

    with patch("pixvqa_simplevqa.adapter.inspect_video", return_value=probe), patch(
        "pixvqa_simplevqa.adapter.importlib.util.find_spec", return_value=object()
    ), patch("pixvqa_simplevqa.adapter.cpu_overlay_test_demo", side_effect=_identity_overlay), patch(
        "pixvqa_simplevqa.adapter.cuda_single_decode_overlay_test_demo", side_effect=_identity_overlay
    ):
        info = validate(repo, checkpoint, video)
        assert info["input"] == probe
        rs = info["runtime_stack"]
        assert isinstance(rs, dict)
        assert rs.get("python_version")
        assert "python_implementation" in rs
        result = run(repo, checkpoint, video, device="cpu")
        assert result["quality_score"] == 0.723456
        assert result["device"] == "cpu"
        assert result["upstream_test_demo"] == "cpu-overlay"
        assert result["runtime_stack"] == rs
        assert result["temporal_semantics"] == "pooled-video"
        assert result["checkpoint_sha256"]
        assert result["backend"] == "upstream-test_demo"
        result = run(repo, checkpoint, video, device="cuda")
        assert result["device"] == "cuda"
        assert result["upstream_test_demo"] == "symlink-unmodified"
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
        with patch("pixvqa_simplevqa.adapter.inspect_video", return_value={**probe, "transfer": "arib-std-b67"}):
            try:
                validate(repo, checkpoint, video)
            except SetupError:
                pass
            else:
                raise AssertionError("HLG accepted")

    for value in ("nan", "inf", "not-a-score", "1.0,extra"):
        try:
            parse_score(f"{video},{value}\n", video)
        except InferenceError:
            pass
        else:
            raise AssertionError(f"accepted invalid score: {value}")
    try:
        parse_score(f"{video}\n", video)
    except InferenceError:
        pass
    else:
        raise AssertionError("accepted non-CSV line")
    try:
        parse_score(f"{d / 'other.mp4'},0.4\n", video)
    except InferenceError:
        pass
    else:
        raise AssertionError("accepted mismatched video")
    raw = d / "raw.yuv"
    raw.write_bytes(b"raw")
    try:
        inspect_video(raw)
    except SetupError:
        pass
    else:
        raise AssertionError("raw YUV was accepted by upstream demo")

    # --- inspect_video / ffprobe mocks (deterministic preflight) ---
    stream_base = {
        "avg_frame_rate": "60/1",
        "codec_name": "h264",
        "width": 1280,
        "height": 720,
        "color_transfer": "bt709",
    }

    def _patch_ffprobe_sequence(responses: list[CompletedProcess[str]]):
        def _fake_run(*_a, **_k):
            return responses.pop(0)

        return patch("pixvqa_simplevqa.adapter.subprocess.run", side_effect=_fake_run)

    # Malformed nb_frames
    bad_nb = {**stream_base, "nb_frames": "not-an-int"}
    with _patch_ffprobe_sequence(_ffprobe_json_pair(bad_nb, {"duration": "2.0"})):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "nb_frames" in str(e).lower() or "malformed" in str(e).lower()
        else:
            raise AssertionError("malformed nb_frames accepted")

    # Missing nb_frames and missing duration
    no_len = {**stream_base}
    with _patch_ffprobe_sequence(_ffprobe_json_pair(no_len, {})):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "nb_frames" in str(e).lower() or "duration" in str(e).lower()
        else:
            raise AssertionError("missing length accepted")

    # Duration < 1s
    short = {**stream_base, "nb_frames": 30}
    with _patch_ffprobe_sequence(_ffprobe_json_pair(short, {"duration": "0.5"})):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "second" in str(e).lower()
        else:
            raise AssertionError("subsecond duration accepted")

    # nb_frames too low vs rounded fps
    low_fc = {**stream_base, "nb_frames": 20}
    with _patch_ffprobe_sequence(_ffprobe_json_pair(low_fc, {"duration": "5.0"})):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "frame count" in str(e).lower() or "full second" in str(e).lower()
        else:
            raise AssertionError("low frame count accepted")

    # Invalid stream JSON
    with patch(
        "pixvqa_simplevqa.adapter.subprocess.run",
        side_effect=[
            _completed(0, stdout="{ not json"),
            _completed(0, stdout='{"format":{}}'),
        ],
    ):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "json" in str(e).lower()
        else:
            raise AssertionError("invalid JSON accepted")

    # Valid JSON but root is not an object (e.g. list)
    with patch(
        "pixvqa_simplevqa.adapter.subprocess.run",
        side_effect=[
            _completed(0, stdout="[]"),
            _completed(0, stdout='{"format":{}}'),
        ],
    ):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "root" in str(e).lower() or "object" in str(e).lower()
        else:
            raise AssertionError("non-object stream JSON root accepted")

    # Stream record is not an object
    with patch(
        "pixvqa_simplevqa.adapter.subprocess.run",
        side_effect=[
            _completed(0, stdout='{"streams":["not-a-dict"]}'),
            _completed(0, stdout='{"format":{}}'),
        ],
    ):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "stream" in str(e).lower() and "object" in str(e).lower()
        else:
            raise AssertionError("non-object stream element accepted")

    # Format JSON root not an object
    with patch(
        "pixvqa_simplevqa.adapter.subprocess.run",
        side_effect=[
            _completed(0, stdout=json.dumps({"streams": [{**stream_base, "nb_frames": 120}]})),
            _completed(0, stdout='"format-as-string"'),
        ],
    ):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "format" in str(e).lower() and "object" in str(e).lower()
        else:
            raise AssertionError("non-object format JSON root accepted")

    # format value not an object
    with patch(
        "pixvqa_simplevqa.adapter.subprocess.run",
        side_effect=[
            _completed(0, stdout=json.dumps({"streams": [{**stream_base, "nb_frames": 120}]})),
            _completed(0, stdout=json.dumps({"format": "bad"})),
        ],
    ):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "format" in str(e).lower() and "object" in str(e).lower()
        else:
            raise AssertionError("non-object format record accepted")

    # ffprobe launch failure
    def _boom(*_a, **_k):
        raise OSError("no ffprobe")

    with patch("pixvqa_simplevqa.adapter.subprocess.run", side_effect=_boom):
        try:
            inspect_video(video)
        except SetupError as e:
            assert "ffprobe" in str(e).lower()
        else:
            raise AssertionError("missing ffprobe not SetupError")

    # --- upstream subprocess diagnostics (run) ---
    ok_stream = {**stream_base, "nb_frames": 120}
    ok_fmt = {"duration": "2.0"}
    with patch(
        "pixvqa_simplevqa.adapter.importlib.util.find_spec", return_value=object()
    ), patch("pixvqa_simplevqa.adapter.cpu_overlay_test_demo", side_effect=_identity_overlay), patch(
        "pixvqa_simplevqa.adapter.subprocess.run"
    ) as mock_run:
        ffprobe_calls = {"n": 0}

        def _dual_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == "ffprobe":
                ffprobe_calls["n"] += 1
                if ffprobe_calls["n"] == 1:
                    return _completed(0, json.dumps({"streams": [ok_stream]}))
                return _completed(0, json.dumps({"format": ok_fmt}))
            return _completed(1, stderr="upstream stderr tail for diagnostics")

        mock_run.side_effect = _dual_run
        try:
            run(repo, checkpoint, video, device="cpu")
        except InferenceError as e:
            assert "upstream" in str(e).lower()
            assert "stderr" in str(e).lower() or "failed" in str(e).lower()
        else:
            raise AssertionError("failed upstream accepted")

    with patch(
        "pixvqa_simplevqa.adapter.importlib.util.find_spec", return_value=object()
    ), patch("pixvqa_simplevqa.adapter.cpu_overlay_test_demo", side_effect=_identity_overlay), patch(
        "pixvqa_simplevqa.adapter.subprocess.run"
    ) as mock_run2:
        ffprobe_calls2 = {"n": 0}

        def _dual_run2(cmd, *args, **kwargs):
            if cmd and cmd[0] == "ffprobe":
                ffprobe_calls2["n"] += 1
                if ffprobe_calls2["n"] == 1:
                    return _completed(0, json.dumps({"streams": [ok_stream]}))
                return _completed(0, json.dumps({"format": ok_fmt}))
            return _completed(1, stdout="print fail", stderr="")

        mock_run2.side_effect = _dual_run2
        try:
            run(repo, checkpoint, video, device="cpu")
        except InferenceError as e:
            assert "stdout" in str(e).lower() or "fail" in str(e).lower()
        else:
            raise AssertionError("empty-stderr failure not surfaced")

    checkpoint.unlink()
    try:
        validate(repo, checkpoint, video, check_imports=False, probe_video=False)
    except SetupError:
        pass
    else:
        raise AssertionError("missing checkpoint accepted")

    # Pip metadata snapshot (stdlib only; optional wheels may be absent in CI)
    meta = runtime_stack_versions()
    assert meta.get("python_version")
    assert meta.get("python_implementation")

    # --- CUDA single-decode overlay (no PyTorch) ---
    real_upstream = ROOT / "deps/SimpleVQA/test_demo.py"
    if real_upstream.is_file():
        patched_cuda = cuda_single_decode_overlay_test_demo(real_upstream)
        assert "video_processing_unified" in patched_cuda
        assert _CUDA_SINGLE_DECODE_MAIN_OLD not in patched_cuda
        assert "video_processing_spatial(config.dist)" not in patched_cuda.split("def main")[1]
        assert _CUDA_UNIFIED_FN_ANCHOR in patched_cuda
        bad_cuda = d / "bad_cuda_upstream"
        bad_cuda.mkdir()
        (bad_cuda / "test_demo.py").write_text("# no main block\n", encoding="utf-8")
        try:
            cuda_single_decode_overlay_test_demo(bad_cuda / "test_demo.py")
        except SetupError as e:
            assert "layout changed" in str(e).lower() or "not found" in str(e).lower()
        else:
            raise AssertionError("missing CUDA overlay block accepted")

    # --- CPU overlay text generation (no PyTorch) ---
    overlay_src = d / "overlay_upstream"
    overlay_src.mkdir()
    (overlay_src / "test_demo.py").write_text(
        "# header\n" + _CPU_OVERLAY_OLD_BLOCK + "\nwith torch.no_grad():\n"
        "    feature_motion = torch.zeros([n_clip, 2048+256])\n",
        encoding="utf-8",
    )
    patched = cpu_overlay_test_demo(overlay_src / "test_demo.py")
    assert "DataParallel" not in patched
    assert "map_location=device" in patched
    assert "torch.inference_mode()" in patched
    assert "device=device" in patched
    bad_src = d / "bad_upstream"
    bad_src.mkdir()
    (bad_src / "test_demo.py").write_text("# no matching block\n", encoding="utf-8")
    try:
        cpu_overlay_test_demo(bad_src / "test_demo.py")
    except SetupError as e:
        assert "layout changed" in str(e).lower() or "not found" in str(e).lower()
    else:
        raise AssertionError("missing overlay block accepted")

    # Opt-in: full CPU inference on official fixture (slow; needs deps + GPU weights cache).
    _smoke = ROOT / "deps/SimpleVQA"
    _official = _smoke / "videos/2999049224_original_centercrop_960x540_8s.mp4"
    _ckpt = _smoke / "ckpts/UGC_BVQA_model.pth"
    if __import__("os").environ.get("SIMPLEVQA_CPU_SMOKE") == "1":
        if not (_smoke / "test_demo.py").is_file() or not _official.is_file() or not _ckpt.is_file():
            print("SKIP: SIMPLEVQA_CPU_SMOKE=1 but official deps/fixture missing", file=sys.stderr)
        else:
            rec = run(_smoke, _ckpt, _official, device="cpu", timeout=600)
            assert rec["device"] == "cpu"
            assert rec["upstream_test_demo"] == "cpu-overlay"
            assert 0.0 < rec["quality_score"] < 10.0
            print(f"PASS: CPU smoke score {rec['quality_score']}", file=sys.stderr)

print(
    "PASS: SimpleVQA reference adapter preflight, staging, output, "
    "PQ/HLG rejection, score/ffprobe/upstream error handling"
)
