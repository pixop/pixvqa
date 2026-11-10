#!/usr/bin/env python3
"""Check SimpleVQA staged inference code keeps both learned models in eval mode."""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pixvqa_simplevqa.adapter import (  # noqa: E402
    _CPU_OVERLAY_OLD_BLOCK,
    _CUDA_SINGLE_DECODE_MAIN_OLD,
    _CUDA_UNIFIED_FN_ANCHOR,
    cpu_overlay_test_demo,
    cuda_single_decode_overlay_test_demo,
    cuda_shared_gpu_overlay_test_demo,
)


def main() -> int:
    # The score record uses os.environ for decoder provenance. Ensure the
    # runtime adapter imports os (a previous regression raised NameError).
    adapter_file = Path(__file__).resolve().parents[1] / "python" / "pixvqa_simplevqa" / "adapter.py"
    module = ast.parse(adapter_file.read_text(encoding="utf-8"))
    imports_os = any(
        isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names)
        for node in module.body
    )
    assert imports_os, "SimpleVQA adapter references os.environ but does not import os"
    # Synthetic upstream fixture contains precisely the pinned textual anchors.
    src_text = (
        "def slowfast():\n    pass\n\n"
        "def pack_pathway_output(frames, device):\n    pass\n\n"
        "def video_processing_spatial(dist):\n"
        "    cap = cv2.VideoCapture(dist)\n"
        "    return cap, dist\n\n"
        "def main(config):\n"
        "    model_motion = slowfast()\n"
        "    model_motion = model_motion.to(device)\n"
        + _CPU_OVERLAY_OLD_BLOCK + "\n"
        + "    with torch.no_grad():\n"
        + "        feature_motion = torch.zeros([n_clip, 2048+256])\n"
        + "        if True:\n"
        + "            video_dist_spatial = video_dist_spatial.to(device)\n"
        + "            for idx, ele in enumerate(video_dist_motion):\n"
        + "                ele = pack_pathway_output(ele, device)\n"
        + "                ele_slow_feature, ele_fast_feature = model_motion(ele)\n"
        + "            outputs = model(video_dist_spatial, feature_motion)\n"
        + _CUDA_SINGLE_DECODE_MAIN_OLD + "\n"
        + _CUDA_UNIFIED_FN_ANCHOR + "\n    pass\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test_demo.py"
        path.write_text(src_text, encoding="utf-8")
        for name, transform in (
            ("cpu", cpu_overlay_test_demo),
            ("cuda", cuda_single_decode_overlay_test_demo),
            ("shared-gpu", cuda_shared_gpu_overlay_test_demo),
        ):
            staged = transform(path)
            assert (
                "    model_motion = model_motion.to(device)\n"
                "    model_motion.eval()"
            ) in staged, name
            assert staged.count("model_motion.eval()") == (
                2 if name == "shared-gpu" else 1
            ), name
            assert "def _pixvqa_deterministic_inference():" in staged, name
            assert staged.count("_pixvqa_deterministic_inference()") == (
                3 if name == "shared-gpu" else 2
            ), name
            assert "torch.use_deterministic_algorithms(True)" in staged, name
            assert "def _pixvqa_trace_tensor(stage, value):" in staged, name
            assert "_pixvqa_video_capture(dist)" in staged, name
            assert "from pixvqa_video.ffmpeg_bgr import FFmpegCapture as _pixvqa_video_capture" in staged, name
            assert "PIXVQA_LEARNED_DECODER" not in staged, name
            assert "PIXVQA_SIMPLEVQA_LEGACY_BOUNDED" not in staged, name
            assert "video_processing_unified" not in staged, name
            assert 'get("PIXVQA_LEARNED_DETERMINISTIC", "1")' in staged, name
            assert "cv2.VideoCapture(dist)" not in staged, name
            assert "_pixvqa_trace_tensor('spatial_input', video_dist_spatial)" in staged, name
            assert "_pixvqa_trace_tensor('motion_features', feature_motion)" in staged, name
            compile(staged, f"<{name}-overlay>", "exec")
            print(f"OK   simplevqa-{name}-motion-eval")
    print("PASS: SimpleVQA motion evaluation-mode regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
