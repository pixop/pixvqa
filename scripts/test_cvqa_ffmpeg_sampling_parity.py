#!/usr/bin/env python3
"""Check CVQA production sampler against pinned upstream frame selection.

Decodes a short prefix with FFmpeg BGR24 twice: once through the new
production sampler and once through the pinned upstream transform/schedule
written out explicitly. No model checkpoints or CUDA needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms
    from pixvqa_video.ffmpeg_bgr import FFmpegCapture
    from pixvqa_cvqahdr.adapter import preprocess_custom_interval, sampling_plan, sample_schedule

    with FFmpegCapture(str(args.video)) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(frame_count, args.frames)
    plan = sampling_plan(limit, fps)
    actual, _ = preprocess_custom_interval(str(args.video), plan, torch)

    size = 384
    if height > width:
        rw = size
        rh = int(rw / width * height)
    else:
        rh = size
        rw = int(rh / height * width)
    transformation = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    reference = torch.zeros_like(actual)
    fps_rounded = int(round(fps))
    sampled = 0
    with FFmpegCapture(str(args.video)) as capture:
        for index in range(limit):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"decoding failed at frame {index}")
            # Pinned NR/VQA_NR.py condition, not the production sample_schedule().
            if index % fps_rounded == int(fps_rounded / 2) and sampled < reference.shape[0]:
                resized = cv2.resize(bgr, (rw, rh))
                rgb = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                reference[sampled] = transformation(rgb)
                sampled += 1

    assert sampled == actual.shape[0], (sampled, actual.shape[0])
    assert torch.equal(actual, reference), (
        "production CVQA preprocess differs from pinned NR midpoint transform",
        (actual - reference).abs().max().item(),
    )
    indices = [x["frame_index"] for x in sample_schedule(plan)]
    print(f"PASS: CVQA FFmpeg sample tensors identical at {len(indices)} midpoints {indices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
