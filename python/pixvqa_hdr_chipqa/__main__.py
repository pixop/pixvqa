"""HDR-ChipQA unified runner (SDR BT.709 + PQ HDR + experimental HLG; `pixvqa --metrics hdr-chipqa`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import InferenceError, SetupError, run, validate


def protocol_lines(record: dict, output_json) -> list[str]:
    return [
        "pixvqa-protocol 1",
        "pixvqa-value metric hdr-chipqa",
        f"pixvqa-value quality_score {record.get('quality_score')!r}",
        "pixvqa-value temporal_semantics pooled-video",
        "pixvqa-value device 'cpu'",
        f"pixvqa-value output_json {output_json if output_json else 'none'}",
        "pixvqa-end",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path, help="Upstream deps/hdr_chipqa checkout")
    ap.add_argument("--test", required=True, type=Path, help="SDR compressed or PQ/SDR raw YUV test")
    ap.add_argument("--width", type=int, default=0, help="Raw YUV width (required for .yuv)")
    ap.add_argument("--height", type=int, default=0, help="Raw YUV height (required for .yuv)")
    ap.add_argument("--bit-depth", type=int, default=0, help="Raw YUV bit depth (8=SDR, 10=PQ HDR)")
    ap.add_argument(
        "--transfer",
        default="",
        help="Transfer hint: sdr, pq, hlg, arib-std-b67 (HLG is experimental and uncalibrated)",
    )
    ap.add_argument("--fps-num", type=int, default=0)
    ap.add_argument("--fps-den", type=int, default=0)
    ap.add_argument("--feature-python", default="")
    ap.add_argument("--inference-python", default="")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-frames", type=int, default=-1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--output-json", default="")
    ap.add_argument("--transient-output", action="store_true")
    ap.add_argument("--protocol", choices=("json", "lines"), default="json")
    args = ap.parse_args(argv)
    if args.max_frames not in (-1, 0):
        print("pixvqa-hdr-chipqa: --max-frames is not enforced (full-clip pooled features)",
              file=sys.stderr)
    transfer = args.transfer or None
    fps_num = args.fps_num if args.fps_num > 0 else None
    fps_den = args.fps_den if args.fps_den > 0 else None
    width = args.width if args.width > 0 else None
    height = args.height if args.height > 0 else None
    bit_depth = args.bit_depth if args.bit_depth > 0 else None
    try:
        if args.check:
            result = validate(
                args.repo,
                args.test,
                width,
                height,
                bit_depth=bit_depth,
                transfer=transfer,
                fps_num=fps_num,
                fps_den=fps_den,
                feature_python=args.feature_python or None,
                inference_python=args.inference_python or None,
                ffprobe=args.ffprobe,
            )
            result["status"] = "assets_present"
        else:
            result = run(
                args.repo,
                args.test,
                width,
                height,
                bit_depth=bit_depth,
                transfer=transfer,
                fps_num=fps_num,
                fps_den=fps_den,
                feature_python=args.feature_python or None,
                inference_python=args.inference_python or None,
                ffprobe=args.ffprobe,
                timeout=args.timeout,
            )
    except SetupError as exc:
        print(f"pixvqa-hdr-chipqa: setup: {exc}", file=sys.stderr)
        return 2
    except (InferenceError, OSError) as exc:
        print(f"pixvqa-hdr-chipqa: runtime: {exc}", file=sys.stderr)
        return 1
    output_json = None if args.transient_output else (args.output_json or None)
    if not args.check and args.protocol == "lines":
        if not args.output_json:
            print("pixvqa-hdr-chipqa: --protocol lines requires --output-json", file=sys.stderr)
            return 2
        print("\n".join(protocol_lines(result, output_json)))
        return 0
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
