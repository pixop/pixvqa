"""ChipQA SDR reference runner (standalone and PixVQA `pixvqa --metrics chipqa` backend)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import InferenceError, SetupError, run, validate


def protocol_lines(record: dict, output_json) -> list[str]:
    lines = [
        "pixvqa-protocol 1",
        "pixvqa-value metric chipqa",
        f"pixvqa-value quality_score {record.get('quality_score')!r}",
        "pixvqa-value temporal_semantics pooled-video",
        f"pixvqa-value device 'cpu'",
        f"pixvqa-value output_json {output_json if output_json else 'none'}",
        "pixvqa-end",
    ]
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path, help="Upstream deps/ChipQA checkout")
    ap.add_argument("--test", required=True, type=Path, help="Compressed SDR test video")
    ap.add_argument("--feature-python", default="", help="Python for upstream chipqa.py")
    ap.add_argument("--inference-python", default="", help="Python 3.10 + sklearn 1.0.2 for SVR")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-frames", type=int, default=-1,
                    help="Accepted for CLI parity; upstream scores the full clip.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--output-json", default="")
    ap.add_argument("--transient-output", action="store_true")
    ap.add_argument("--protocol", choices=("json", "lines"), default="json")
    args = ap.parse_args(argv)
    if args.max_frames not in (-1, 0):
        print("pixvqa-chipqa: --max-frames is not enforced (full-clip pooled features)",
              file=sys.stderr)
    feat_py = args.feature_python or None
    inf_py = args.inference_python or None
    try:
        if args.check:
            result = validate(args.repo, args.test, ffprobe=args.ffprobe,
                              feature_python=feat_py, inference_python=inf_py)
            result["status"] = "assets_present"
            result["note"] = "Successful feature extract and inference not checked."
        else:
            result = run(
                args.repo,
                args.test,
                feature_python=feat_py,
                inference_python=inf_py,
                ffprobe=args.ffprobe,
                timeout=args.timeout,
            )
    except SetupError as exc:
        print(f"pixvqa-chipqa: setup: {exc}", file=sys.stderr)
        return 2
    except (InferenceError, OSError) as exc:
        print(f"pixvqa-chipqa: runtime: {exc}", file=sys.stderr)
        return 1
    output_json = None if args.transient_output else (args.output_json or None)
    if not args.check and args.protocol == "lines":
        if not args.output_json:
            print("pixvqa-chipqa: --protocol lines requires --output-json", file=sys.stderr)
            return 2
        print("\n".join(protocol_lines(result, output_json)))
        return 0
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
