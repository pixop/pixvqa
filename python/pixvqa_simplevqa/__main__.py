"""SimpleVQA reference runner (standalone and PixVQA `pixvqa --metrics simplevqa` backend)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pixvqa_scheduler.admission import AdmissionError, optional_admission

from .adapter import InferenceError, SetupError, run, validate


def resolve_device(req: str) -> str:
    if req in ("cpu", "cuda"):
        return req
    if req == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    raise ValueError(req)


def protocol_lines(record: dict, output_json) -> list[str]:
    device = record.get("device") or "unknown"
    score = record.get("quality_score")
    temporal = str(record.get("temporal_semantics") or "pooled-video")
    lines = [
        "pixvqa-protocol 1",
        "pixvqa-value metric simplevqa",
        f"pixvqa-value quality_score {score!r}",
        f"pixvqa-value device {device!r}",
        f"pixvqa-value temporal_semantics {temporal!r}",
        f"pixvqa-value output_json {output_json if output_json else 'none'}",
        "pixvqa-end",
    ]
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-frames", type=int, default=-1,
                    help="Accepted for CLI parity; upstream demo scores the full clip.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--output-json", default="")
    ap.add_argument(
        "--transient-output",
        action="store_true",
        help="Internal: write JSON to --output-json path but do not persist naming in reports.",
    )
    ap.add_argument("--protocol", choices=("json", "lines"), default="json")
    ap.add_argument(
        "--use-worker",
        action="store_true",
        help="Opt-in: reuse a persistent CUDA worker to amortize import cost across clips.",
    )
    ap.add_argument(
        "--use-shared-gpu-worker",
        action="store_true",
        help=(
            "Opt-in prototype: one cross-process CUDA model scores video paths from "
            "multiple clients (serialized GPU queue; not pipelined)."
        ),
    )
    args = ap.parse_args(argv)
    if args.max_frames not in (-1, 0):
        print("pixvqa-simplevqa: --max-frames is not enforced by the upstream demo (full-clip score)",
              file=sys.stderr)
    try:
        if args.check:
            result = validate(args.repo, args.checkpoint, args.test, ffprobe=args.ffprobe)
            result["status"] = "assets_and_imports_present"
            result["note"] = "SlowFast pretrained cache and successful inference not checked."
        else:
            dev = resolve_device(args.device)
            if args.use_worker and args.use_shared_gpu_worker:
                print(
                    "pixvqa-simplevqa: --use-worker and --use-shared-gpu-worker are mutually exclusive",
                    file=sys.stderr,
                )
                return 2
            with optional_admission(
                "simplevqa", shared_gpu=args.use_shared_gpu_worker
            ):
                result = run(
                    args.repo,
                    args.checkpoint,
                    args.test,
                    device=dev,
                    ffprobe=args.ffprobe,
                    timeout=args.timeout,
                    use_worker=args.use_worker,
                    use_shared_gpu_worker=args.use_shared_gpu_worker,
                )
    except AdmissionError as exc:
        print(f"pixvqa-simplevqa: admission: {exc}", file=sys.stderr)
        return 3
    except SetupError as exc:
        print(f"pixvqa-simplevqa: setup: {exc}", file=sys.stderr)
        return 2
    except (InferenceError, OSError) as exc:
        print(f"pixvqa-simplevqa: runtime: {exc}", file=sys.stderr)
        return 1
    output_json = None if args.transient_output else (args.output_json or None)
    if not args.check and args.protocol == "lines":
        if not args.output_json:
            print("pixvqa-simplevqa: --protocol lines requires --output-json", file=sys.stderr)
            return 2
        print("\n".join(protocol_lines(result, output_json)))
        return 0
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
