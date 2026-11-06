"""Reference SimpleVQA standalone runner (not yet wired into pixvqa --metrics)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import InferenceError, SetupError, run, validate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.check:
            result = validate(args.repo, args.checkpoint, args.test, ffprobe=args.ffprobe)
            result["status"] = "assets_and_imports_present"
            result["note"] = "SlowFast pretrained cache and successful inference not checked."
        else:
            result = run(args.repo, args.checkpoint, args.test,
                         device=args.device, ffprobe=args.ffprobe, timeout=args.timeout)
    except SetupError as exc:
        print(f"pixvqa-simplevqa: setup: {exc}", file=sys.stderr)
        return 2
    except (InferenceError, OSError) as exc:
        print(f"pixvqa-simplevqa: runtime: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
