"""Command-line entry point for the optional ColorVideoVDP adapter."""

import argparse
import json
import sys

from pixvqa_cvvdp.adapter import AdapterError, run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a video pair with the official cvvdp CLI.")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--interpretation", required=True, choices=["sdr", "pq"])
    parser.add_argument("--display", required=True)
    parser.add_argument("--nframes", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--cvvdp-bin")
    parser.add_argument("--transient-output", action="store_true",
                        help="Internal: the caller will remove the output record after this run.")
    args = parser.parse_args(argv)
    try:
        record = run(
            args.ref, args.test, args.interpretation, args.display, args.nframes, args.device,
            args.output_json, args.cvvdp_bin,
        )
    except AdapterError as exc:
        print(f"pixvqa-cvvdp: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "metric": "cvvdp", "jod": record["jod"], "display": record["display"],
        "interpretation": record["interpretation"], "device": record["device"],
        "elapsed_seconds": record["elapsed_seconds"],
        "output_json": None if args.transient_output else args.output_json,
        "runtime": record["runtime"],
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
