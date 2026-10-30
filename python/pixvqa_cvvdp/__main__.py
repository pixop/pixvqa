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
    parser.add_argument("--raw-width", type=int)
    parser.add_argument("--raw-height", type=int)
    parser.add_argument("--raw-fps-num", type=int)
    parser.add_argument("--raw-fps-den", type=int)
    parser.add_argument("--transient-output", action="store_true",
                        help="Internal: the caller will remove the output record after this run.")
    args = parser.parse_args(argv)
    try:
        record = run(
            args.ref, args.test, args.interpretation, args.display, args.nframes, args.device,
            args.output_json, args.cvvdp_bin, args.raw_width, args.raw_height, args.raw_fps_num,
            args.raw_fps_den,
        )
    except AdapterError as exc:
        print(f"pixvqa-cvvdp: {exc}", file=sys.stderr)
        return 1
    payload = {
        "metric": "cvvdp", "jod": record["jod"], "display": record["display"],
        "interpretation": record["interpretation"], "device": record["device"],
        "elapsed_seconds": record["elapsed_seconds"],
        "output_json": None if args.transient_output else args.output_json,
        "runtime": record["runtime"],
    }
    if "raw_yuv" in record:
        payload["raw_yuv"] = {
            "width": record["raw_yuv"]["width"], "height": record["raw_yuv"]["height"],
            "fps_encoded": record["raw_yuv"]["fps_encoded"], "pix_fmt": record["raw_yuv"]["pix_fmt"],
            "range": record["raw_yuv"]["range"], "matrix": record["raw_yuv"]["matrix"],
        }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
