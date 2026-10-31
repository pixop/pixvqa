"""Command-line entry point for the optional ColorVideoVDP adapter.

Exit status: 0 success, 2 preflight failure (nothing was scored), 1 runtime failure.
"""

import argparse
import contextlib
import json
import os
import sys

from pixvqa_cvvdp.adapter import AdapterError, PreflightError, run


@contextlib.contextmanager
def stdout_to_stderr():
    """Library chatter, including native writes to fd 1, must not reach the report stream."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def protocol_lines(record: dict, output_json) -> list[str]:
    frames = record.get("frames_evaluated")
    lines = [
        "pixvqa-protocol 1",
        "pixvqa-value metric cvvdp",
        f"pixvqa-value jod {record['jod']!r}",
        f"pixvqa-value frames_evaluated {'unknown' if frames is None else frames}",
        f"pixvqa-value display {record['display']}",
        f"pixvqa-value interpretation {record['interpretation']}",
        f"pixvqa-value device {record['device']}",
        f"pixvqa-value device_requested {record.get('device_requested', record['device'])}",
    ]
    temporal = record.get("temporal")
    if temporal is not None:
        lines += [
            f"pixvqa-value temporal_frames {temporal['frame_count']}",
            f"pixvqa-value pooled_parity {1 if temporal['pooled_parity'] else 0}",
            f"pixvqa-value qcs_eligible {1 if temporal['qcs_eligible'] else 0}",
            f"pixvqa-value temporal_basis {temporal['basis']}",
        ]
        for i, sample in enumerate(temporal["samples"]):
            t = sample["time_seconds"]
            lines.append(f"pixvqa-sample {i} {sample['frame_index']} {0.0 if t is None else t!r} "
                         f"{sample['distortion']!r} {sample['jod_equivalent']!r}")
    lines += [f"pixvqa-value output_json {output_json if output_json else 'none'}", "pixvqa-end"]
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a video pair with the official cvvdp CLI.")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--interpretation", required=True, choices=["sdr", "pq", "hlg"])
    parser.add_argument("--display", required=True)
    parser.add_argument("--nframes", type=int, default=-1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto picks CUDA when visible, else CPU; cuda never falls back.")
    parser.add_argument("--gpu-mem", type=float,
                        help="Optional CVVDP CUDA memory cap in decimal GB; omitted uses upstream default.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--cvvdp-bin")
    parser.add_argument("--raw-width", type=int)
    parser.add_argument("--raw-height", type=int)
    parser.add_argument("--raw-fps-num", type=int)
    parser.add_argument("--raw-fps-den", type=int)
    parser.add_argument("--transient-output", action="store_true",
                        help="Internal: the caller will remove the output record after this run.")
    parser.add_argument("--check", action="store_true",
                        help="Validate inputs, display, and the cvvdp executable, then exit.")
    parser.add_argument("--protocol", choices=["json", "lines"], default="json",
                        help="stdout format. json is the standalone summary.")
    parser.add_argument("--temporal", action="store_true",
                        help="Also derive the per-frame JOD-equivalent sequence from the official "
                             "--features output (used for QCS).")
    args = parser.parse_args(argv)
    try:
        with stdout_to_stderr():
            record = run(
                args.ref, args.test, args.interpretation, args.display, args.nframes, args.device,
                args.output_json, args.cvvdp_bin, args.raw_width, args.raw_height, args.raw_fps_num,
                args.raw_fps_den, check_only=args.check, temporal=args.temporal, gpu_mem_gb=args.gpu_mem,
            )
    except PreflightError as exc:
        print(f"pixvqa-cvvdp: {exc}", file=sys.stderr)
        return 2
    except AdapterError as exc:
        print(f"pixvqa-cvvdp: {exc}", file=sys.stderr)
        return 1
    if args.check:
        print("pixvqa-check ok")
        return 0
    output_json = None if args.transient_output else args.output_json
    if args.protocol == "lines":
        print("\n".join(protocol_lines(record, output_json)))
        return 0
    payload = {
        "metric": "cvvdp", "jod": record["jod"], "display": record["display"],
        "interpretation": record["interpretation"], "device": record["device"],
        "device_requested": record.get("device_requested", record["device"]),
        "gpu_mem_gb": record.get("gpu_mem_gb"),
        "elapsed_seconds": record["elapsed_seconds"],
        "output_json": output_json,
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
