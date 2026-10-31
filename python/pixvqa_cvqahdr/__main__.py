"""Command-line entry point for the optional CompressedVQA-HDR NR adapter.

Exit status: 0 success, 2 preflight failure (nothing was scored), 1 runtime failure.
"""

import argparse
import contextlib
import json
import os
import sys

from pixvqa_cvqahdr.adapter import AdapterError, PreflightError, preflight, run


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
    sampling = record["sampling"]
    temporal = record["temporal"]

    def known(value):
        return "unknown" if value is None else repr(value)

    lines = [
        "pixvqa-protocol 1",
        "pixvqa-value metric cvqa-hdr-nr",
        f"pixvqa-value quality_score {record['quality_score']!r}",
        f"pixvqa-value raw_model_score {record['raw_model_score']!r}",
        f"pixvqa-value sampled_frames {sampling['sampled_frames']}",
        f"pixvqa-value opencv_frame_count {sampling['opencv_frame_count']}",
        f"pixvqa-value opencv_fps {sampling['opencv_fps']!r}",
        f"pixvqa-value rounded_fps {sampling['rounded_fps']}",
        f"pixvqa-value sampling_density_hz {temporal['sampling_density_hz']!r}",
        f"pixvqa-value decoded_frame_count {known(temporal['decoded_frame_count'])}",
        f"pixvqa-value padded_samples {known(temporal['padded_samples'])}",
        f"pixvqa-value pooled_parity {1 if temporal['pooled_parity'] else 0}",
        f"pixvqa-value qcs_eligible {1 if temporal['qcs_eligible'] else 0}",
        f"pixvqa-value device {record['device']}",
        f"pixvqa-value device_requested {record.get('device_requested', record['device'])}",
        f"pixvqa-value output_json {output_json if output_json else 'none'}",
    ]
    for sample in temporal["samples"]:
        lines.append(
            f"pixvqa-sample {sample['index']} {sample['frame_index']} {sample['time_seconds']!r} "
            f"{sample['raw_model_score']!r} {sample['quality_score']!r}"
        )
    lines.append("pixvqa-end")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a compressed video with the official CompressedVQA-HDR NR model."
    )
    parser.add_argument("--test", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto picks CUDA when visible, else CPU; cuda never falls back.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--transient-output",
        action="store_true",
        help="Internal: the caller will remove the output record after this run.",
    )
    parser.add_argument("--check", action="store_true",
                        help="Validate imports, assets, and the video sampler, then exit.")
    parser.add_argument("--protocol", choices=["json", "lines"], default="json",
                        help="stdout format. json is the standalone summary.")
    parser.add_argument("--verify-decoded-frames", action="store_true",
                        help="Count decoded frames so padded samples are reported.")
    args = parser.parse_args(argv)
    try:
        with stdout_to_stderr():
            if args.check:
                preflight(args.test, args.repo, args.checkpoint, args.profile, args.device)
            else:
                record = run(
                    args.test,
                    args.repo,
                    args.checkpoint,
                    args.profile,
                    args.device,
                    args.output_json,
                    args.verify_decoded_frames,
                )
    except PreflightError as exc:
        print(f"pixvqa-cvqahdr: {exc}", file=sys.stderr)
        return 2
    except AdapterError as exc:
        print(f"pixvqa-cvqahdr: {exc}", file=sys.stderr)
        return 1
    if args.check:
        print("pixvqa-check ok")
        return 0
    output_json = None if args.transient_output else args.output_json
    if args.protocol == "lines":
        print("\n".join(protocol_lines(record, output_json)))
        return 0
    print(
        json.dumps(
            {
                "metric": record["metric"],
                "quality_score": record["quality_score"],
                "raw_model_score": record["raw_model_score"],
                "sampled_frames": record["sampling"]["sampled_frames"],
                "device": record["device"],
                "device_requested": record.get("device_requested", record["device"]),
                "output_json": output_json,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
