#!/usr/bin/env python3
"""Focused dispatch and validation tests for optional CVVDP."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXVQA = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
# A developer's config/learned_metrics.local.toml must not fill in options under test.
os.environ["PIXVQA_LEARNED_CONFIG"] = "none"


def run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(PIXVQA), *args], capture_output=True, text=True, env=env)


def expect_fail(name: str, args: list[str], needle: str, env: dict[str, str] | None = None) -> None:
    result = run(args, env)
    if result.returncode == 0 or needle not in result.stderr:
        raise SystemExit(f"{name}: unexpected result {result.returncode}\n{result.stdout}\n{result.stderr}")
    print(f"OK   {name}")


def test_official_csv() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from pixvqa_cvvdp.adapter import parse_result_csv

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "official.csv"
        path.write_text(
            "test, reference, cvvdp\n"
            "distorted.mkv, reference.mkv, 9.311432838439941\n",
            encoding="utf-8",
        )
        value = parse_result_csv(path)
    if value != 9.311432838439941:
        raise SystemExit(f"official-csv: expected 9.311432838439941, got {value!r}")
    print("OK   official-csv")


def defaulting_decode(fname: str) -> dict:
    """Upstream decode_video_props behavior: silent defaults, then filename tokens."""
    sys.path.insert(0, str(ROOT / "python"))
    from pixvqa_cvvdp.adapter import AdapterError, strict_decode_video_props

    props = {"width": 1920, "height": 1080, "fps": 24, "bit_depth": 8, "color_space": "709", "chroma_ss": "420"}
    try:
        props.update(strict_decode_video_props(fname))
    except AdapterError:
        return props
    return props


def test_raw_yuv() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from pixvqa_cvvdp.adapter import (
        AdapterError, encode_fps, run, strict_decode_video_props, validate_raw_yuv, yuv_metadata_name,
    )

    name = yuv_metadata_name("reference", 1920, 1080, encode_fps(60, 1))
    props = strict_decode_video_props(name)
    if props != {"width": 1920, "height": 1080, "bit_depth": 10, "chroma_ss": "420", "color_space": "2020", "fps": 60.0}:
        raise SystemExit(f"metadata: {name} -> {props}")
    if "59.94fps.yuv" not in yuv_metadata_name("test", 1920, 1080, encode_fps(60000, 1001)):
        raise SystemExit("ntsc fps was not encoded at the official 3-decimal precision")
    try:
        strict_decode_video_props("clip.yuv")
    except AdapterError as exc:
        if "missing" not in str(exc):
            raise
    else:
        raise SystemExit("incomplete YUV filename was accepted")
    defaults = defaulting_decode("clip.yuv")
    if defaults["bit_depth"] != 8 or defaults["fps"] != 24 or defaults["color_space"] != "709":
        raise SystemExit(f"defaulting decoder lost its defaults: {defaults}")
    print("OK   yuv-metadata")

    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = tmp
        try:
            frame = 16 * 16 * 3
            reference = Path(tmp) / "reference.yuv"
            reconstructed = Path(tmp) / "reconstructed.yuv"
            reference.write_bytes(b"\x00" * frame)
            reconstructed.write_bytes(b"\x00" * frame * 2)
            try:
                validate_raw_yuv(str(Path(tmp) / "missing.yuv"), 16, 16, 1)
            except AdapterError as exc:
                if "not found" not in str(exc):
                    raise
            else:
                raise SystemExit("missing raw file was accepted")
            short = Path(tmp) / "short.yuv"
            short.write_bytes(b"\x00" * (frame - 1))
            try:
                validate_raw_yuv(str(short), 16, 16, 1)
            except AdapterError as exc:
                if "exact" not in str(exc):
                    raise
            else:
                raise SystemExit("misaligned raw file was accepted")
            high = bytearray(frame)
            high[1] = 4
            high_path = Path(tmp) / "high.yuv"
            high_path.write_bytes(high)
            try:
                validate_raw_yuv(str(high_path), 16, 16, 1)
            except AdapterError as exc:
                if "10-bit" not in str(exc):
                    raise
            else:
                raise SystemExit("value above 1023 was accepted")
            try:
                validate_raw_yuv(str(reference), 16, 16, 2)
            except AdapterError as exc:
                if "fewer than requested" not in str(exc):
                    raise
            else:
                raise SystemExit("short frame count was accepted")

            script = Path(tmp) / "fake-cvvdp"
            script.write_text(
                """#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]
ref, test, result = value("--ref"), value("--test"), value("--result")
if not (os.path.islink(ref) and os.path.islink(test)):
    raise SystemExit("cvvdp inputs are not symlinks")
if os.path.basename(ref) != "reference_16x16_10b_420_2020_60fps.yuv":
    raise SystemExit(os.path.basename(ref))
if os.path.basename(test) != "test_16x16_10b_420_2020_60fps.yuv":
    raise SystemExit(os.path.basename(test))
if not os.path.samefile(ref, os.environ["RAW_REFERENCE"]) or not os.path.samefile(test, os.environ["RAW_TEST"]):
    raise SystemExit("symlink does not preserve the original file")
print("[INFO] Running on device: " + os.environ.get("FAKE_LOGGED_DEVICE", value("--device")), file=sys.stderr)
open(result, "w", encoding="utf-8").write("test, reference, cvvdp\\n%s, %s, 9.5\\n" % (test, ref))
""",
                encoding="utf-8",
            )
            script.chmod(0o755)
            os.environ["RAW_REFERENCE"] = str(reference)
            os.environ["RAW_TEST"] = str(reconstructed)
            output = Path(tmp) / "result.json"
            record = run(
                str(reference), str(reconstructed), "pq", "standard_hdr_pq", 1, "cpu", str(output),
                str(script), 16, 16, 60, 1, defaulting_decode,
            )
            command = record["command"]
            if record["jod"] != 9.5 or "--metric" not in command or command[command.index("--metric") + 1] != "cvvdp":
                raise SystemExit(f"raw dispatch record: {record}")
            if any(Path(tmp).glob("pixvqa-cvvdp-yuv-*")) or not reference.is_file() or reference.is_symlink():
                raise SystemExit("temporary symlink leaked or the original was replaced")
            print("OK   yuv-dispatch-cleanup")
            os.environ["FAKE_LOGGED_DEVICE"] = "cuda"
            try:
                run(
                    str(reference), str(reconstructed), "pq", "standard_hdr_pq", 1, "cpu",
                    str(Path(tmp) / "wrong-device.json"), str(script), 16, 16, 60, 1, defaulting_decode,
                )
            except AdapterError as exc:
                if "cvvdp reported device 'cuda', not cpu" not in str(exc):
                    raise
            else:
                raise SystemExit("a run on an unrequested device was accepted")
            finally:
                os.environ.pop("FAKE_LOGGED_DEVICE", None)
            print("OK   cvvdp-device-mismatch-rejected")
            try:
                run(
                    str(reference), str(reconstructed), "pq", "standard_hdr_pq", 1, "cpu",
                    str(Path(tmp) / "failed.json"), str(script) + "-missing", 16, 16, 60, 1,
                    defaulting_decode,
                )
            except AdapterError as exc:
                if "could not execute cvvdp" not in str(exc):
                    raise
            else:
                raise SystemExit("missing cvvdp backend was accepted")
            if any(Path(tmp).glob("pixvqa-cvvdp-yuv-*")):
                raise SystemExit("symlink directory leaked after backend failure")
            print("OK   yuv-failure-cleanup")
        finally:
            if previous is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous


def main() -> int:
    test_official_csv()
    test_raw_yuv()
    if not PIXVQA.is_file():
        print(f"missing pixvqa: {PIXVQA}", file=sys.stderr)
        return 2
    base = ["--metrics", "cvvdp", "-r", "reference.mkv", "-t", "distorted.mkv"]
    expect_fail("missing-interpretation", [*base, "--cvvdp-display", "standard_hdr_pq"],
                "--cvvdp-interpretation")
    expect_fail("invalid-interpretation", [*base, "--cvvdp-interpretation", "rec709",
                                            "--cvvdp-display", "standard_hdr_pq"],
                "expects sdr, pq, or hlg")
    # Mixed runs validate native inputs before any backend starts.
    expect_fail("mixed-metrics", ["--metrics", "psnr,cvvdp", "-r", "r", "-t", "d",
                                   "--cvvdp-interpretation", "pq", "--cvvdp-display", "standard_hdr_pq"],
                "cannot open reference")
    expect_fail("mixed-metrics-repeated-native-first", ["--metrics", "psnr", "--metrics", "cvvdp",
                                                         "-r", "r", "-t", "d",
                                                         "--cvvdp-interpretation", "pq",
                                                         "--cvvdp-display", "standard_hdr_pq"],
                "cannot open reference")
    expect_fail("mixed-metrics-repeated-cvvdp-first", ["--metrics", "cvvdp", "--metrics", "psnr",
                                                        "-r", "r", "-t", "d",
                                                        "--cvvdp-interpretation", "pq",
                                                        "--cvvdp-display", "standard_hdr_pq"],
                "cannot open reference")
    expect_fail("qcs-range-requires-qcs", [*base, "--cvvdp-interpretation", "pq",
                                           "--cvvdp-display", "standard_hdr_pq",
                                           "--qcs-range", "cvvdp=7:10"],
                "--qcs-range requires --qcs")
    expect_fail("native-option-incompatible", [*base, "--cvvdp-interpretation", "pq",
                                                "--cvvdp-display", "standard_hdr_pq", "--threads", "2"],
                "incompatible")
    expect_fail("options-without-cvvdp", ["--metrics", "psnr", "-r", "r", "-t", "d",
                                          "--cvvdp-display", "standard_hdr_pq"],
                "require --metrics cvvdp")
    expect_fail("missing-interpreter", [*base, "--cvvdp-interpretation", "pq",
                                        "--cvvdp-display", "standard_hdr_pq",
                                        "--cvvdp-python", "/no/such/python"],
                "could not execute CVVDP interpreter")
    raw = ["-r", "reference.yuv", "-t", "reconstructed.yuv"]
    expect_fail("raw-unsupported-format", ["--metrics", "cvvdp", *raw, "-s", "16x16", "-f", "420p:8",
                                           "--fps", "60", "--cvvdp-interpretation", "pq",
                                           "--cvvdp-display", "standard_hdr_pq"],
                "YUV420p10le")
    expect_fail("raw-odd-size", ["--metrics", "cvvdp", *raw, "-s", "15x16", "-f", "420p:10",
                                 "--fps", "60", "--cvvdp-interpretation", "pq",
                                 "--cvvdp-display", "standard_hdr_pq"],
                "even width")
    expect_fail("raw-missing-geometry", ["--metrics", "cvvdp", "-r", "reference.yuv", "-t", "reconstructed.yuv",
                                       "--cvvdp-interpretation", "pq", "--cvvdp-display", "standard_hdr_pq"],
                "requires -s and -f")
    expect_fail("raw-missing-fps", ["--metrics", "cvvdp", *raw, "-s", "16x16", "-f", "420p:10",
                                    "--cvvdp-interpretation", "pq",
                                    "--cvvdp-display", "standard_hdr_pq"],
                "requires --fps")
    expect_fail("raw-sdr-rejected", ["--metrics", "cvvdp", *raw, "-s", "16x16", "-f", "420p:10",
                                     "--fps", "60", "--cvvdp-interpretation", "sdr",
                                     "--cvvdp-display", "standard_fhd"],
                "SDR raw YUV is unsupported")

    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "pixvqa_cvvdp"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "__main__.py").write_text(
            """
import json, os, sys
args = sys.argv[1:]
record = args[args.index("--output-json") + 1]
if os.environ.get("MOCK_CVVDP_FAIL"):
    open(record, "w", encoding="utf-8").write("failed")
    path_file = os.environ.get("MOCK_CVVDP_RECORD_PATH_FILE")
    if path_file:
        open(path_file, "w", encoding="utf-8").write(record)
    print("mock backend failed", file=sys.stderr)
    raise SystemExit(9)
with open(record, "w", encoding="utf-8") as out:
    json.dump({"record": True}, out)
def optional(flag):
    return args[args.index(flag) + 1] if flag in args else None
print(json.dumps({"metric":"cvvdp","jod":9.5,"display":args[args.index("--display") + 1],
                  "interpretation":args[args.index("--interpretation") + 1],
                  "device":args[args.index("--device") + 1],
                  "reference":args[args.index("--ref") + 1],
                  "test":args[args.index("--test") + 1],
                  "raw_width":optional("--raw-width"),
                  "raw_height":optional("--raw-height"),
                  "raw_fps_num":optional("--raw-fps-num"),
                  "raw_fps_den":optional("--raw-fps-den"),
                  "output_json":None if "--transient-output" in args else record,
                  "backend_record":record},
                 separators=(",", ":")))
""",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = tmp + os.pathsep + env.get("PYTHONPATH", "")
        output = Path(tmp) / "result.json"
        result = run([
            "--metrics", "cvvdp", "-r", "-reference.mkv", "-t", "-distorted.mkv",
            "--frames", "7", "--metric-device", "cpu", "--cvvdp-interpretation", "pq",
            "--cvvdp-interpretation", "pq", "--cvvdp-display", "standard_hdr_pq",
            "--cvvdp-display", "standard_hdr_pq", "--cvvdp-python", sys.executable,
            "--cvvdp-output-json", str(output),
        ], env)
        if result.returncode != 0:
            raise SystemExit(f"mock-success: {result.stderr}")
        payload = json.loads(result.stdout)
        if (payload["jod"] != 9.5 or payload["device"] != "cpu" or payload["reference"] != "-reference.mkv"
                or payload["test"] != "-distorted.mkv" or not output.is_file()):
            raise SystemExit(f"mock-success: bad payload {result.stdout!r}")
        print("OK   mocked-dispatch")
        raw_dispatch = run([
            "--metrics", "cvvdp", "-r", "reference.yuv", "-t", "reconstructed.yuv",
            "-s", "1920x1080", "-f", "420p:10", "--fps", "60", "--cvvdp-interpretation", "pq",
            "--cvvdp-display", "standard_hdr_pq", "--cvvdp-python", sys.executable,
        ], env)
        if raw_dispatch.returncode != 0:
            raise SystemExit(f"raw-cli-dispatch: {raw_dispatch.stderr}")
        raw_payload = json.loads(raw_dispatch.stdout)
        if (raw_payload["raw_width"] != "1920" or raw_payload["raw_height"] != "1080"
                or raw_payload["raw_fps_num"] != "60" or raw_payload["raw_fps_den"] != "1"
                or raw_payload["reference"] != "reference.yuv" or raw_payload["test"] != "reconstructed.yuv"):
            raise SystemExit(f"raw-cli-dispatch payload: {raw_dispatch.stdout!r}")
        print("OK   raw-cli-dispatch")
        expect_fail("conflicting-repeated-display", [*base, "--cvvdp-interpretation", "pq",
                                                      "--cvvdp-display", "standard_hdr_pq",
                                                      "--cvvdp-display", "standard_fhd"],
                    "conflicting --cvvdp-display", env)
        transient = run([*base, "--cvvdp-interpretation", "pq",
                         "--cvvdp-display", "standard_hdr_pq", "--cvvdp-python", sys.executable], env)
        if transient.returncode != 0:
            raise SystemExit(f"transient-success: {transient.stderr}")
        transient_payload = json.loads(transient.stdout)
        if transient_payload["output_json"] is not None or Path(transient_payload["backend_record"]).exists():
            raise SystemExit(f"transient-success: output was retained {transient.stdout!r}")
        print("OK   temporary-output-cleanup-success")
        env["MOCK_CVVDP_FAIL"] = "1"
        record_path_file = Path(tmp) / "failed-record-path"
        env["MOCK_CVVDP_RECORD_PATH_FILE"] = str(record_path_file)
        expect_fail("backend-failure", [*base, "--cvvdp-interpretation", "pq",
                                        "--cvvdp-display", "standard_hdr_pq",
                                        "--cvvdp-python", sys.executable],
                    "CVVDP backend exited 9", env)
        if not record_path_file.is_file() or Path(record_path_file.read_text(encoding="utf-8")).exists():
            raise SystemExit("temporary output was not removed after backend failure")
        print("OK   temporary-output-cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
