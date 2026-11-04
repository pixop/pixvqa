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
    from pixvqa_cvvdp.adapter import AdapterError, parse_result_csv

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
        for bad, needle in (
            ("test, reference, cvvdp\nonly-two-cols\n", "column(s)"),
            ("test, reference, cvvdp\na, b\n", "column(s)"),
            ("test, reference\na, b, 1.0\n", "no cvvdp column"),
            ("test, reference, cvvdp\na, b, \n", "empty"),
        ):
            bad_path = Path(tmp) / "bad.csv"
            bad_path.write_text(bad, encoding="utf-8")
            try:
                parse_result_csv(bad_path)
            except AdapterError as exc:
                if needle not in str(exc):
                    raise SystemExit(f"official-csv bad case: {exc}") from exc
            else:
                raise SystemExit(f"official-csv accepted malformed CSV: {bad!r}")
    print("OK   official-csv")


def test_csv_parse_failure_record() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from pixvqa_cvvdp.adapter import AdapterError, run

    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = tmp
        try:
            frame = 16 * 16 * 3
            ref = Path(tmp) / "reference.yuv"
            test = Path(tmp) / "test.yuv"
            ref.write_bytes(b"\x00\x00" * (frame // 2))
            test.write_bytes(b"\x00\x00" * (frame // 2))
            script = Path(tmp) / "bad-csv-cvvdp"
            script.write_text(
                """#!/usr/bin/env python3
import sys
result = sys.argv[sys.argv.index("--result") + 1]
open(result, "w", encoding="utf-8").write("test, reference, cvvdp\\nonly-one-col\\n")
print("[INFO] Running on device: cpu", file=sys.stderr)
""",
                encoding="utf-8",
            )
            script.chmod(0o755)
            out = Path(tmp) / "out.json"
            try:
                run(str(ref), str(test), "pq", "standard_hdr_pq", 1, "cpu", str(out), str(script),
                    16, 16, 60, 1, defaulting_decode)
            except AdapterError as exc:
                if "column" not in str(exc):
                    raise SystemExit(f"csv-failure-record: {exc}") from exc
            else:
                raise SystemExit("csv-failure-record: malformed CSV was accepted")
            payload = json.loads(out.read_text(encoding="utf-8"))
            if payload.get("exit_code") != 0 or "stdout" not in payload or "stderr" not in payload:
                raise SystemExit(f"csv-failure-record: incomplete record {payload.keys()}")
            if "only-one-col" not in payload.get("result_csv_text", ""):
                raise SystemExit("csv-failure-record: result_csv_text missing")
        finally:
            if previous is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous
    print("OK   csv-parse-failure-record")


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

            frame_bytes = 16 * 16 * 3
            good = b"\x00\x00" * (frame_bytes // 2)
            bad_tail = b"\x00\x04" + b"\x00\x00" * (frame_bytes // 2 - 1)
            padded = Path(tmp) / "padded.yuv"
            padded.write_bytes(good + bad_tail + good)
            validate_raw_yuv(str(padded), 16, 16, 1)
            try:
                validate_raw_yuv(str(padded), 16, 16, 2)
            except AdapterError as exc:
                if "10-bit" not in str(exc):
                    raise
            else:
                raise SystemExit("invalid sample in frame 2 was accepted with nframes=2")

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


def test_mixed_input_prepare() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from pixvqa_cvvdp.adapter import encode_fps, prepare_inputs, yuv_metadata_name

    pq_meta = {
        "path": "/tmp/dist.mp4", "codec": "hevc", "profile": "Main 10", "width": 64, "height": 64,
        "pix_fmt": "yuv420p10le", "color_transfer": "smpte2084", "color_primaries": "bt2020",
        "color_space": "bt2020nc", "color_range": "tv", "nb_frames": "4", "duration": "0.16",
        "r_frame_rate": "24/1",
    }

    def fake_probe(path: str) -> dict:
        if path.endswith(".mp4"):
            return dict(pq_meta, path=path)
        raise AssertionError(f"unexpected probe {path}")

    import pixvqa_cvvdp.adapter as adapter

    real_probe = adapter.probe_video
    adapter.probe_video = fake_probe
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TMPDIR"] = tmp
            frame = 64 * 64 * 3 * 2
            raw = Path(tmp) / "ref.yuv"
            raw.write_bytes(b"\x00" * frame * 2)
            script = Path(tmp) / "fake-cvvdp"
            script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            script.chmod(0o755)
            prepared = prepare_inputs(
                str(raw), "/tmp/dist.mp4", "pq", "standard_hdr_pq", 2, str(script),
                64, 64, 24, 1, None, raw_ref=True, raw_test=False,
            )
            if not os.path.islink(prepared["ref_path"]) or prepared["test_path"] != "/tmp/dist.mp4":
                raise SystemExit(f"mixed ref-raw: {prepared['ref_path']} {prepared['test_path']}")
            name = yuv_metadata_name("reference", 64, 64, encode_fps(24, 1))
            if os.path.basename(prepared["ref_path"]) != name:
                raise SystemExit(f"bad symlink name {prepared['ref_path']}")
            prepared2 = prepare_inputs(
                "/tmp/ref.mp4", str(raw), "pq", "standard_hdr_pq", 2, str(script),
                64, 64, 24, 1, None, raw_ref=False, raw_test=True,
            )
            if prepared2["ref_path"] != "/tmp/ref.mp4" or not os.path.islink(prepared2["test_path"]):
                raise SystemExit("mixed test-raw layout wrong")
            frame_bytes = 64 * 64 * 3 * 2
            raw_ref2 = Path(tmp) / "ref2.yuv"
            raw_test2 = Path(tmp) / "test2.yuv"
            raw_ref2.write_bytes(b"\x01" * frame_bytes * 2)
            raw_test2.write_bytes(b"\x02" * frame_bytes * 2)
            prepared_both = prepare_inputs(
                str(raw_ref2), str(raw_test2), "pq", "standard_hdr_pq", 2, str(script),
                64, 64, 24, 1, None,
            )
            if not os.path.islink(prepared_both["ref_path"]) or not os.path.islink(prepared_both["test_path"]):
                raise SystemExit("both-raw layout wrong")
            prepared_sdr_both = prepare_inputs(
                str(raw_ref2), str(raw_test2), "sdr", "standard_fhd", 2, str(script),
                64, 64, 24, 1, None, raw_ref=True, raw_test=True,
            )
            if not (os.path.islink(prepared_sdr_both["ref_path"])
                    and os.path.islink(prepared_sdr_both["test_path"])):
                raise SystemExit("SDR raw/raw symlinks missing")
            if not prepared_sdr_both["ref_path"].endswith("_709_24fps.yuv"):
                raise SystemExit("SDR raw/raw reference should be BT.709")
            prepared_cc = prepare_inputs(
                "/tmp/ref.mp4", "/tmp/dist.mp4", "pq", "standard_hdr_pq", 2, str(script),
                None, None, None, None, None, raw_ref=False, raw_test=False,
            )
            if prepared_cc["ref_path"] != "/tmp/ref.mp4" or prepared_cc["test_path"] != "/tmp/dist.mp4":
                raise SystemExit("both-compressed should pass paths through")
            prepared["links"].cleanup()
            prepared2["links"].cleanup()
            prepared_both["links"].cleanup()
            prepared_sdr_both["links"].cleanup()
    finally:
        adapter.probe_video = real_probe
        os.environ.pop("TMPDIR", None)
    print("OK   cvvdp-four-input-prepare")


def test_mixed_routing() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from unittest.mock import patch

    import pixvqa_cvvdp.adapter as adapter
    from pixvqa_cvvdp.adapter import uses_python_video_source

    if not uses_python_video_source({"reference_is_raw": True, "test_is_raw": False}, "sdr"):
        raise SystemExit("python source ref-raw")
    if not uses_python_video_source({"reference_is_raw": True, "test_is_raw": True}, "sdr"):
        raise SystemExit("python source SDR raw/raw")
    if uses_python_video_source({"reference_is_raw": True, "test_is_raw": True}, "pq"):
        raise SystemExit("PQ raw/raw must use upstream source")
    if uses_python_video_source(None, "sdr"):
        raise SystemExit("python source without raw metadata")

    prepared = {
        "binary": "/bin/false",
        "test_path": "/tmp/dist.mp4",
        "ref_path": "/tmp/ref.yuv",
        "ref_meta": {},
        "test_meta": {},
        "display_model": {},
        "raw_record": {"reference_is_raw": True, "test_is_raw": False, "fps_requested": 60.0},
        "links": None,
    }
    mixed_calls: list[dict] = []

    def fake_mixed(*_args, **_kwargs):
        mixed_calls.append({"ok": True})
        return {
            "jod": 7.5,
            "performance": {},
            "stdout": "",
            "stderr": "",
        }

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        with patch("pixvqa_cvvdp.python_api.run_mixed_predict", side_effect=fake_mixed):
            with patch.object(adapter.subprocess, "run", side_effect=AssertionError("CLI must not run")):
                record = adapter.execute(
                    prepared, "pq", "standard_hdr_pq", 2, "cpu", str(out), temporal=False,
                )
        if not mixed_calls or record.get("jod") != 7.5:
            raise SystemExit(f"mixed routing failed: {mixed_calls} {record}")
        if not out.is_file():
            raise SystemExit("mixed routing did not write JSON")
    print("OK   cvvdp-mixed-python-routing")


def test_sdr_metadata_fallback() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from pixvqa_cvvdp.adapter import AdapterError, check_pair

    def meta(transfer="unknown", primaries="unknown", matrix="bt709", range_="tv"):
        return {"width": 16, "height": 16, "r_frame_rate": "30/1", "nb_frames": "3",
                "color_space": matrix, "color_transfer": transfer,
                "color_primaries": primaries, "color_range": range_}

    ref, test = meta("bt709", "bt709"), meta()
    check_pair(ref, test, "sdr", 0)
    if (test["color_transfer"], test["color_primaries"]) != ("bt709", "bt709"):
        raise SystemExit("SDR fallback did not resolve unspecified tags")
    if set(test.get("inferred_sdr_metadata", {})) != {"color_transfer", "color_primaries"}:
        raise SystemExit("SDR fallback missing provenance")
    if test["original_color_metadata"]["color_transfer"] != "unknown":
        raise SystemExit("SDR fallback overwrote original color metadata")
    # A partially tagged stream is resolved without overwriting an explicit tag.
    partial = meta("bt709", "unspecified")
    check_pair(meta("bt709", "bt709"), partial, "sdr", 0)
    if partial.get("inferred_sdr_metadata") != {"color_primaries": "bt709"}:
        raise SystemExit("SDR partial fallback failed")
    for name, invalid in (
        ("explicit PQ transfer", meta("smpte2084", "unknown")),
        ("conflicting primaries", meta("bt709", "bt2020")),
        ("unknown matrix", meta(matrix="unknown")),
        ("HDR matrix", meta(matrix="bt2020nc")),
        ("full range", meta(range_="pc")),
    ):
        try:
            check_pair(meta("bt709", "bt709"), invalid, "sdr", 0)
        except AdapterError:
            pass
        else:
            raise SystemExit(f"SDR accepted {name}")
    # PQ and HLG retain strict transfer/primaries checks.
    for mode in ("pq", "hlg"):
        try:
            check_pair(meta("bt709", "bt709"), meta(), mode, 0)
        except AdapterError:
            pass
        else:
            raise SystemExit(f"{mode} accepted unspecified transfer")
    print("OK   sdr-bt709-metadata-fallback")


def main() -> int:
    test_sdr_metadata_fallback()
    test_official_csv()
    test_csv_parse_failure_record()
    test_raw_yuv()
    test_mixed_input_prepare()
    test_mixed_routing()

    if not PIXVQA.is_file():
        print(f"missing pixvqa: {PIXVQA}", file=sys.stderr)
        return 2

    # All input fixtures are local to the test run; nothing depends on a
    # workstation-specific dataset or an FFmpeg encoder installation.
    with tempfile.TemporaryDirectory(prefix="pixvqa_cvvdp_cli_") as tmp:
        fixture_dir = Path(tmp)
        raw_ref = fixture_dir / "reference.yuv"
        raw_dist = fixture_dir / "reconstructed.yuv"
        # One 16x16 YUV420p10le frame = 16*16*3 bytes (16-bit samples).
        sample_frame = bytes(16 * 16 * 3)
        raw_ref.write_bytes(sample_frame)
        raw_dist.write_bytes(sample_frame)
        raw = ["-r", str(raw_ref), "-t", str(raw_dist)]
        raw_options = ["-s", "16x16", "-f", "420p:10", "--fps", "60"]
        pq_options = ["--cvvdp-interpretation", "pq", "--cvvdp-display", "standard_hdr_pq"]
        sdr_options = ["--cvvdp-interpretation", "sdr", "--cvvdp-display", "standard_fhd"]
        missing_compressed = ["--metrics", "cvvdp", "-r", "reference.mkv", "-t", "distorted.mkv"]

        # Nonexistent names are intentional in these validation-order tests.
        expect_fail(
            "missing-input-before-interpretation",
            [*missing_compressed, "--cvvdp-display", "standard_hdr_pq"],
            "cannot open test file",
        )
        expect_fail(
            "invalid-interpretation",
            [*missing_compressed, "--cvvdp-interpretation", "rec709", "--cvvdp-display", "standard_hdr_pq"],
            "expects sdr, pq, or hlg",
        )

        # Mixed runs validate native inputs before starting the learned backend.
        mixed = ["-r", "r", "-t", "d", *pq_options]
        expect_fail(
            "mixed-metrics",
            ["--metrics", "psnr,cvvdp", *mixed],
            "cannot open reference",
        )
        expect_fail(
            "mixed-metrics-repeated-native-first",
            ["--metrics", "psnr", "--metrics", "cvvdp", *mixed],
            "cannot open reference",
        )
        expect_fail(
            "mixed-metrics-repeated-cvvdp-first",
            ["--metrics", "cvvdp", "--metrics", "psnr", *mixed],
            "cannot open reference",
        )
        expect_fail(
            "qcs-range-requires-qcs",
            [*missing_compressed, *pq_options, "--qcs-range", "cvvdp=7:10"],
            "--qcs-range requires --qcs",
        )
        expect_fail(
            "native-option-incompatible",
            [*missing_compressed, *pq_options, "--threads", "2"],
            "incompatible",
        )
        expect_fail(
            "options-without-cvvdp",
            ["--metrics", "psnr", "-r", "r", "-t", "d", "--cvvdp-display", "standard_hdr_pq"],
            "require --metrics cvvdp",
        )

        # Exercise interpreter validation using generated, valid raw files.
        # No hardcoded /mnt/d/... video or external encoder is required.
        expect_fail(
            "missing-interpreter",
            [
                "--metrics", "cvvdp", *raw, *raw_options, *sdr_options,
                "--cvvdp-python", str(fixture_dir / "no-such-python"),
            ],
            "could not execute CVVDP interpreter",
        )

        expect_fail(
            "raw-unsupported-format",
            ["--metrics", "cvvdp", *raw, "-s", "16x16", "-f", "420p:8", "--fps", "60", *pq_options],
            "YUV420p10le",
        )
        expect_fail(
            "raw-odd-size",
            ["--metrics", "cvvdp", *raw, "-s", "15x16", "-f", "420p:10", "--fps", "60", *pq_options],
            "even width",
        )
        expect_fail(
            "raw-missing-geometry",
            ["--metrics", "cvvdp", *raw, *pq_options],
            "requires -s and -f",
        )
        expect_fail(
            "raw-missing-fps",
            ["--metrics", "cvvdp", *raw, "-s", "16x16", "-f", "420p:10", *pq_options],
            "requires --fps",
        )

        # Mock only the learned backend process. The compiled C++ frontend and
        # its raw/raw argument dispatch still run for real.
        mock_dir = fixture_dir / "mock"
        package = mock_dir / "pixvqa_cvvdp"
        package.mkdir(parents=True)
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
        env["PYTHONPATH"] = str(mock_dir) + os.pathsep + env.get("PYTHONPATH", "")
        output = fixture_dir / "result.json"

        # These are successful frontend/backend-dispatch tests. Use real, tiny,
        # generated raw inputs rather than nonexistent compressed placeholders.
        # The mock module avoids pycvvdp inference and any dependence on FFmpeg.
        result = run(
            [
                "--metrics", "cvvdp", *raw, *raw_options,
                "--frames", "1", "--metric-device", "cpu",
                "--cvvdp-interpretation", "sdr", "--cvvdp-interpretation", "sdr",
                "--cvvdp-display", "standard_fhd", "--cvvdp-display", "standard_fhd",
                "--cvvdp-python", sys.executable, "--cvvdp-output-json", str(output),
            ],
            env,
        )
        if result.returncode != 0:
            raise SystemExit(f"mock-success: {result.stderr}")
        payload = json.loads(result.stdout)
        if (
            payload["jod"] != 9.5
            or payload["device"] != "cpu"
            or payload["reference"] != str(raw_ref)
            or payload["test"] != str(raw_dist)
            or not output.is_file()
        ):
            raise SystemExit(f"mock-success: bad payload {result.stdout!r}")
        print("OK   mocked-dispatch")

        raw_dispatch = run(
            [
                "--metrics", "cvvdp", *raw, *raw_options, *pq_options,
                "--cvvdp-python", sys.executable,
            ],
            env,
        )
        if raw_dispatch.returncode != 0:
            raise SystemExit(f"raw-cli-dispatch: {raw_dispatch.stderr}")
        raw_payload = json.loads(raw_dispatch.stdout)
        if (
            raw_payload["raw_width"] != "16"
            or raw_payload["raw_height"] != "16"
            or raw_payload["raw_fps_num"] != "60"
            or raw_payload["raw_fps_den"] != "1"
            or raw_payload["reference"] != str(raw_ref)
            or raw_payload["test"] != str(raw_dist)
        ):
            raise SystemExit(f"raw-cli-dispatch payload: {raw_dispatch.stdout!r}")
        print("OK   raw-cli-dispatch")

        # Positive SDR raw/raw dispatch test: backend import is mocked, but
        # both input files exist and frontend validation must accept them.
        raw_sdr = run(
            ["--metrics", "cvvdp", *raw, *raw_options, *sdr_options, "--cvvdp-python", sys.executable],
            env,
        )
        if raw_sdr.returncode != 0:
            raise SystemExit(f"raw-sdr-both-dispatched: {raw_sdr.stderr}")
        sdr_payload = json.loads(raw_sdr.stdout)
        if (
            sdr_payload.get("jod") != 9.5
            or sdr_payload.get("interpretation") != "sdr"
            or sdr_payload.get("reference") != str(raw_ref)
            or sdr_payload.get("test") != str(raw_dist)
            or sdr_payload.get("raw_width") != "16"
            or sdr_payload.get("raw_height") != "16"
        ):
            raise SystemExit(f"raw-sdr-both-dispatched: bad payload {raw_sdr.stdout!r}")
        print("OK   raw-sdr-both-dispatched")

        expect_fail(
            "conflicting-repeated-display",
            [*missing_compressed, *pq_options, "--cvvdp-display", "standard_fhd"],
            "conflicting --cvvdp-display",
            env,
        )
        transient = run(
            ["--metrics", "cvvdp", *raw, *raw_options, *sdr_options,
             "--cvvdp-python", sys.executable],
            env,
        )
        if transient.returncode != 0:
            raise SystemExit(f"transient-success: {transient.stderr}")
        transient_payload = json.loads(transient.stdout)
        if transient_payload["output_json"] is not None or Path(transient_payload["backend_record"]).exists():
            raise SystemExit(f"transient-success: output was retained {transient.stdout!r}")
        print("OK   temporary-output-cleanup-success")

        env["MOCK_CVVDP_FAIL"] = "1"
        record_path_file = fixture_dir / "failed-record-path"
        env["MOCK_CVVDP_RECORD_PATH_FILE"] = str(record_path_file)
        expect_fail(
            "backend-failure",
            ["--metrics", "cvvdp", *raw, *raw_options, *sdr_options,
             "--cvvdp-python", sys.executable],
            "CVVDP backend exited 9",
            env,
        )
        if not record_path_file.is_file() or Path(record_path_file.read_text(encoding="utf-8")).exists():
            raise SystemExit("temporary output was not removed after backend failure")
        print("OK   temporary-output-cleanup")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
