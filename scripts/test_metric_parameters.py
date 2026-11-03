#!/usr/bin/env python3
"""SDR/PQ/HLG metric parameter resolution, provenance, and CLI conformance suite.

Runs positive CVVDP/HDR tests, declarative negative CLI matrix, and FFmpeg build checks.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXVQA = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
DECODE = Path(os.environ.get("PIXVQA_DECODE_BIN", ROOT / "build" / "swdec" / "pixvqa"))
FFMPEG = os.environ.get("PIXVQA_TEST_FFMPEG") or shutil.which("ffmpeg")
sys.path.insert(0, str(ROOT / "python"))


def fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}", file=sys.stderr)
    raise SystemExit(1)


def ok(name: str) -> None:
    print(f"OK   {name}")


def write_yuv(path: Path, frames: int = 2, size: int = 64) -> None:
    path.write_bytes(b"\x00\x00" * (size * size * 3 // 2 * frames))


def make_pq_mp4(path: Path) -> None:
    assert FFMPEG
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
         "-frames:v", "2", "-vf",
         "setparams=color_trc=smpte2084:color_primaries=bt2020:colorspace=bt2020nc:range=tv",
         "-c:v", "libx264", "-qp", "28", "-pix_fmt", "yuv420p10le",
         "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
         "-color_range", "tv", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        fail("fixtures", proc.stderr[-300:])


def test_resolver_unit() -> None:
    # Documented defaults (see include/pixvqa/metric_resolve.cpp and ColorVideoVDP registry).
    defaults = {"pq": "standard_hdr_pq", "hlg": "standard_hdr_hlg", "sdr": "standard_fhd"}
    for family, display in defaults.items():
        if not display.startswith("standard_"):
            fail("display-default", family)
    ok("resolver-defaults-documented")


def assert_cvvdp_json(stdout: str, *, interpretation: str, display: str) -> None:
    try:
        payload = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        fail("cvvdp-json", f"{exc}\n{stdout[-400:]}")
    if payload.get("metric") != "cvvdp":
        fail("cvvdp-json-metric", stdout[-400:])
    if payload.get("interpretation") != interpretation:
        fail("cvvdp-json-interpretation", stdout[-400:])
    if payload.get("display") != display:
        fail("cvvdp-json-display", stdout[-400:])


def assert_native_hdr_inferred(stdout: str) -> None:
    if "transfer=pq (override)" in stdout:
        fail("native-hdr-not-override", stdout[-800:])
    if "reference: transfer=pq (inferred)" not in stdout:
        fail("native-hdr-inferred-ref", stdout[-800:])
    if "distorted: transfer=pq (stream)" not in stdout:
        fail("native-hdr-inferred-dist", stdout[-800:])


def assert_input_summary_legacy_defaults(stdout: str) -> None:
    if "legacy --cs/--tf defaults" not in stdout:
        fail("input-summary-legacy-defaults", stdout[-500:])


def test_cli_inference() -> None:
    if not DECODE.is_file() or not FFMPEG:
        print("SKIP cli inference (need swdec pixvqa and ffmpeg)")
        return
    if DECODE.stat().st_mtime < (ROOT / "src" / "main.cpp").stat().st_mtime:
        print("SKIP cli inference (rebuild PIXVQA_DECODE_BIN after main.cpp changes)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = tmp_path / "ref.yuv"
        pq = tmp_path / "test.mp4"
        write_yuv(raw)
        make_pq_mp4(pq)
        env = cvvdp_mock_env(tmp_path)
        base = [str(DECODE), "--metrics", "cvvdp", "-r", str(raw), "-t", str(pq),
                "-s", "64x64", "-f", "420p:10", "--fps", "24/1",
                "--cvvdp-python", sys.executable]
        proc = subprocess.run(base, capture_output=True, text=True, env=env, timeout=60)
        if proc.returncode != 0:
            fail("cvvdp-inferred-pq", f"rc={proc.returncode}\n{proc.stderr}\n{proc.stdout}")
        if "info: cvvdp interpretation pq" not in proc.stderr:
            fail("cvvdp-inferred-pq-log", proc.stderr[-500:])
        if "info: cvvdp display standard_hdr_pq" not in proc.stderr:
            fail("cvvdp-inferred-display", proc.stderr[-500:])
        assert_cvvdp_json(proc.stdout, interpretation="pq", display="standard_hdr_pq")
        ok("cvvdp-inferred-pq")

        override = subprocess.run([*base, "--hdr-transfer", "hlg"], capture_output=True, text=True, env=env,
                                    timeout=60)
        if override.returncode != 0 or "info: cvvdp interpretation hlg" not in override.stderr:
            fail("cvvdp-hdr-transfer-override", override.stderr[-400:])
        if "info: cvvdp display standard_hdr_hlg" not in override.stderr:
            fail("cvvdp-hdr-transfer-display", override.stderr[-400:])
        assert_cvvdp_json(override.stdout, interpretation="hlg", display="standard_hdr_hlg")
        ok("cvvdp-hdr-transfer-overrides-metadata")

        tf_override = subprocess.run([*base, "--tf", "stdb67"], capture_output=True, text=True, env=env,
                                     timeout=60)
        if tf_override.returncode != 0 or "info: cvvdp interpretation hlg" not in tf_override.stderr:
            fail("cvvdp-tf-override", tf_override.stderr[-400:])
        if "info: cvvdp display standard_hdr_hlg" not in tf_override.stderr:
            fail("cvvdp-tf-override-display", tf_override.stderr[-400:])
        ok("cvvdp-tf-overrides-metadata")

        per_side = subprocess.run(
            [*base, "--dist-hdr-transfer", "hlg"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        if per_side.returncode != 0 or "info: cvvdp interpretation hlg" not in per_side.stderr:
            fail("cvvdp-dist-hdr-transfer-override", per_side.stderr[-400:])
        if "info: cvvdp display standard_hdr_hlg" not in per_side.stderr:
            fail("cvvdp-dist-hdr-transfer-display", per_side.stderr[-400:])
        ok("cvvdp-dist-hdr-transfer-overrides-metadata")

        per_side_conflict = subprocess.run(
            [*base, "--ref-hdr-transfer", "sdr", "--dist-hdr-transfer", "hlg"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        if per_side_conflict.returncode == 0 or "disagree" not in per_side_conflict.stderr:
            fail("cvvdp-per-side-transfer-conflict", per_side_conflict.stderr[-400:])
        ok("cvvdp-rejects-conflicting-per-side-transfers")

        conflict = subprocess.run(
            [*base, "--cvvdp-interpretation", "pq", "--hdr-transfer", "hlg"],
            capture_output=True, text=True, env=env, timeout=60,
        )
        if conflict.returncode == 0 or "conflicts" not in conflict.stderr:
            fail("cvvdp-explicit-conflict", conflict.stderr[-400:])
        ok("cvvdp-rejects-explicit-interpretation-transfer-conflict")

        native_reject = subprocess.run(
            [*base, "--hdr-primaries", "bt2020"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        if native_reject.returncode != 2 or "applies only to native metrics" not in native_reject.stderr:
            fail("cvvdp-rejects-hdr-primaries", native_reject.stderr[-400:])
        ok("cvvdp-rejects-native-only-hdr-primaries")

        native_base = [
            str(DECODE),
            "--metrics",
            "psnr,cvvdp",
            "-r",
            str(raw),
            "-t",
            str(pq),
            "-s",
            "64x64",
            "-f",
            "420p:10",
            "--fps",
            "24/1",
            "-n",
            "1",
            "--cvvdp-python",
            sys.executable,
        ]
        native_inferred = subprocess.run(
            native_base,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        if native_inferred.returncode != 0:
            fail(
                "native-cvvdp-inferred-pq",
                f"rc={native_inferred.returncode}\n{native_inferred.stderr}\n{native_inferred.stdout}",
            )
        if "info: cvvdp interpretation pq" not in native_inferred.stderr:
            fail("native-cvvdp-inferred-pq-log", native_inferred.stderr[-500:])
        assert_native_hdr_inferred(native_inferred.stdout)
        assert_input_summary_legacy_defaults(native_inferred.stdout)
        if "[ CVVDP jod:" not in native_inferred.stdout or "interpretation:pq" not in native_inferred.stdout:
            fail("native-cvvdp-inferred-summary", native_inferred.stdout[-400:])
        ok("native-cvvdp-inferred-pq-hdr-provenance")

        native_override = subprocess.run(
            [*native_base, "--hdr-transfer", "hlg"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        if native_override.returncode != 0 or "info: cvvdp interpretation hlg" not in native_override.stderr:
            fail("native-cvvdp-hdr-transfer", native_override.stderr[-500:] + native_override.stdout[-500:])
        if "info: cvvdp display standard_hdr_hlg" not in native_override.stderr:
            fail("native-cvvdp-hdr-transfer-display", native_override.stderr[-400:])
        if "transfer=hlg (override)" not in native_override.stdout:
            fail("native-cvvdp-hdr-transfer-override", native_override.stdout[-800:])
        if "[ CVVDP jod:" not in native_override.stdout or "interpretation:hlg" not in native_override.stdout:
            fail("native-cvvdp-hdr-transfer-summary", native_override.stdout[-400:])
        if "display:standard_hdr_hlg" not in native_override.stdout:
            fail("native-cvvdp-hdr-transfer-summary-display", native_override.stdout[-400:])
        ok("native-cvvdp-explicit-hdr-transfer-override")
        ok("mixed-native-cvvdp-accepts-hdr-transfer")


def cvvdp_mock_main_py() -> str:
    return '''import json
import sys

args = sys.argv[1:]

def value(flag):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None

if "--check" in args:
    print("pixvqa-check ok")
    raise SystemExit(0)

interpretation = value("--interpretation") or "pq"
display = value("--display") or "standard_hdr_pq"
device = value("--device") or "auto"
jod = 1.0
transient = "--transient-output" in args
out_json = value("--output-json")
output_json = None if transient else out_json

if value("--protocol") == "lines":
    lines = [
        "pixvqa-protocol 1",
        "pixvqa-value metric cvvdp",
        f"pixvqa-value jod {jod!r}",
        "pixvqa-value frames_evaluated 2",
        f"pixvqa-value display {display}",
        f"pixvqa-value interpretation {interpretation}",
        f"pixvqa-value device {device}",
        f"pixvqa-value output_json {output_json if output_json else 'none'}",
        "pixvqa-end",
    ]
    print("\\n".join(lines))
    raise SystemExit(0)

payload = {
    "metric": "cvvdp",
    "jod": jod,
    "display": display,
    "interpretation": interpretation,
    "device": device,
    "output_json": output_json,
}
print(json.dumps(payload, separators=(",", ":")))
'''


def cvvdp_mock_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PIXVQA_LEARNED_CONFIG"] = "none"
    env["PYTHONPATH"] = str(tmp_path / "mock")
    mock = tmp_path / "mock" / "pixvqa_cvvdp"
    mock.mkdir(parents=True)
    (mock / "__init__.py").write_text("", encoding="utf-8")
    (mock / "__main__.py").write_text(cvvdp_mock_main_py(), encoding="utf-8")
    return env


def test_learned_native_guard() -> None:
    if not DECODE.is_file():
        print("SKIP learned native guard (need swdec pixvqa)")
        return
    if DECODE.stat().st_mtime < (ROOT / "src" / "main.cpp").stat().st_mtime:
        print("SKIP learned native guard (rebuild PIXVQA_DECODE_BIN after main.cpp changes)")
        return
    env = os.environ.copy()
    env["PIXVQA_LEARNED_CONFIG"] = "none"
    cvqa = subprocess.run(
        [str(DECODE), "--metrics", "cvqa-hdr-nr", "-t", "dist.mp4", "--hdr-primaries", "bt2020"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if cvqa.returncode != 2 or "applies only to native metrics" not in cvqa.stderr:
        fail("cvqa-rejects-hdr-primaries", cvqa.stderr[-400:])
    ok("cvqa-rejects-native-only-hdr-primaries")

    transfer_ok = subprocess.run(
        [str(DECODE), "--metrics", "cvvdp", "-r", "ref.yuv", "-t", "dist.mp4",
         "--hdr-transfer", "pq", "--cvvdp-python", sys.executable],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if "applies only to native metrics" in transfer_ok.stderr:
        fail("cvvdp-hdr-transfer-not-native-only", transfer_ok.stderr[-400:])
    ok("cvvdp-hdr-transfer-not-classified-native-only")


def main() -> int:
    test_resolver_unit()
    test_cli_inference()
    test_learned_native_guard()

    inv = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "test_numerical_cli_invoke.py")],
        cwd=ROOT,
    )
    if inv.returncode != 0:
        return 1

    sys.path.insert(0, str(ROOT / "scripts"))
    from metric_cli_conformance import run_cli_conformance_suite  # noqa: E402

    strict = os.environ.get("PIXVQA_CLI_CONFORMANCE_STRICT", "") in ("1", "true", "yes")
    harness = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "test_metric_fixture_harness.py")],
        cwd=ROOT,
    )
    if harness.returncode != 0:
        return 1

    rc = run_cli_conformance_suite(
        json_report=ROOT / "reports" / "metric_cli_conformance.json",
        strict=strict,
    )

    ffmpeg_build = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "test_metric_resolve_ffmpeg_build.py")],
        cwd=ROOT,
    )
    if ffmpeg_build.returncode != 0:
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
