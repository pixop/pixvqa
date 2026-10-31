#!/usr/bin/env python3
"""Focused dispatch and sampling tests for optional CompressedVQA-HDR NR."""
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


def test_sampling_plan() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from pixvqa_cvqahdr.adapter import AdapterError, sampling_plan

    plan = sampling_plan(120, 60000 / 1001)
    if plan["rounded_fps"] != 60 or plan["sampled_frames"] != 2 or plan["tail_frames"] != 0:
        raise SystemExit(f"sampling-safe: {plan}")
    plan = sampling_plan(60, 24)
    if plan["sampled_frames"] != 2 or plan["tail_frames"] != 12:
        raise SystemExit(f"sampling-boundary: {plan}")
    for frames, needle in ((50, "at least"), (100, "sampling-buffer")):
        try:
            sampling_plan(frames, 60)
        except AdapterError as exc:
            if needle not in str(exc):
                raise
        else:
            raise SystemExit(f"sampling-{frames}: accepted unsafe input")
    print("OK   sampling-plan")


def main() -> int:
    test_sampling_plan()
    if not PIXVQA.is_file():
        print(f"missing pixvqa: {PIXVQA}", file=sys.stderr)
        return 2
    base = ["--metrics", "cvqa-hdr-nr", "-t", "distorted.mp4"]
    expect_fail("missing-assets", base, "--cvqahdr-repo")
    expect_fail("mixed-native-needs-reference", ["--metrics", "psnr,cvqa-hdr-nr", "-t", "d"],
                "required by full-reference metrics: psnr")
    expect_fail("mixed-cvvdp-needs-reference", ["--metrics", "cvvdp", "--metrics", "cvqa-hdr-nr", "-t", "d"],
                "required by full-reference metrics: cvvdp")
    expect_fail("reference-ignored", [*base, "-r", "reference.mp4"], "ignoring -r reference.mp4")
    expect_fail("raw-extension-rejected", [*base, "-t", "distorted.yuv"], "raw YUV")
    expect_fail("raw-options-rejected", [*base, "-s", "1920x1080"], "incompatible")
    expect_fail("frames-incompatible", [*base, "--frames", "7"], "incompatible")
    expect_fail("native-options-incompatible", [*base, "--threads", "2"], "incompatible")
    expect_fail("qcs-default-range-still-needs-assets", [*base, "--qcs"], "--cvqahdr-repo")
    expect_fail("options-without-metric", ["--metrics", "psnr", "-r", "r", "-t", "d",
                                            "--cvqahdr-repo", "repo"],
                "require --metrics cvqa-hdr-nr")
    expect_fail("missing-interpreter", [
        *base, "--cvqahdr-python", "/no/such/python", "--cvqahdr-repo", "repo",
        "--cvqahdr-checkpoint", "checkpoint", "--cvqahdr-profile", "profile",
    ], "could not execute CVQA-HDR-NR interpreter")

    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "pixvqa_cvqahdr"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "__main__.py").write_text(
            """
import json, os, sys
args = sys.argv[1:]
record = args[args.index("--output-json") + 1]
if os.environ.get("MOCK_CVQAHDR_FAIL"):
    open(record, "w", encoding="utf-8").write("failed")
    path_file = os.environ.get("MOCK_CVQAHDR_RECORD_PATH_FILE")
    if path_file:
        open(path_file, "w", encoding="utf-8").write(record)
    print("mock backend failed", file=sys.stderr)
    raise SystemExit(9)
with open(record, "w", encoding="utf-8") as out:
    json.dump({"record": True}, out)
def value(flag):
    return args[args.index(flag) + 1]
if "--ref" in args:
    raise SystemExit("NR backend received a reference")
print(json.dumps({"metric":"cvqa-hdr-nr","quality_score":7.409712345678,
                  "raw_model_score":0.123456789,"sampled_frames":2,
                  "device":value("--device"),
                  "test":value("--test"),
                  "repo":value("--repo"),
                  "checkpoint":value("--checkpoint"),
                  "profile":value("--profile"),
                  "output_json":None if "--transient-output" in args else record,
                  "backend_record":record}, separators=(",", ":")))
""",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = tmp + os.pathsep + env.get("PYTHONPATH", "")
        output = Path(tmp) / "result.json"
        result = run([
            *base, "--metric-device", "cpu", "--cvqahdr-python", sys.executable,
            "--cvqahdr-repo", "/upstream", "--cvqahdr-checkpoint", "/checkpoint.pth",
            "--cvqahdr-profile", "/profile.npy", "--cvqahdr-output-json", str(output),
        ], env)
        if result.returncode != 0:
            raise SystemExit(f"mock-success: {result.stderr}")
        payload = json.loads(result.stdout)
        if (payload["quality_score"] != 7.409712345678 or payload["raw_model_score"] != 0.123456789
                or payload["device"] != "cpu" or payload["test"] != "distorted.mp4"
                or not output.is_file()):
            raise SystemExit(f"mock-success: bad payload {result.stdout!r}")
        print("OK   mocked-dispatch")
        transient = run([
            *base, "--cvqahdr-python", sys.executable, "--cvqahdr-repo", "/upstream",
            "--cvqahdr-checkpoint", "/checkpoint.pth", "--cvqahdr-profile", "/profile.npy",
        ], env)
        if transient.returncode != 0:
            raise SystemExit(f"transient-success: {transient.stderr}")
        transient_payload = json.loads(transient.stdout)
        if transient_payload["output_json"] is not None or Path(transient_payload["backend_record"]).exists():
            raise SystemExit(f"transient-success: output was retained {transient.stdout!r}")
        print("OK   temporary-output-cleanup-success")
        env["MOCK_CVQAHDR_FAIL"] = "1"
        record_path_file = Path(tmp) / "failed-record-path"
        env["MOCK_CVQAHDR_RECORD_PATH_FILE"] = str(record_path_file)
        expect_fail("backend-failure", [
            *base, "--cvqahdr-python", sys.executable, "--cvqahdr-repo", "/upstream",
            "--cvqahdr-checkpoint", "/checkpoint.pth", "--cvqahdr-profile", "/profile.npy",
        ], "CVQA-HDR-NR backend exited 9", env)
        if not record_path_file.is_file() or Path(record_path_file.read_text(encoding="utf-8")).exists():
            raise SystemExit("temporary output was not removed after backend failure")
        print("OK   temporary-output-cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
