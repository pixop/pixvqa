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


def main() -> int:
    test_official_csv()
    if not PIXVQA.is_file():
        print(f"missing pixvqa: {PIXVQA}", file=sys.stderr)
        return 2
    base = ["--metrics", "cvvdp", "-r", "reference.mkv", "-t", "distorted.mkv"]
    expect_fail("missing-interpretation", [*base, "--cvvdp-display", "standard_hdr_pq"],
                "--cvvdp-interpretation")
    expect_fail("invalid-interpretation", [*base, "--cvvdp-interpretation", "hlg",
                                            "--cvvdp-display", "standard_hdr_pq"],
                "expects sdr or pq")
    expect_fail("mixed-metrics", ["--metrics", "psnr,cvvdp", "-r", "r", "-t", "d",
                                   "--cvvdp-interpretation", "pq", "--cvvdp-display", "standard_hdr_pq"],
                "standalone")
    expect_fail("mixed-metrics-repeated-native-first", ["--metrics", "psnr", "--metrics", "cvvdp",
                                                         "-r", "r", "-t", "d",
                                                         "--cvvdp-interpretation", "pq",
                                                         "--cvvdp-display", "standard_hdr_pq"],
                "standalone")
    expect_fail("mixed-metrics-repeated-cvvdp-first", ["--metrics", "cvvdp", "--metrics", "psnr",
                                                        "-r", "r", "-t", "d",
                                                        "--cvvdp-interpretation", "pq",
                                                        "--cvvdp-display", "standard_hdr_pq"],
                "standalone")
    expect_fail("qcs-incompatible", [*base, "--cvvdp-interpretation", "pq",
                                      "--cvvdp-display", "standard_hdr_pq", "--qcs"],
                "incompatible")
    expect_fail("options-without-cvvdp", ["--metrics", "psnr", "-r", "r", "-t", "d",
                                          "--cvvdp-display", "standard_hdr_pq"],
                "require --metrics cvvdp")
    expect_fail("missing-interpreter", [*base, "--cvvdp-interpretation", "pq",
                                        "--cvvdp-display", "standard_hdr_pq",
                                        "--cvvdp-python", "/no/such/python"],
                "could not execute CVVDP interpreter")

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
print(json.dumps({"metric":"cvvdp","jod":9.5,"display":args[args.index("--display") + 1],
                  "interpretation":args[args.index("--interpretation") + 1],
                  "device":args[args.index("--device") + 1],
                  "reference":args[args.index("--ref") + 1],
                  "test":args[args.index("--test") + 1],
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
