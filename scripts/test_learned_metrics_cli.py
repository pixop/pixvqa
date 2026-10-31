#!/usr/bin/env python3
"""Mocked compatibility suite for mixed native, CVVDP, and CompressedVQA-HDR NR runs.

No GPU, PyTorch, checkpoint, or upstream checkout is needed. The learned backends are
replaced by mock Python packages on PYTHONPATH. Cases that need a native decoder use
PIXVQA_DECODE_BIN (default build/swdec/pixvqa) and ffmpeg; they print SKIP when either
is absent.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXVQA = Path(os.environ.get("PIXVQA_BIN", ROOT / "build" / "cpu" / "pixvqa"))
DECODE_BIN = Path(os.environ.get("PIXVQA_DECODE_BIN", ROOT / "build" / "swdec" / "pixvqa"))
FFMPEG = os.environ.get("PIXVQA_TEST_FFMPEG") or shutil.which("ffmpeg")
FAILS = 0

CVVDP_JOD = "9.311432838439941"
NR_SCORE = "7.40969689895428"
NR_RAW = "-0.1277415007352829"
# frame_index, time_seconds, raw quality1, calibrated score (120 frames at 60000/1001)
NR_SAMPLES = [
    [30, 30 / (60000 / 1001), -0.10000000149011612, 7.3],
    [90, 90 / (60000 / 1001), -0.15548300743103027, 7.5],
]
# CVVDP per frame: pooled distortion q[f] and met2jod(q[f]).
CVVDP_FRAMES = [[0.05, 9.6], [0.21, 8.25]]

MOCK = r'''
import json, os, sys
NAME = "@NAME@"
ENV = "@ENV@"
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1] if flag in args else None
mode = "check" if "--check" in args else "run"
log = os.environ.get("MOCK_LOG")
if log:
    seen = ""
    native_stdout = os.environ.get("MOCK_NATIVE_STDOUT")
    if mode == "run" and native_stdout and "[ PSNR" in open(native_stdout, encoding="utf-8").read():
        seen = " native-before"
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(NAME + " " + mode + seen + " " + json.dumps(args) + "\n")
print("mock " + NAME + " warning on stderr", file=sys.stderr)
if mode == "check":
    if os.environ.get("MOCK_" + ENV + "_PREFLIGHT_FAIL"):
        print("mock " + NAME + " preflight failed", file=sys.stderr)
        raise SystemExit(2)
    print("pixvqa-check ok")
    raise SystemExit(0)
record = value("--output-json")
path_file = os.environ.get("MOCK_" + ENV + "_RECORD_PATH_FILE")
if path_file:
    open(path_file, "w", encoding="utf-8").write(record)
if os.environ.get("MOCK_" + ENV + "_FAIL"):
    open(record, "w", encoding="utf-8").write("failed")
    print("mock backend failed", file=sys.stderr)
    raise SystemExit(9)
with open(record, "w", encoding="utf-8") as out:
    json.dump({"record": True}, out)
transient = "--transient-output" in args
if value("--protocol") == "lines":
    if os.environ.get("MOCK_" + ENV + "_BAD_PROTOCOL"):
        print("not a protocol line")
        raise SystemExit(0)
    lines = ["pixvqa-protocol 1", "pixvqa-value metric " + NAME]
    if NAME == "cvvdp":
        frames = json.loads(os.environ.get("MOCK_CVVDP_FRAMES", "@CVVDP_FRAMES@"))
        lines += ["pixvqa-value jod @JOD@",
                  "pixvqa-value frames_evaluated " + os.environ.get("MOCK_CVVDP_EVALUATED", str(len(frames))),
                  "pixvqa-value display " + value("--display"),
                  "pixvqa-value interpretation " + value("--interpretation"),
                  "pixvqa-value device " + value("--device")]
        if "--temporal" in args:
            lines += ["pixvqa-value temporal_frames %d" % len(frames),
                      "pixvqa-value pooled_parity " + os.environ.get("MOCK_CVVDP_PARITY", "1"),
                      "pixvqa-value temporal_basis per-frame-jod-equivalent"]
            for i, (q, jod) in enumerate(frames):
                lines.append("pixvqa-sample %d %d %r %r %r" % (i, i, i / 60.0, q, jod))
    else:
        samples = json.loads(os.environ.get("MOCK_NR_SAMPLES", "@SAMPLES@"))
        padded = os.environ.get("MOCK_NR_PADDED", "0" if "--verify-decoded-frames" in args else "unknown")
        lines += ["pixvqa-value quality_score @SCORE@", "pixvqa-value raw_model_score @RAW@",
                  "pixvqa-value sampled_frames %d" % len(samples),
                  "pixvqa-value opencv_frame_count 120", "pixvqa-value opencv_fps 59.94005994005994",
                  "pixvqa-value rounded_fps 60",
                  "pixvqa-value sampling_density_hz 0.999000999000999",
                  "pixvqa-value padded_samples " + padded,
                  "pixvqa-value pooled_parity " + os.environ.get("MOCK_NR_PARITY", "1")]
        for i, (frame, t, raw, cal) in enumerate(samples):
            lines.append("pixvqa-sample %d %d %r %r %r" % (i, frame, t, raw, cal))
    lines += ["pixvqa-value device " + value("--device"),
              "pixvqa-value output_json " + ("none" if transient else record), "pixvqa-end"]
    text = "\n".join(lines) + "\n"
elif NAME == "cvvdp":
    text = json.dumps({"metric": "cvvdp", "jod": @JOD@, "display": value("--display"),
                       "interpretation": value("--interpretation"), "device": value("--device"),
                       "reference": value("--ref"), "test": value("--test"),
                       "output_json": None if transient else record,
                       "backend_record": record}, separators=(",", ":")) + "\n"
else:
    text = json.dumps({"metric": "cvqa-hdr-nr", "quality_score": @SCORE@,
                       "raw_model_score": @RAW@, "sampled_frames": 2, "device": value("--device"),
                       "test": value("--test"), "reference": value("--ref"),
                       "output_json": None if transient else record,
                       "backend_record": record}, separators=(",", ":")) + "\n"
if os.environ.get("MOCK_FLOOD"):
    chunk = 65536
    noise = "e" * (chunk - 1) + "\n"
    pieces = [text[i:i + chunk] for i in range(0, len(text), chunk)]
    for i in range(32):
        sys.stderr.write(noise)
        sys.stderr.flush()
        if i < len(pieces):
            sys.stdout.write(pieces[i])
            sys.stdout.flush()
    sys.stdout.write("".join(pieces[32:]))
else:
    sys.stdout.write(text)
'''


def fail(name: str, detail: str) -> None:
    global FAILS
    FAILS += 1
    print(f"FAIL {name}: {detail}", file=sys.stderr)


def ok(name: str) -> None:
    print(f"OK   {name}")


def write_mocks(root: Path) -> None:
    for package, name, env in (("pixvqa_cvvdp", "cvvdp", "CVVDP"), ("pixvqa_cvqahdr", "cvqa-hdr-nr", "NR")):
        directory = root / package
        directory.mkdir()
        (directory / "__init__.py").write_text("", encoding="utf-8")
        source = (MOCK.replace("@NAME@", name).replace("@ENV@", env).replace("@JOD@", CVVDP_JOD)
                  .replace("@SCORE@", NR_SCORE).replace("@RAW@", NR_RAW)
                  .replace("@SAMPLES@", json.dumps(NR_SAMPLES))
                  .replace("@CVVDP_FRAMES@", json.dumps(CVVDP_FRAMES)))
        (directory / "__main__.py").write_text(source, encoding="utf-8")


def write_yuv(path: Path, frames: int, value: int = 0, size: int = 16, bits: int = 10) -> None:
    """Flat 4:2:0 frames; 10-bit samples are little-endian uint16."""
    sample = value.to_bytes(2 if bits > 8 else 1, "little")
    path.write_bytes(sample * (size * size * 3 // 2) * frames)


class Suite:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.mocks = tmp / "mocks"
        self.mocks.mkdir()
        write_mocks(self.mocks)
        self.log = tmp / "backend.log"
        self.ref = tmp / "reference.yuv"
        self.test = tmp / "test.yuv"
        self.short = tmp / "short.yuv"
        write_yuv(self.ref, 2, 64)
        write_yuv(self.test, 2, 80)
        write_yuv(self.short, 1, 80)
        self.fake_mp4 = tmp / "fake.mp4"
        self.fake_mp4.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)

    def env(self, **extra: str) -> dict[str, str]:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("MOCK_") or key in ("PIXVQA_CVVDP_PYTHON", "PIXVQA_CVQAHDR_PYTHON",
                                                  "PIXVQA_QCS_TRACE"):
                env.pop(key)
        env["PYTHONPATH"] = str(self.mocks)
        env["MOCK_LOG"] = str(self.log)
        env["PIXVQA_LEARNED_CONFIG"] = "none"
        env.update(extra)
        return env

    def run(self, args: list[str], binary: Path | None = None, stdout_file: Path | None = None,
            **extra: str) -> subprocess.CompletedProcess[str]:
        if self.log.exists():
            self.log.unlink()
        env = self.env(**extra)
        cmd = [str(binary or PIXVQA), *args]
        if stdout_file is None:
            return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        env["MOCK_NATIVE_STDOUT"] = str(stdout_file)
        with stdout_file.open("w", encoding="utf-8") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True, env=env, timeout=120)
        proc.stdout = stdout_file.read_text(encoding="utf-8")
        return proc

    def backend_log(self) -> list[str]:
        if not self.log.exists():
            return []
        return [" ".join(line.split(" ")[:2]) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def backend_args(self, name: str, mode: str = "run") -> list[str]:
        for line in self.log.read_text(encoding="utf-8").splitlines():
            parts = line.split(" ", 3)
            if parts[0] == name and parts[1] == mode:
                return json.loads(line[line.index("["):])
        return []

    def native_before(self, name: str) -> bool:
        return any(line.startswith(name + " run native-before")
                   for line in self.log.read_text(encoding="utf-8").splitlines())

    def cvvdp(self) -> list[str]:
        return ["--cvvdp-interpretation", "pq", "--cvvdp-display", "standard_hdr_pq",
                "--cvvdp-python", sys.executable]

    def nr(self) -> list[str]:
        return ["--cvqahdr-python", sys.executable, "--cvqahdr-repo", "/upstream",
                "--cvqahdr-checkpoint", "/checkpoint.pth", "--cvqahdr-profile", "/profile.npy"]

    def raw_pair(self, ref: Path | None = None, test: Path | None = None) -> list[str]:
        return ["-r", str(ref or self.ref), "-t", str(test or self.test), "-s", "16x16", "-f", "420p:10",
                "--fps", "60"]


def expect(name: str, proc: subprocess.CompletedProcess[str], code: int, needles=(), stdout_needles=(),
           absent_stdout=()) -> bool:
    good = proc.returncode == code
    good = good and all(n in proc.stderr for n in needles)
    good = good and all(n in proc.stdout for n in stdout_needles)
    good = good and not any(n in proc.stdout for n in absent_stdout)
    if not good:
        fail(name, f"exit {proc.returncode} (want {code})\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")
        return False
    return True


def summary_block(stdout: str) -> str:
    return stdout[stdout.index("Summary:"):] if "Summary:" in stdout else ""


def expected_qcs(values, lower, upper) -> float:
    return 100.0 * sum(min(max((v - lower) / (upper - lower), 0.0), 1.0) for v in values) / len(values)


def cli_cases(s: Suite) -> None:
    # Native only: no backend runs and the native summary is unchanged.
    native_args = ["--metrics", "psnr", "--qcs", *s.raw_pair()]
    native = s.run(native_args)
    if expect("native-only", native, 0, stdout_needles=["Summary:", "[ PSNR", "[ PSNR-QCS value:"],
              absent_stdout=["CVVDP", "CVQA-HDR-NR"]) and not s.backend_log():
        ok("native-only")
    elif native.returncode == 0:
        fail("native-only", f"backend ran: {s.backend_log()}")

    # CVVDP only keeps the standalone JSON on stdout; warnings stay on stderr.
    proc = s.run(["--metrics", "cvvdp", "-r", "ref.mkv", "-t", "dist.mkv", *s.cvvdp()])
    if expect("cvvdp-only", proc, 0, needles=["mock cvvdp warning"]):
        payload = json.loads(proc.stdout)
        if repr(payload["jod"]) != CVVDP_JOD or "warning" in proc.stdout or s.backend_log() != ["cvvdp run"]:
            fail("cvvdp-only", proc.stdout)
        else:
            ok("cvvdp-only-standalone-json")

    proc = s.run(["--metrics", "cvqa-hdr-nr", "-t", "dist.mp4", *s.nr()])
    if expect("nr-only", proc, 0, needles=["mock cvqa-hdr-nr warning"]):
        payload = json.loads(proc.stdout)
        if (repr(payload["quality_score"]) != NR_SCORE or repr(payload["raw_model_score"]) != NR_RAW
                or payload["reference"] is not None):
            fail("nr-only", proc.stdout)
        else:
            ok("nr-only-standalone-json-full-precision")

    proc = s.run(["--metrics", "cvqa-hdr-nr", "-r", "/no/such/reference.mp4", "-t", "dist.mp4", *s.nr()])
    if expect("nr-ignores-reference", proc, 0, needles=["ignoring -r /no/such/reference.mp4"]):
        if "--ref" in s.backend_args("cvqa-hdr-nr") or json.loads(proc.stdout)["reference"] is not None:
            fail("nr-ignores-reference", "reference reached the backend")
        else:
            ok("nr-unnecessary-reference-ignored-unopened")

    # Native + CVVDP on raw YUV420p10le PQ.
    out = s.tmp / "native-cvvdp.out"
    proc = s.run(["--metrics", "psnr,cvvdp", *s.raw_pair(), *s.cvvdp()], stdout_file=out)
    if expect("native-cvvdp", proc, 0, stdout_needles=["[ PSNR", f"[ CVVDP jod:{CVVDP_JOD} frames:2"]):
        args = s.backend_args("cvvdp")
        if (s.backend_log() != ["cvvdp check", "cvvdp run"] or not s.native_before("cvvdp")
                or args[args.index("--raw-width") + 1] != "16" or args[args.index("--raw-fps-num") + 1] != "60"):
            fail("native-cvvdp", f"{s.backend_log()} {args}")
        else:
            ok("native-cvvdp-order-and-raw-dispatch")

    # Native QCS is unchanged by an added learned metric. CVVDP QCS integrates the
    # per-frame JOD-equivalent sequence over the default [0, 10].
    cvvdp_jods = [frame[1] for frame in CVVDP_FRAMES]
    proc = s.run(["--metrics", "psnr", "--metrics", "cvvdp", "--qcs", *s.raw_pair(), *s.cvvdp()],
                 PIXVQA_QCS_TRACE="1")
    want = f"[ CVVDP-QCS value:{expected_qcs(cvvdp_jods, 0.0, 10.0):.6f} range:[0,10] frames:2 " \
           "basis:per-frame-jod-equivalent ]"
    if expect("native-qcs-unchanged", proc, 0, stdout_needles=[want], needles=["per-frame JOD-equivalent"]):
        native_qcs = re.findall(r"\[ PSNR-QCS [^\n]*", native.stdout)
        mixed_qcs = re.findall(r"\[ PSNR-QCS [^\n]*", proc.stdout)
        native_frames = re.findall(r"^\d+\s*:.*$", native.stdout, re.M)
        mixed_frames = re.findall(r"^\d+\s*:.*$", proc.stdout, re.M)
        traced = [float(v) for v in re.findall(r"qcs_sample cvvdp (\S+)", proc.stderr)]
        if not native_qcs or native_qcs != mixed_qcs or native_frames != mixed_frames:
            fail("native-qcs-unchanged", f"{native_qcs} vs {mixed_qcs}")
        elif traced != cvvdp_jods:
            fail("native-qcs-unchanged", f"cvvdp trace {traced}")
        elif "--temporal" not in s.backend_args("cvvdp") or "--temporal" not in s.backend_args("cvvdp", "check"):
            fail("native-qcs-unchanged", "cvvdp --temporal not requested")
        else:
            ok("native-qcs-unchanged-cvvdp-qcs-default-0-10")

    proc = s.run(["--metrics", "psnr,cvvdp", *s.raw_pair(), *s.cvvdp()])
    if expect("cvvdp-no-qcs-no-temporal", proc, 0, absent_stdout=["CVVDP-QCS"]):
        if "--temporal" in s.backend_args("cvvdp"):
            fail("cvvdp-no-qcs-no-temporal", "temporal requested without --qcs")
        else:
            ok("cvvdp-temporal-only-with-qcs")
    proc = s.run(["--metrics", "psnr,cvvdp", "--qcs", "--qcs-range", "cvvdp=8:9.5", *s.raw_pair(), *s.cvvdp()])
    want = f"value:{expected_qcs(cvvdp_jods, 8.0, 9.5):.6f} range:[8,9.5]"
    if expect("cvvdp-qcs-range", proc, 0, stdout_needles=["[ CVVDP-QCS " + want],
              absent_stdout=["unsupported"]):
        ok("cvvdp-qcs-range-override")
    cvvdp_qcs = ["--metrics", "psnr,cvvdp", "--qcs", *s.raw_pair(), *s.cvvdp()]
    for name, extra, reason in (
            ("cvvdp-qcs-parity", {"MOCK_CVVDP_PARITY": "0"}, "pooled-parity-mismatch"),
            ("cvvdp-qcs-frame-count", {"MOCK_CVVDP_EVALUATED": "3"}, "frame-count-mismatch"),
            ("cvvdp-qcs-single-frame", {"MOCK_CVVDP_FRAMES": "[[0.1, 9.5]]", "MOCK_CVVDP_EVALUATED": "1"},
             "single-frame"),
            ("cvvdp-qcs-failed", {"MOCK_CVVDP_FAIL": "1"}, "backend-failed")):
        proc = s.run(cvvdp_qcs, **extra)
        code = 1 if "MOCK_CVVDP_FAIL" in extra else 0
        if expect(name, proc, code, stdout_needles=[f"[ CVVDP-QCS value:unavailable range:[0,10] reason:{reason} ]"]):
            ok(name)

    # CVVDP + NR without native metrics: text report, preflight both, then run in order.
    proc = s.run(["--metrics", "cvvdp,cvqa-hdr-nr", "-r", "ref.mkv", "-t", "dist.mkv", *s.cvvdp(), *s.nr()])
    if expect("cvvdp-nr", proc, 0,
              stdout_needles=[f"[ CVVDP jod:{CVVDP_JOD} frames:2 display:standard_hdr_pq interpretation:pq "
                              "device:auto ]",
                              f"[ CVQA-HDR-NR quality_score:{NR_SCORE} "
                              f"raw_model_score:{NR_RAW} samples:2 frames:120 density:0.999/s device:auto ]"],
              absent_stdout=["warning"]):
        if s.backend_log() != ["cvvdp check", "cvqa-hdr-nr check", "cvvdp run", "cvqa-hdr-nr run"]:
            fail("cvvdp-nr", str(s.backend_log()))
        elif "--ref" in s.backend_args("cvqa-hdr-nr"):
            fail("cvvdp-nr", "NR received the reference")
        else:
            ok("cvvdp-nr-preflight-then-ordered-run")

    # Reference and test requirements.
    proc = s.run(["--metrics", "cvvdp,cvqa-hdr-nr", "-t", "dist.mkv", *s.cvvdp(), *s.nr()])
    if expect("missing-reference", proc, 2, needles=["required by full-reference metrics: cvvdp\n"]) \
            and not s.backend_log():
        ok("missing-reference-learned")
    proc = s.run(["--metrics", "psnr,cvqa-hdr-nr", "-t", str(s.test), *s.nr()])
    if expect("missing-reference-native", proc, 2, needles=["required by full-reference metrics: psnr\n"]) \
            and not s.backend_log():
        ok("missing-reference-native-plus-nr")
    proc = s.run(["--metrics", "cvqa-hdr-nr", "-r", "ref.mp4", *s.nr()])
    if expect("missing-test", proc, 2, needles=["-t <distorted> is required"]) and not s.backend_log():
        ok("missing-test")

    # Format compatibility is resolved before any metric runs.
    proc = s.run(["--metrics", "psnr,cvvdp,cvqa-hdr-nr", *s.raw_pair(), *s.cvvdp(), *s.nr()])
    if expect("nr-raw-rejected", proc, 2, needles=["compressed video only"]) and not s.backend_log():
        ok("all-three-raw-test-rejected-before-run")
    proc = s.run(["--metrics", "cvqa-hdr-nr", "-t", "dist.yuv", *s.nr()])
    if expect("nr-raw-extension", proc, 2, needles=["raw YUV is unsupported"]):
        ok("nr-raw-extension-rejected")
    proc = s.run(["--metrics", "cvqa-hdr-nr", "-t", "dist.mp4", "-s", "16x16", *s.nr()])
    if expect("nr-raw-option", proc, 2, needles=["-s is incompatible with cvqa-hdr-nr"]):
        ok("nr-raw-option-rejected")
    proc = s.run(["--metrics", "psnr,cvqa-hdr-nr", "-r", str(s.fake_mp4), "-t", str(s.fake_mp4), *s.nr()])
    if expect("cpu-build-compressed", proc, 2, needles=["no FFmpeg decoder for native metrics"]) \
            and not s.backend_log():
        ok("cpu-build-native-compressed-clear-error")
    proc = s.run(["--metrics", "cvvdp", "--threads", "2", "-r", "r.mkv", "-t", "d.mkv", *s.cvvdp()])
    if expect("native-option-alone", proc, 2, needles=["--threads is incompatible with cvvdp alone"]):
        ok("native-only-option-rejected-without-native-metric")

    # --frames conflicts with NR in every combination.
    for name, metrics in (("frames-nr", "cvqa-hdr-nr"), ("frames-native-nr", "psnr,cvqa-hdr-nr")):
        args = ["--metrics", metrics, "--frames", "5", "-t", "dist.mp4", *s.nr()]
        if "psnr" in metrics:
            args = ["-r", str(s.ref), *args]
        proc = s.run(args)
        if expect(name, proc, 2, needles=["--frames is incompatible with cvqa-hdr-nr"]) and not s.backend_log():
            ok(name)

    # Invalid environments fail preflight with exit 2 and nothing runs.
    proc = s.run(["--metrics", "cvvdp", "-r", "r.mkv", "-t", "d.mkv", "--cvvdp-interpretation", "pq",
                  "--cvvdp-display", "standard_hdr_pq", "--cvvdp-python", "/no/such/python"])
    if expect("missing-interpreter", proc, 2, needles=["could not execute CVVDP interpreter"]):
        ok("missing-interpreter-exit-2")
    args = ["--metrics", "cvvdp,cvqa-hdr-nr", "-r", "r.mkv", "-t", "d.mkv", *s.cvvdp(), *s.nr()]
    proc = s.run(args, MOCK_NR_PREFLIGHT_FAIL="1")
    if expect("model-preflight", proc, 2, needles=["mock cvqa-hdr-nr preflight failed",
                                                   "CVQA-HDR-NR preflight failed"]):
        if any(line.endswith(" run") for line in s.backend_log()):
            fail("model-preflight", str(s.backend_log()))
        else:
            ok("invalid-model-preflight-exit-2-nothing-ran")
    proc = s.run(["--metrics", "psnr,cvvdp", *s.raw_pair(), *s.cvvdp()], MOCK_CVVDP_PREFLIGHT_FAIL="1")
    if expect("native-preflight", proc, 2, absent_stdout=["Summary:"]) and s.backend_log() == ["cvvdp check"]:
        ok("preflight-before-native")
    elif proc.returncode == 2:
        fail("native-preflight", str(s.backend_log()))

    # Partial failures keep completed results and exit 1.
    proc = s.run(args, MOCK_CVVDP_FAIL="1")
    if expect("partial-learned", proc, 1, needles=["CVVDP backend exited 9"],
              stdout_needles=["[ CVVDP status:failed ]", f"[ CVQA-HDR-NR quality_score:{NR_SCORE}"]):
        ok("partial-failure-learned-exit-1")
    proc = s.run(["--metrics", "psnr,cvvdp", *s.raw_pair(), *s.cvvdp()], MOCK_CVVDP_FAIL="1")
    if expect("partial-native", proc, 1, stdout_needles=["[ PSNR", "[ CVVDP status:failed ]", "Total frames:2"]):
        ok("partial-failure-native-kept")
    proc = s.run(["--metrics", "psnr,cvvdp", *s.raw_pair(test=s.short), *s.cvvdp()])
    if expect("native-runtime-failure", proc, 1, needles=["frame count mismatch"],
              stdout_needles=["[ NATIVE status:failed ]", f"[ CVVDP jod:{CVVDP_JOD}"]):
        ok("native-runtime-failure-learned-still-reported")
    proc = s.run(args, MOCK_NR_BAD_PROTOCOL="1")
    if expect("bad-protocol", proc, 1, needles=["did not start with pixvqa-protocol"],
              stdout_needles=["[ CVQA-HDR-NR status:failed ]"]):
        ok("malformed-protocol-is-runtime-failure")

    # NR QCS defaults to [0, 10], like CVVDP. An explicit range still overrides it.
    nr_only = ["--metrics", "cvqa-hdr-nr", "-t", "dist.mp4", *s.nr()]
    calibrated = [sample[3] for sample in NR_SAMPLES]
    proc = s.run([*nr_only, "--qcs"], PIXVQA_QCS_TRACE="1")
    default_qcs = f"value:{expected_qcs(calibrated, 0.0, 10.0):.6f} range:[0,10]"
    if expect("nr-qcs-default-range", proc, 0,
              stdout_needles=["[ CVQA-HDR-NR-QCS " + default_qcs, "basis:sampled-frames",
                              f"quality_score:{NR_SCORE} "],
              needles=["sampled frames"]):
        ok("nr-qcs-default-range-0-10-score-unclamped")
    proc = s.run(["--metrics", "psnr", "--qcs", "--qcs-range", "cvqa-hdr-nr=7:8", *s.raw_pair()])
    if expect("nr-range-without-nr", proc, 2, needles=["requires cvqa-hdr-nr in --metrics"]):
        ok("nr-qcs-range-requires-nr")
    for lower, upper in ((7.0, 8.0), (7.4, 7.45), (6.0, 7.0), (7.6, 9.0)):
        proc = s.run([*nr_only, "--qcs", "--qcs-range", f"cvqa-hdr-nr={lower}:{upper}"],
                     PIXVQA_QCS_TRACE="1")
        want = f"value:{expected_qcs(calibrated, lower, upper):.6f}"
        name = f"nr-qcs-threshold-{lower}-{upper}"
        if expect(name, proc, 0, stdout_needles=["[ CVQA-HDR-NR-QCS " + want, "samples:2 basis:sampled-frames ]"],
                  needles=["sampled frames"]):
            traced = re.findall(r"qcs_sample cvqa-hdr-nr (\S+)", proc.stderr)
            if [float(v) for v in traced] != calibrated or "--verify-decoded-frames" not in s.backend_args("cvqa-hdr-nr"):
                fail(name, f"trace {traced}")
            else:
                ok(name)
    proc = s.run([*nr_only, "--qcs", "--qcs-range", "cvqa-hdr-nr=7:8"], MOCK_NR_PADDED="1")
    if expect("nr-qcs-padded", proc, 0, stdout_needles=["value:unavailable range:[7,8] reason:padded-samples"]):
        ok("nr-qcs-padded-samples-not-scored")
    proc = s.run([*nr_only, "--qcs", "--qcs-range", "cvqa-hdr-nr=7:8"], MOCK_NR_PARITY="0")
    if expect("nr-qcs-parity", proc, 0, stdout_needles=["reason:pooled-parity-mismatch"]):
        ok("nr-qcs-parity-mismatch-unavailable")
    proc = s.run([*nr_only, "--qcs", "--qcs-range", "cvqa-hdr-nr=7:8"], MOCK_NR_FAIL="1")
    if expect("nr-qcs-failed", proc, 1, stdout_needles=["[ CVQA-HDR-NR status:failed ]", "reason:backend-failed"]):
        ok("nr-qcs-backend-failure")

    # Stream isolation under load: 2 MiB of stderr interleaved with stdout.
    proc = s.run(["--metrics", "cvvdp,cvqa-hdr-nr", "-r", "r.mkv", "-t", "d.mkv", *s.cvvdp(), *s.nr()],
                 MOCK_FLOOD="1")
    if expect("flood-report", proc, 0, stdout_needles=[f"[ CVVDP jod:{CVVDP_JOD}"], absent_stdout=["eeee"]):
        ok("no-deadlock-report-mode")
    proc = s.run(["--metrics", "cvqa-hdr-nr", "-t", "dist.mp4", *s.nr()], MOCK_FLOOD="1")
    if expect("flood-json", proc, 0) and repr(json.loads(proc.stdout)["quality_score"]) == NR_SCORE:
        ok("no-deadlock-standalone-json")

    # Temporary records are removed after success and failure in combined runs.
    for name, extra, code in (("cleanup-success", {}, 0), ("cleanup-failure", {"MOCK_NR_FAIL": "1"}, 1)):
        cvvdp_path = s.tmp / f"{name}-cvvdp"
        nr_path = s.tmp / f"{name}-nr"
        proc = s.run(args, MOCK_CVVDP_RECORD_PATH_FILE=str(cvvdp_path), MOCK_NR_RECORD_PATH_FILE=str(nr_path),
                     **extra)
        if expect(name, proc, code):
            records = [Path(p.read_text(encoding="utf-8")) for p in (cvvdp_path, nr_path)]
            if any(r.exists() or r.parent.exists() for r in records):
                fail(name, f"temporary record kept: {records}")
            else:
                ok(name)
    persistent = s.tmp / "persistent-nr.json"
    proc = s.run([*args, "--cvqahdr-output-json", str(persistent)])
    if expect("persistent-record", proc, 0) and persistent.is_file():
        ok("persistent-record-kept")

    # --metrics all selects every metric. --preset hdr fills PQ, the HDR VMAF
    # model, and the CVVDP display before any file or backend is opened.
    proc = s.run(["--metrics", "all", "-r", "ref.mp4", "-t", "dist.mp4"])
    if expect("metrics-all", proc, 2, needles=["--cvvdp-interpretation"]) and not s.backend_log():
        ok("metrics-all-selects-cvvdp")
    proc = s.run(["--preset", "sdr"])
    if expect("preset-unknown", proc, 2, needles=["--preset expects hdr"]) and not s.backend_log():
        ok("preset-rejects-unknown")
    proc = s.run(["--preset", "hdr", "--hdr-transfer", "sdr", "-r", "ref.mp4", "-t", "dist.mp4"])
    if expect("preset-transfer-conflict", proc, 2, needles=["--preset hdr uses pq; transfer sdr conflicts"]) \
            and not s.backend_log():
        ok("preset-hdr-rejects-non-pq-transfer")
    proc = s.run(["--preset", "hdr", "--cvvdp-interpretation", "sdr", "-r", "ref.mp4", "-t", "dist.mp4", *s.nr()])
    if expect("preset-cvvdp-conflict", proc, 2, needles=["--cvvdp-interpretation pq"]) and not s.backend_log():
        ok("preset-hdr-rejects-sdr-cvvdp")
    group_cases(s)
    config_cases(s)
    device_cases(s)
    proc = s.run(["--preset", "hdr", "-r", "ref.mp4", "-t", "dist.mp4"])
    if expect("preset-selects-nr", proc, 2, needles=["--cvqahdr-repo"]) and not s.backend_log():
        if "requires --cvvdp-interpretation" in proc.stderr or "requires --model" in proc.stderr:
            fail("preset-selects-nr", proc.stderr)
        else:
            ok("preset-hdr-selects-all-and-fills-cvvdp")
    proc = s.run(["--preset", "hdr", "--metrics", "vmaf", *s.raw_pair()])
    if proc.returncode in (0, 2) and "requires --model" not in proc.stderr and (
            "no libvmaf" in proc.stderr or "hdr_vmaf_v1.1.8_3d0h" in proc.stderr or
            "[ VMAF" in proc.stdout):
        ok("preset-hdr-supplies-hdr-vmaf-model")
    else:
        fail("preset-vmaf", f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def device_cases(s: Suite) -> None:
    """--device/--metric-device: auto stays auto for the adapters, gpu becomes cuda."""
    pair = ["--metrics", "cvvdp,cvqa-hdr-nr", "-r", "ref.mkv", "-t", "dist.mkv", *s.cvvdp(), *s.nr()]
    for flag, choice, token in (("--device", "cpu", "cpu"), ("--device", "gpu", "cuda"),
                                ("--metric-device", "gpu", "cuda"), ("--device", "auto", "auto"),
                                (None, None, "auto")):
        name = f"device-{flag or 'default'}-{choice or 'none'}"
        proc = s.run([*pair, *([flag, choice] if flag else [])])
        if not expect(name, proc, 0, stdout_needles=[f"interpretation:pq device:{token} ]",
                                                     f"density:0.999/s device:{token} ]"]):
            continue
        sent = [s.backend_args(m, mode)[s.backend_args(m, mode).index("--device") + 1]
                for m in ("cvvdp", "cvqa-hdr-nr") for mode in ("check", "run")]
        if sent != [token] * 4:
            fail(name, str(sent))
        else:
            ok(name)
    proc = s.run([*pair, "--device", "cpu", "--metric-device", "gpu"])
    if expect("device-conflict", proc, 2, needles=["conflicting --device cpu and --metric-device gpu"]) \
            and not s.backend_log():
        ok("device-conflict-rejected")
    proc = s.run([*pair, "--device", "cpu", "--metric-device", "cpu"])
    if expect("device-repeat-agrees", proc, 0, stdout_needles=["device:cpu ]"]):
        ok("device-repeat-agrees")
    proc = s.run([*pair, "--device", "metal"])
    if expect("device-invalid", proc, 2, needles=["unsupported --device metal (use auto, cpu, or gpu)"]) \
            and not s.backend_log():
        ok("device-invalid-rejected")

    base = ["--metrics", "cvvdp", "-r", "ref.mkv", "-t", "dist.mkv", *s.cvvdp()]
    for value in ("0", "-1", "none"):
        proc = s.run([*base, "--cvvdp-gpu-mem", value])
        if expect(f"cvvdp-gpu-mem-{value}", proc, 2,
                  needles=["--cvvdp-gpu-mem expects a positive number of GB"]) and not s.backend_log():
            ok(f"cvvdp-gpu-mem-{value}-rejected")
    proc = s.run([*base, "--cvvdp-gpu-mem", "1.5"])
    if expect("cvvdp-gpu-mem-forward", proc, 0):
        args = s.backend_args("cvvdp")
        if "--gpu-mem" not in args or args[args.index("--gpu-mem") + 1] != "1.5":
            fail("cvvdp-gpu-mem-forward", str(args))
        else:
            ok("cvvdp-gpu-mem-forwarded")


def group_cases(s: Suite) -> None:
    """--metrics pq|hlg|sdr: recommended metrics, transfer, and CVVDP defaults."""
    proc = s.run(["--metrics", "pq", "--metrics", "vmaf", "-r", "ref.mp4", "-t", "dist.mp4"])
    if expect("group-pq-selects", proc, 2, needles=["--cvqahdr-repo"]) and not s.backend_log():
        if "requires --cvvdp-interpretation" in proc.stderr or "requires --model" in proc.stderr:
            fail("group-pq-selects", proc.stderr)
        else:
            ok("group-pq-selects-nr-and-fills-cvvdp")
    proc = s.run(["--metrics", "pq", "--metrics", "pq", "--hdr-transfer", "hlg", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-pq-transfer", proc, 2, needles=["--metrics pq uses pq; transfer hlg conflicts"]):
        ok("group-pq-rejects-hlg-transfer")
    hlg_ref, hlg_test = s.tmp / "hlg-ref.yuv", s.tmp / "hlg-test.yuv"
    write_yuv(hlg_ref, 2, 64, size=32)
    write_yuv(hlg_test, 2, 96, size=32)
    proc = s.run(["--metrics", "hlg", "-r", str(hlg_ref), "-t", str(hlg_test), "-s", "32x32",
                  "-f", "420p:10", "--fps", "60"])
    if expect("group-hlg", proc, 0,
              stdout_needles=["[ PSNR", "[ PU21-PSNR", "display:standard_hdr_hlg interpretation:hlg"],
              absent_stdout=["[ VMAF", "CVQA-HDR-NR"]):
        args = s.backend_args("cvvdp")
        if args[args.index("--interpretation") + 1] != "hlg" or \
                args[args.index("--display") + 1] != "standard_hdr_hlg":
            fail("group-hlg", str(args))
        else:
            ok("group-hlg-selects-pu21-and-cvvdp-hlg")
    proc = s.run(["--metrics", "hlg,vmaf", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-hlg-vmaf", proc, 2, needles=["excludes vmaf"]) and not s.backend_log():
        ok("group-hlg-rejects-vmaf")
    proc = s.run(["--metrics", "hlg,cvqa-hdr-nr", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-hlg-nr", proc, 2, needles=["excludes cvqa-hdr-nr"]) and not s.backend_log():
        ok("group-hlg-rejects-nr")
    proc = s.run(["--metrics", "hlg", "--cvvdp-display", "standard_hdr_pq", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-hlg-display", proc, 2, needles=["uses --cvvdp-display standard_hdr_hlg"]) \
            and not s.backend_log():
        ok("group-hlg-rejects-pq-display")
    proc = s.run(["--metrics", "sdr", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-sdr-display", proc, 2, needles=["--metrics sdr needs --cvvdp-display"]) \
            and not s.backend_log():
        ok("group-sdr-requires-display")
    proc = s.run(["--metrics", "sdr,pu21-psnr", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-sdr-pu21", proc, 2, needles=["excludes pu21-psnr"]) and not s.backend_log():
        ok("group-sdr-rejects-pu21")
    proc = s.run(["--metrics", "pq", *s.raw_pair(), *s.nr()])
    if proc.returncode == 2 and ("no libvmaf" in proc.stderr or
                                 "cvqa-hdr-nr supports compressed video only" in proc.stderr):
        if "requires --cvvdp" in proc.stderr or "requires --model" in proc.stderr:
            fail("group-pq-fills", proc.stderr)
        else:
            ok("group-pq-fills-cvvdp-and-hdr-vmaf-model")
    else:
        fail("group-pq-fills", f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    proc = s.run(["--metrics", "pq", "--hdr-primaries", "bt709", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-pq-primaries", proc, 2, needles=["uses BT.2020 primaries"]):
        ok("group-pq-rejects-bt709-primaries")
    proc = s.run(["--metrics", "psnr,xpsnr", *s.raw_pair()])
    if expect("individual-metrics", proc, 0, stdout_needles=["[ PSNR", "[ XPSNR"],
              absent_stdout=["[ SSIM", "[ CVVDP", "PU21", "[ VMAF"]):
        ok("comma-separated-metrics-unchanged")
    proc = s.run(["--metrics", "sdr,pq", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-combine", proc, 2, needles=["--metrics sdr and pq cannot be combined"]):
        ok("group-rejects-two-groups")
    proc = s.run(["--preset", "hdr", "--metrics", "hlg", "-r", "a.mp4", "-t", "b.mp4"])
    if expect("group-preset", proc, 2, needles=["--preset hdr cannot be combined with --metrics hlg"]):
        ok("group-rejects-preset-with-hlg")


def config_cases(s: Suite) -> None:
    """config/learned_metrics.local.toml: lookup, precedence, path resolution, errors."""
    cfg_dir = s.tmp / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    cfg = cfg_dir / "learned.toml"
    (cfg_dir / "repo").mkdir(exist_ok=True)
    (cfg_dir / "nr.pth").write_bytes(b"x")
    (cfg_dir / "nr.npy").write_bytes(b"x")
    cfg.write_text(f'# comment\n[learned]\npython = "{sys.executable}"\nadapter_path = "{s.mocks}"\n\n'
                   '[cvqa-hdr-nr]\nrepository = "./repo"  # relative to this file\n'
                   'checkpoint = "./nr.pth"\nprofile = "./nr.npy"\n', encoding="utf-8")
    nr_only = ["--metrics", "cvqa-hdr-nr", "-t", "dist.mp4"]
    proc = s.run([*nr_only, "--learned-config", str(cfg)], PYTHONPATH="")
    if expect("config-nr", proc, 0, needles=[f"learned-metric config {cfg}"],
              stdout_needles=[f'"quality_score":{NR_SCORE}']):
        args = s.backend_args("cvqa-hdr-nr")
        if args[args.index("--repo") + 1] != str(cfg_dir / "repo") or \
                args[args.index("--checkpoint") + 1] != str(cfg_dir / "nr.pth"):
            fail("config-nr", str(args))
        else:
            ok("config-supplies-python-adapter-and-nr-assets")
    proc = s.run([*nr_only], PIXVQA_LEARNED_CONFIG=str(cfg), PYTHONPATH="")
    if expect("config-env", proc, 0, needles=[f"learned-metric config {cfg}"]):
        ok("config-from-PIXVQA_LEARNED_CONFIG")
    other = s.tmp / "other-repo"
    other.mkdir(exist_ok=True)
    proc = s.run([*nr_only, "--learned-config", str(cfg), "--cvqahdr-repo", str(other)], PYTHONPATH="")
    if expect("config-cli-wins", proc, 0):
        args = s.backend_args("cvqa-hdr-nr")
        if args[args.index("--repo") + 1] != str(other):
            fail("config-cli-wins", str(args))
        else:
            ok("config-cli-option-overrides")
    proc = s.run([*nr_only, "--learned-config", str(cfg), "--cvqahdr-python", "/nonexistent/python"],
                 PYTHONPATH="")
    if expect("config-cli-python", proc, 2, needles=["/nonexistent/python"]):
        ok("config-cli-python-overrides")
    proc = s.run([*nr_only, "--learned-config", str(cfg)], PYTHONPATH="",
                 PIXVQA_CVQAHDR_PYTHON="/nonexistent/env-python")
    if expect("config-env-python", proc, 2, needles=["/nonexistent/env-python"]):
        ok("config-env-python-overrides-config")
    for name, text, needle in (
            ("config-unknown-key", '[learned]\npyhton = "x"\n', ":2: unknown key learned.pyhton"),
            ("config-unknown-section", '[vmaf]\nmodel = "x"\n', ":1: unknown section [vmaf]"),
            ("config-duplicate", '[learned]\npython = "a"\npython = "b"\n', ":3: duplicate key learned.python"),
            ("config-unquoted", '[learned]\npython = /usr/bin/python3\n', ":2: only quoted string values")):
        bad = cfg_dir / f"{name}.toml"
        bad.write_text(text, encoding="utf-8")
        proc = s.run([*nr_only, "--learned-config", str(bad)])
        if expect(name, proc, 2, needles=[needle]) and not s.backend_log():
            ok(name)
    proc = s.run([*nr_only, "--learned-config", str(cfg_dir / "missing.toml")])
    if expect("config-missing", proc, 2, needles=["missing.toml"]) and not s.backend_log():
        ok("config-explicit-missing-file-fails")
    proc = s.run(["--metrics", "psnr", *s.raw_pair(), "--learned-config", str(cfg)])
    if expect("config-native-only", proc, 2, needles=["--learned-config requires cvvdp or cvqa-hdr-nr"]):
        ok("config-rejected-without-learned-metric")
    proc = s.run(["--metrics", "psnr", *s.raw_pair()], PIXVQA_LEARNED_CONFIG=str(cfg_dir / "config-unknown-key.toml"))
    if expect("config-native-ignored", proc, 0, absent_stdout=["CVQA"]) and "learned-metric config" not in proc.stderr:
        ok("config-not-read-for-native-only")


def decode_cases(s: Suite) -> None:
    if not DECODE_BIN.is_file() or not FFMPEG:
        print("SKIP native+learned compressed cases (need PIXVQA_DECODE_BIN and ffmpeg)")
        return
    clips = {}
    for name, q in (("ref", 0), ("dist", 30)):
        path = s.tmp / f"{name}.mp4"
        made = subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
                               "-frames:v", "4", "-c:v", "libx264", "-qp", str(q), "-pix_fmt", "yuv420p",
                               str(path)], capture_output=True, text=True)
        if made.returncode != 0:
            print(f"SKIP native+learned compressed cases (ffmpeg: {made.stderr.strip()[-200:]})")
            return
        clips[name] = str(path)
    out = s.tmp / "native-nr.out"
    proc = s.run(["--metrics", "psnr,cvqa-hdr-nr", "-r", clips["ref"], "-t", clips["dist"], *s.nr()],
                 binary=DECODE_BIN, stdout_file=out)
    if expect("native-nr", proc, 0, stdout_needles=["[ PSNR", f"[ CVQA-HDR-NR quality_score:{NR_SCORE}"]):
        if s.backend_log() != ["cvqa-hdr-nr check", "cvqa-hdr-nr run"] or not s.native_before("cvqa-hdr-nr"):
            fail("native-nr", str(s.backend_log()))
        else:
            ok("native-nr-compressed")
    cvvdp_compressed = ["--cvvdp-interpretation", "sdr", "--cvvdp-display", "standard_fhd",
                        "--cvvdp-python", sys.executable]
    proc = s.run(["--metrics", "psnr,cvvdp,cvqa-hdr-nr", "--qcs", "--qcs-range", "cvqa-hdr-nr=7:8",
                  "-r", clips["ref"], "-t", clips["dist"], *cvvdp_compressed, *s.nr()],
                 binary=DECODE_BIN, stdout_file=s.tmp / "all.out")
    if expect("all-three", proc, 0, stdout_needles=["[ PSNR-QCS", "[ CVVDP-QCS value:", "range:[0,10] frames:2",
                                                     "[ CVQA-HDR-NR-QCS value:40.000000"]):
        order = s.backend_log()
        if order != ["cvvdp check", "cvqa-hdr-nr check", "cvvdp run", "cvqa-hdr-nr run"] \
                or not s.native_before("cvvdp"):
            fail("all-three", str(order))
        else:
            ok("all-three-compressed")
    raw_ref = s.tmp / "mixed-ref.yuv"
    write_yuv(raw_ref, 4, 64, size=64, bits=8)
    proc = s.run(["--metrics", "psnr,cvvdp", "-r", str(raw_ref), "-s", "64x64", "-f", "420p:8", "--fps", "24",
                  "-t", clips["dist"], *s.cvvdp()], binary=DECODE_BIN)
    if expect("cvvdp-mixed-input", proc, 2, needles=["both inputs as raw YUV or both as compressed"]) \
            and not s.backend_log():
        ok("cvvdp-raw-compressed-mix-rejected")


def official_logistic(x: float, b1: float, b2: float, b3: float, b4: float) -> float:
    """VQA_NR.logistic_func, written with math instead of numpy."""
    return b2 + (b1 - b2) / (1 + math.exp(-((x - b3) / abs(b4))))


def device_unit_cases() -> None:
    from types import SimpleNamespace

    from pixvqa_cvvdp.adapter import logged_device
    from pixvqa_device import DeviceError, auto_notice, device_record, resolve_device

    def fake_torch(available: bool):
        return SimpleNamespace(__version__="0.0+stub", version=SimpleNamespace(cuda="12.8"),
                               get_num_threads=lambda: 4,
                               cuda=SimpleNamespace(is_available=lambda: available))

    with_cuda, without = fake_torch(True), fake_torch(False)
    resolved = [resolve_device(r, t) for r, t in (("auto", with_cuda), ("auto", without),
                                                   ("cpu", with_cuda), ("cuda", with_cuda))]
    if resolved != ["cuda", "cpu", "cpu", "cuda"]:
        fail("device-resolve", str(resolved))
    else:
        ok("device-resolve-auto-cpu-cuda")
    for requested, needle in (("cuda", "not falling back to CPU"), ("gpu", "use auto, cpu, or cuda"),
                              ("mps", "use auto, cpu, or cuda")):
        try:
            resolve_device(requested, without)
        except DeviceError as exc:
            if needle not in str(exc):
                fail(f"device-reject-{requested}", str(exc))
                continue
        else:
            fail(f"device-reject-{requested}", "accepted")
            continue
        ok(f"device-reject-{requested}")
    if auto_notice("auto", "cpu") is None or auto_notice("auto", "cuda") or auto_notice("cpu", "cpu"):
        fail("device-auto-notice", "notice only when auto selects cpu")
    else:
        ok("device-auto-notice")
    record = device_record("auto", "cpu", without)
    if record["effective"] != "cpu" or record["torch_version"] != "0.0+stub" or "gpu" in record:
        fail("device-record", str(record))
    else:
        ok("device-record")
    logs = ("[INFO] Running on device: cuda\n", "x\n[INFO] Running on device: cpu\n", "no log")
    if [logged_device(text) for text in logs] != ["cuda", "cpu", None]:
        fail("cvvdp-logged-device", str([logged_device(text) for text in logs]))
    else:
        ok("cvvdp-logged-device")


def adapter_cases() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from pixvqa_cvqahdr import __main__ as nr_main
    from pixvqa_cvqahdr.adapter import (AdapterError, PreflightError, calibrate, sample_schedule,
                                        sampling_plan, temporal_record)
    device_unit_cases()

    plan = sampling_plan(120, 60000 / 1001)
    schedule = sample_schedule(plan)
    if [e["frame_index"] for e in schedule] != [30, 90] or schedule[1]["time_seconds"] != 90 / (60000 / 1001):
        fail("sample-schedule", str(schedule))
    else:
        ok("sample-schedule-indices-and-timestamps")
    plan24 = sampling_plan(60, 24)
    if [e["frame_index"] for e in sample_schedule(plan24)] != [12, 36]:
        fail("sample-schedule-24", str(sample_schedule(plan24)))
    for frames, needle in ((50, "at least"), (100, "sampling-buffer")):
        try:
            sampling_plan(frames, 60)
        except PreflightError as exc:
            if needle not in str(exc):
                fail(f"sampling-{frames}", str(exc))
        else:
            fail(f"sampling-{frames}", "unsafe clip accepted")
    ok("sampling-overflow-and-short-clip-are-preflight-errors")

    profile = (9.2, 1.1, -0.9, 0.35)
    raw = [-1.8, -0.1, 0.4]
    calibrated = calibrate(raw, profile, official_logistic)
    pooled = official_logistic(sum(raw) / len(raw), *profile)
    mean_calibrated = sum(calibrated) / len(calibrated)
    if calibrated != [official_logistic(r, *profile) for r in raw] or abs(pooled - mean_calibrated) < 1e-3:
        fail("nonlinear-calibration", f"{calibrated} pooled={pooled} mean={mean_calibrated}")
    else:
        ok("per-sample-calibration-is-nonlinear-pooled-score-unchanged")
    try:
        import numpy as np
    except ImportError:
        print("SKIP numpy logistic parity (numpy is absent)")
    else:
        def numpy_logistic(X, b1, b2, b3, b4):
            logistic_part = 1 + np.exp(np.negative(np.divide(X - b3, np.abs(b4))))
            return b2 + np.divide(b1 - b2, logistic_part)
        array_profile = np.asarray(profile, dtype=np.float64)
        if calibrate(raw, array_profile, numpy_logistic) != [float(numpy_logistic(r, *array_profile)) for r in raw]:
            fail("numpy-logistic", "per-sample calibration diverged")
        else:
            ok("numpy-official-logistic-per-sample")

    record = temporal_record(plan, raw[:2], calibrated[:2], "float64", True, 120)
    if record["padded_samples"] != 0 or not record["qcs_eligible"] or record["sample_count"] != 2 \
            or record["basis"] != "sampled-frames" or abs(record["sampling_density_hz"] - 0.999000999) > 1e-9:
        fail("temporal-record", json.dumps(record))
    padded = temporal_record(plan, raw[:2], calibrated[:2], "float64", True, 60)
    if padded["padded_samples"] != 1 or padded["qcs_eligible"] or not padded["samples"][1]["padded"]:
        fail("temporal-padded", json.dumps(padded))
    try:
        temporal_record(plan, raw, calibrated, "float64", True, None)
    except AdapterError:
        ok("temporal-record-metadata-padding-and-count")
    else:
        fail("temporal-count", "sample count mismatch accepted")

    fake = {
        "quality_score": float(NR_SCORE), "raw_model_score": float(NR_RAW), "device": "cpu",
        "sampling": {**plan},
        "temporal": {**record, "samples": record["samples"]},
    }
    lines = nr_main.protocol_lines(fake, None)
    values = dict(line.split(" ", 2)[1:] for line in lines if line.startswith("pixvqa-value "))
    if values["quality_score"] != NR_SCORE or values["raw_model_score"] != NR_RAW \
            or float(lines[-2].split()[5]) != calibrated[1] or lines[-1] != "pixvqa-end":
        fail("protocol-lines", "\n".join(lines))
    else:
        ok("protocol-full-precision")

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "clip.mp4"
        video.write_bytes(b"")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "python")
        proc = subprocess.run([sys.executable, "-m", "pixvqa_cvqahdr", "--check", "--test", str(video),
                               "--repo", tmp, "--checkpoint", str(Path(tmp) / "missing.pth"),
                               "--profile", str(Path(tmp) / "missing.npy"), "--device", "cpu",
                               "--output-json", str(Path(tmp) / "out.json")],
                              capture_output=True, text=True, env=env)
        if proc.returncode != 2 or "checkpoint not found" not in proc.stderr or proc.stdout:
            fail("adapter-check-assets", f"{proc.returncode} {proc.stdout!r} {proc.stderr!r}")
        else:
            ok("real-adapter-check-rejects-missing-model-exit-2")

    try:
        import torch
    except ImportError:
        print("SKIP quality1 hook parity (PyTorch is absent)")
        return
    from pixvqa_cvqahdr.adapter import forward_with_samples

    class Tiny(torch.nn.Module):
        """Same pooling as NR_model.SigLIP2_384_multi_dataset.forward for quality1."""

        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(3)
            self.features = torch.nn.Linear(12, 8)
            self.quality1 = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 1))

        def forward(self, x):
            size = x.shape
            x = x.view(-1, size[2] * size[3] * size[4])
            x = self.features(x)
            x1 = self.quality1(x)
            x1 = x1.view(size[0], size[1])
            return torch.mean(x1, dim=1), x1

    model = Tiny().eval()
    batch = torch.randn(1, 5, 3, 2, 2)
    with torch.no_grad():
        reference_outputs = model(batch)
    outputs, raw_score, raw_values, parity = forward_with_samples(model, batch, torch)
    if (not parity or raw_score != reference_outputs[0].item()
            or raw_values != reference_outputs[1][0].tolist() or model.quality1._forward_hooks):
        fail("quality1-hook", f"{raw_score} {raw_values} {parity}")
    else:
        ok("quality1-hook-float32-samples-pooled-parity")


def main() -> int:
    adapter_cases()
    if not PIXVQA.is_file():
        print(f"missing pixvqa: {PIXVQA}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        suite = Suite(Path(tmp))
        cli_cases(suite)
        decode_cases(suite)
    if FAILS:
        print(f"{FAILS} failure(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
