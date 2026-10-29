#!/usr/bin/env python3
"""CLI regressions: zero frames, short frames, missing files, one valid frame, empty FIFO."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "cpu" / "pixvqa"
W, H = 64, 64
FRAME_BYTES = (W * H + 2 * (W // 2) * (H // 2)) * 2  # 10-bit 4:2:0


def run(args, timeout=20):
    return subprocess.run(
        [str(BIN), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def fail(msg):
    print("FAIL", msg, file=sys.stderr)
    return 1


def assert_not_a_score(done, label):
    text = done.stdout
    for needle in ("Summary results", "Total frames:", "[ PSNR", "[ MS-SSIM", "[ SSIM"):
        if needle in text:
            return fail(f"{label}: stdout looks like a score ({needle!r})")
    if done.returncode == 0:
        return fail(f"{label}: expected nonzero status, got 0")
    return 0


def test_help():
    done = run(["--help"])
    text = done.stdout + done.stderr
    if done.returncode != 0:
        return fail(f"help status {done.returncode}")
    if "msssim" not in text or "docs/QCS.md" not in text:
        return fail("help does not list msssim and docs/QCS.md")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "five-scale" not in readme:
        return fail("README does not identify five-scale MS-SSIM")
    print("OK   help")
    return 0


def test_prod_style_cli(path_ref: Path, path_test: Path):
    """Repeated -s/-f after -r and -t, with --cs/--tf, -n alias, default kernel/strip."""
    blob = bytes(FRAME_BYTES)
    path_ref.write_bytes(blob)
    path_test.write_bytes(blob)
    done = run([
        "--cs", "bt2020l", "--tf", "st2084",
        "-r", str(path_ref), "-s", f"{W}x{H}", "-f", "420p:10",
        "-t", str(path_test), "-s", f"{W}x{H}", "-f", "420p:10",
        "--metrics", "psnr,ssim", "--threads", "8", "-v", "1", "-n", "1"])
    if done.returncode != 0:
        return fail(f"prod-style status {done.returncode} stderr={done.stderr}")
    out = done.stdout
    if "Input: 64x64 yuv420 (10b,10b,10b) bt2020l st2048" not in out:
        return fail(f"prod-style Input line mismatch: {out}")
    if "0  :[ PSNR" not in out or "[ SSIM" not in out:
        return fail(f"prod-style frame line mismatch: {out}")
    if "Summary results:" not in out or "Total frames:1" not in out:
        return fail(f"prod-style summary mismatch: {out}")
    if "kernel=" in out or "strip=" in out:
        return fail("prod-style leaked kernel/strip into Input line")
    print("OK   prod-style-cli")
    return 0


def test_mismatch_size(path_ref: Path, path_test: Path):
    path_ref.write_bytes(bytes(FRAME_BYTES))
    path_test.write_bytes(bytes(FRAME_BYTES))
    done = run([
        "-r", str(path_ref), "-s", f"{W}x{H}", "-f", "420p:10",
        "-t", str(path_test), "-s", "128x128", "-f", "420p:10",
        "--metrics", "psnr", "-v", "0"])
    if done.returncode == 0:
        return fail("mismatch-size expected nonzero")
    if "disagree" not in done.stderr:
        return fail(f"mismatch-size stderr={done.stderr!r}")
    if "Summary results" in done.stdout or "Total frames:" in done.stdout:
        return fail("mismatch-size printed a score summary")
    print("OK   mismatch-size")
    return 0


def test_unsupported_wsssim():
    done = run([
        "--wsssim", "-r", "/x", "-t", "/y", "-s", f"{W}x{H}", "-f", "420p:10"])
    if done.returncode == 0:
        return fail("wsssim expected nonzero")
    if "unsupported" not in done.stderr:
        return fail(f"wsssim stderr={done.stderr!r}")
    print("OK   unsupported-wsssim")
    return 0


def test_missing():
    done = run([
        "--metrics", "psnr,ssim", "-r", "/no/such_ref.yuv", "-t", "/no/such_test.yuv",
        "-s", f"{W}x{H}", "-f", "420p:10", "-v", "0"])
    err = assert_not_a_score(done, "missing")
    if err:
        return err
    if "cannot open reference file '/no/such_ref.yuv': No such file or directory" not in done.stderr:
        return fail(f"missing: stdout={done.stdout!r} stderr={done.stderr!r}")
    print("OK   missing")
    return 0


def test_missing_compressed(path_test: Path):
    done = run([
        "--metrics", "psnr", "-r", "/no/such_ref.av1", "-t", str(path_test), "-v", "0"])
    err = assert_not_a_score(done, "missing-compressed")
    if err:
        return err
    if "cannot open reference file '/no/such_ref.av1': No such file or directory" not in done.stderr:
        return fail(f"missing-compressed: stderr={done.stderr!r}")
    if "raw YUV and requires" in done.stderr:
        return fail(f"missing-compressed fell back to raw: {done.stderr!r}")
    print("OK   missing-compressed")
    return 0


def test_short(path_ref: Path, path_test: Path):
    path_ref.write_bytes(b"\x00" * 100)
    path_test.write_bytes(b"\x00" * 100)
    done = run([
        "--metrics", "psnr,ssim", "-r", str(path_ref), "-t", str(path_test),
        "-s", f"{W}x{H}", "-f", "420p:10", "-v", "1"])
    err = assert_not_a_score(done, "short")
    if err:
        return err
    if "short YUV frame" not in done.stderr:
        return fail(f"short: stderr={done.stderr!r}")
    print("OK   short")
    return 0


def test_empty_files(path_ref: Path, path_test: Path):
    path_ref.write_bytes(b"")
    path_test.write_bytes(b"")
    done = run([
        "--metrics", "ssim", "-r", str(path_ref), "-t", str(path_test),
        "-s", f"{W}x{H}", "-f", "420p:10", "-v", "0"])
    err = assert_not_a_score(done, "empty-file")
    if err:
        return err
    # main() returns 1 after printing the execution summary and before any score.
    # The banner names the metric ("SSIM: cpu"). A score would be "[ SSIM".
    if done.returncode != 1:
        return fail(f"empty-file: expected exit 1, got {done.returncode}")
    if "no frames read (empty input or FIFO closed before the first frame)" not in done.stderr:
        return fail(f"empty-file: stderr={done.stderr!r}")
    if "SSIM: cpu" not in done.stdout:
        return fail(f"empty-file: missing execution banner\n{done.stdout}")
    if "Summary results" in done.stdout or "[ SSIM" in done.stdout:
        return fail("empty-file: stdout contains a score")
    print("OK   empty-file")
    return 0


def test_one_frame(path_ref: Path, path_test: Path):
    blob = bytes(FRAME_BYTES)  # identical zeros
    path_ref.write_bytes(blob)
    path_test.write_bytes(blob)
    done = run([
        "--metrics", "psnr,ssim", "--threads", "2",  
        "-r", str(path_ref), "-t", str(path_test),
        "-s", f"{W}x{H}", "-f", "420p:10", "-v", "1"])
    if done.returncode != 0:
        return fail(f"one-frame status {done.returncode} stderr={done.stderr}")
    if "Total frames:1" not in done.stdout:
        return fail(f"one-frame missing Total frames: {done.stdout}")
    if "[ SSIM" not in done.stdout or "[ PSNR" not in done.stdout:
        return fail("one-frame missing metric labels")
    if "avg:1.000000" not in done.stdout:
        return fail(f"one-frame expected identical SSIM 1: {done.stdout}")
    if "999.99" not in done.stdout:
        return fail(f"one-frame expected identical PSNR 999.99: {done.stdout}")
    print("OK   one-frame")
    return 0


def test_empty_fifo():
    with tempfile.TemporaryDirectory(prefix="pixvqa_fifo_") as tmp:
        ref = os.path.join(tmp, "ref.fifo")
        test = os.path.join(tmp, "test.fifo")
        os.mkfifo(ref)
        os.mkfifo(test)
        proc = subprocess.Popen(
            [str(BIN), "--metrics", "ssim", "-r", ref, "-t", test, "-s", f"{W}x{H}", "-f", "420p:10", "-v", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            # Opens block until pixvqa opens the read end, then close with no bytes.
            with open(ref, "wb") as rf, open(test, "wb") as tf:
                pass
            stdout, stderr = proc.communicate(timeout=15)
        except Exception as exc:
            proc.kill()
            proc.wait(timeout=5)
            return fail(f"empty-fifo exception {exc}")
        done_code = proc.returncode
        if done_code == 0:
            return fail(f"empty-fifo status 0 stdout={stdout!r} stderr={stderr!r}")
        if "no frames read" not in stderr:
            return fail(f"empty-fifo stderr={stderr!r}")
        for needle in ("Summary results", "Total frames:", "[ SSIM", "[ MS-SSIM"):
            if needle in stdout:
                return fail(f"empty-fifo stdout looks like a score ({needle!r})")
    print("OK   empty-fifo")
    return 0


def main():
    if not BIN.exists():
        print(f"missing {BIN}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="pixvqa_io_") as tmp:
        ref = Path(tmp) / "ref.yuv"
        test = Path(tmp) / "test.yuv"
        rc = 0
        rc |= test_help()
        rc |= test_missing()
        rc |= test_missing_compressed(test)
        rc |= test_short(ref, test)
        rc |= test_empty_files(ref, test)
        rc |= test_one_frame(ref, test)
        rc |= test_prod_style_cli(ref, test)
        rc |= test_mismatch_size(ref, test)
        rc |= test_unsupported_wsssim()
        rc |= test_empty_fifo()
    if rc:
        print("IO regressions failed", file=sys.stderr)
        return 1
    print("All IO regressions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
