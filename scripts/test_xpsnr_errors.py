#!/usr/bin/env python3
"""Error-path checks for --xpsnr. Does not compare metric values."""
import struct
import subprocess
import tempfile
from pathlib import Path

BIN = str(Path(__file__).resolve().parents[1] / "build/cpu/pixvqa")
fails = 0


def check(name, args, expect_rc, stderr_has=None, stdout_lacks="Summary results"):
    global fails
    p = subprocess.run([BIN, *args], capture_output=True, text=True)
    ok = p.returncode == expect_rc
    if stderr_has is not None and stderr_has not in p.stderr and stderr_has not in p.stdout:
        ok = False
    if stdout_lacks and stdout_lacks in p.stdout and expect_rc != 0:
        ok = False
    print(("OK  " if ok else "FAIL"), name, "rc", p.returncode)
    if not ok:
        fails += 1
        print(p.stdout[-200:])
        print(p.stderr[-300:])


def main():
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        w = h = 64
        frame = (w * h + 2 * (w // 2) * (h // 2)) * 2
        ref = t / "r.yuv"
        tst = t / "t.yuv"
        ref.write_bytes(b"\x00" * frame + b"\x01" * 10)
        tst.write_bytes(b"\x00" * frame)
        base = ["-s", "64x64", "-f", "420p:10"]
        check("truncated", ["--metrics", "xpsnr", "--fps", "60", "-r", str(ref), *base, "-t", str(tst), *base], 1, "short YUV")
        empty = t / "e.yuv"
        empty.write_bytes(b"")
        check("empty", ["--metrics", "xpsnr", "--fps", "60", "-r", str(empty), *base, "-t", str(empty), *base], 1, "no frames")
        missing = t / "nope.yuv"
        check("missing", ["--metrics", "xpsnr", "--fps", "60", "-r", str(missing), *base, "-t", str(missing), *base], 1, "failed to open")
        check("bd12", ["--metrics", "xpsnr", "--fps", "60", "-r", str(ref), "-s", "64x64", "-f", "420p:12", "-t", str(tst)], 2, "unsupported")
        check("badfps", ["--metrics", "xpsnr", "--fps", "nope", "-r", str(ref), *base, "-t", str(tst)], 2, "invalid --fps")
        check("nofps", ["--metrics", "xpsnr", "-r", str(ref), *base, "-t", str(tst)], 2, "requires --fps")
        # A complete one-frame file still scores, and -n 1 stops there.
        one = t / "one.yuv"
        blob = b"\x01\x02" * (frame // 2)
        one.write_bytes(blob)
        check("one-frame", ["--metrics", "xpsnr", "--fps", "60", "-r", str(one), *base, "-t", str(one), *base, "-n", "1"], 0, None, stdout_lacks="")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
