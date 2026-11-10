#!/usr/bin/env python3
"""Regression: an isolated upstream demo can import PixVQA's FFmpeg module."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pixvqa_simplevqa.adapter import staged_demo_env  # noqa: E402


def main() -> int:
    original = os.environ.get("PYTHONPATH")
    try:
        # Avoid a false pass from a user-exported PYTHONPATH.
        os.environ.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="pixvqa_svqa_import_test_") as tmp:
            script = Path(tmp) / "test_demo.py"
            script.write_text(
                "from pixvqa_video.ffmpeg_bgr import FFmpegCapture\n"
                "print('staged-import-pass')\n", encoding="utf-8"
            )
            env = staged_demo_env()
            assert env["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "python")
            proc = subprocess.run(
                [sys.executable, str(script)], cwd=tmp, env=env,
                text=True, capture_output=True, check=False,
            )
            assert proc.returncode == 0, proc.stderr
            assert proc.stdout.strip() == "staged-import-pass", proc.stdout

        # Preserve caller additions after the required PixVQA module root.
        os.environ["PYTHONPATH"] = os.pathsep.join(["/tmp/custom-a", str(ROOT / "python"), "/tmp/custom-b"])
        paths = staged_demo_env()["PYTHONPATH"].split(os.pathsep)
        assert paths == [str(ROOT / "python"), "/tmp/custom-a", "/tmp/custom-b"], paths
        print("PASS: staged SimpleVQA import without exported PYTHONPATH")
        return 0
    finally:
        if original is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original


if __name__ == "__main__":
    raise SystemExit(main())
