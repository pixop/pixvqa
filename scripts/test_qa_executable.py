#!/usr/bin/env python3
"""Executable selection for QA after installation and during development."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_qa import select_qa_executable  # noqa: E402


def fail(message: str) -> None:
    print("FAIL", message, file=sys.stderr)
    raise SystemExit(1)


def write_exec(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pixvqa_qa_select_") as tmp:
        root = Path(tmp)
        try:
            select_qa_executable(root)
            fail("missing executable was accepted")
        except FileNotFoundError as exc:
            if "bin/pixvqa" not in str(exc) or "build/cpu/pixvqa" not in str(exc):
                fail(f"unclear missing-executable error: {exc}")
        developed = root / "build" / "cpu" / "pixvqa"
        write_exec(developed, "#!/bin/sh\nprintf 'dev\\n'\n")
        if select_qa_executable(root) != developed:
            fail("development executable was not selected")
        installed = root / "bin" / "pixvqa"
        write_exec(installed, "#!/bin/sh\nprintf 'installed\\n'\n")
        if select_qa_executable(root) != installed:
            fail("installed executable did not replace the development binary")
        other = root / "other" / "pixvqa"
        write_exec(other, "#!/bin/sh\nprintf 'override\\n'\n")
        if select_qa_executable(root, str(other)) != other:
            fail("explicit override was ignored")
        try:
            select_qa_executable(root, str(root / "missing"))
            fail("missing override was accepted")
        except FileNotFoundError:
            pass
        print("OK   executable selection")

    source = ROOT / "build" / "cpu" / "pixvqa"
    if not source.is_file():
        print("SKIP post-install version: build/cpu/pixvqa is absent")
        return 0
    with tempfile.TemporaryDirectory(prefix="pixvqa_qa_installed_") as tmp:
        root = Path(tmp)
        installed = root / "bin" / "pixvqa"
        installed.parent.mkdir()
        installed.write_bytes(source.read_bytes())
        installed.chmod(0o755)
        # A development binary in the same tree must not win.
        write_exec(root / "build" / "cpu" / "pixvqa", "#!/bin/sh\nexit 1\n")
        chosen = select_qa_executable(root)
        if chosen != installed:
            fail(f"post-install selection returned {chosen}")
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        done = subprocess.run(
            [str(chosen), "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        if done.returncode != 0 or "pixvqa version" not in done.stdout:
            fail(done.stderr or done.stdout or "version failed")
        if "build:" not in done.stdout or "decode:" not in done.stdout:
            fail(f"version did not name the build profile:\n{done.stdout}")
        print("OK   post-install executable runs without LD_LIBRARY_PATH")
        print(done.stdout.strip())
    print("All QA executable checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
