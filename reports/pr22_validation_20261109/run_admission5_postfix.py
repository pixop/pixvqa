#!/usr/bin/env python3
"""admission_5 only — post STEP 1 profile/socket fixes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
CVQA_DIR = REPORT / "cvqa_concurrency"
OUT = CVQA_DIR / "admission5_postfix.json"

NESTED = CVQA_DIR / ".admission_daemon_nested_long"


def main() -> int:
    nested = NESTED
    nested.mkdir(parents=True, exist_ok=True)
    nested_s = str(nested)
    code = f"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / "python")!r})
nested = Path({nested_s!r})
os.environ["PIXVQA_SCHEDULER_DIR"] = str(nested)
os.chdir({str(CVQA_DIR)!r})
import run_cvqa_concurrency as c
from pixvqa_scheduler.ipc import socket_address
sock = socket_address()
observed = {{}}
stage = c.run_stage(
    label="admission_5_post_fix",
    n_jobs=5,
    admission=True,
    observed=observed,
    scheduler_dir=nested,
)
walls = [j.get("wall_s", 0) for j in stage.get("jobs", [])]
spread = (max(walls) - min(walls)) if walls else 999
payload = {{
    "socket_path": sock,
    "socket_len": len(sock),
    "stage": stage,
    "five_concurrent_no_queue": spread < 10,
    "job_wall_spread_s": round(spread, 2),
}}
print(json.dumps(payload))
"""
    proc = subprocess.run(
        ["/home/sergio/pixvqa-compressedvqa-env/bin/python", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    ok = payload["stage"].get("all_succeeded") and payload.get("five_concurrent_no_queue")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
