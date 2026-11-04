#!/usr/bin/env python3
"""Offline integration-campaign manifest, runner, resume, blocked-state tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_real_video_integration.py"
sys.path.insert(0, str(RUNNER.parent))
from run_real_video_integration import manifest, parse_metrics, command  # noqa: E402


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(RUNNER), *args],
                          text=True, capture_output=True, timeout=30)


def main() -> None:
    items = manifest()
    assert Counter(c.phase for c in items) == dict(zip("ABCDEFG", (48,12,24,18,18,12,14)))
    assert items[0].id == "A001_sdr_yuv_to_yuv"
    assert items[47].id == "A048_hlg_vvc_to_vvc"
    assert items[-1].id.startswith("G014_hlg_")
    assert all(c.definition == "historical" for c in items[:48])
    assert all(c.definition == "proposed" for c in items[48:])
    assert len({c.id for c in items}) == 146
    assert parse_metrics('[ PSNR y:32.1 u:34 v:35 avg:33.2 ]','psnr')['psnr_avg'] == 33.2
    assert parse_metrics('[ VMAF mean:95.111111 harmonic:94 min:90 max:99 frames:3 ]','vmaf') == {'vmaf': 95.111111}
    assert parse_metrics('[ XPSNR y:inf u:inf v:inf avg:inf min:inf ]','xpsnr') == {'xpsnr': 'inf'}
    with tempfile.TemporaryDirectory(prefix="pixvqa_integration_runner_") as tmp:
        p = Path(tmp)
        fixture = p/'data'/'sdr_ref_2160p_420p10.yuv'
        fixture.parent.mkdir()
        fixture.write_bytes(b'fixture metadata only')
        executable = p/'fake-pixvqa'
        executable.write_text('#!/bin/sh\necho "[ PSNR y:42.01 u:43 v:44 avg:43.0 ]"\necho "Total frames:3  fps:3.00"\n')
        executable.chmod(0o755)
        report_dir = p/'results'
        opts = ('--exe',str(executable),'--data',str(fixture.parent),
                '--output',str(report_dir),'--case-id',items[0].id)
        p0 = run(*opts)
        assert p0.returncode == 0, p0.stderr + p0.stdout
        report = json.loads(next(report_dir.glob('selected_1_*_results.json')).read_text())
        assert report['counts']['PASS'] == 1 and report['complete'] and not report['complete_full_campaign']
        p1 = run(*opts,'--resume')
        assert p1.returncode == 0 and '(resumed)' in p1.stdout, p1.stderr + p1.stdout
        # Invalidate stamp: previously passing case must be rerun and fail.
        executable.write_text('#!/bin/sh\necho "broken result"\nexit 1\n')
        p2 = run(*opts,'--resume')
        assert p2.returncode == 1 and 'FAIL' in p2.stdout
        result = json.loads(next(report_dir.glob('selected_1_*_results.json')).read_text())
        assert result['counts']['FAIL'] == 1 and not result['complete']
        # Missing second fixture must be reported as BLOCKED, never PASS.
        cross = next(c for c in items if c.ref_format != c.test_format)
        blocked = run('--exe',str(executable),'--data',str(fixture.parent),
                      '--output',str(p/'other'),'--case-id',cross.id)
        assert blocked.returncode == 1 and 'BLOCKED' in blocked.stdout
        records = json.loads(next((p/'other').glob('selected_1_*_results.json')).read_text())
        assert records['counts']['BLOCKED'] == 1 and records['counts']['PASS'] == 0
        # F parity uses a single device selection (no cpu/gpu conflict).
        parity = next(c for c in items if c.phase == 'F')
        args = command(parity,executable,fixture.parent,3,'gpu',ROOT)
        assert args.count('--device') == 1 and args[args.index('--device')+1] == 'gpu'
        assert '--metric-device' not in args
    print('PASS 146-case manifest, original A order, F device flag, PASS/FAIL/BLOCKED, resume invalidation')


if __name__ == '__main__':
    main()
