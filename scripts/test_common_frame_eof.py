#!/usr/bin/env python3
"""Regression: reference/test evaluate their common decodable prefix."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'build/cpu/pixvqa'
FRAME = bytes(64 * 64 * 3)  # 64x64 planar 4:2:0, 10-bit little-endian


def evaluate(ref_frames: int, test_frames: int, cap: int | None = None):
    with tempfile.TemporaryDirectory(prefix='pixvqa-eof-') as d:
        ref = Path(d) / 'ref.yuv'
        test = Path(d) / 'test.yuv'
        ref.write_bytes(FRAME * ref_frames)
        test.write_bytes(FRAME * test_frames)
        args = [str(BIN), '--metrics', 'psnr', '-r', str(ref), '-t', str(test),
                '-s', '64x64', '-f', '420p:10', '-v', '0']
        if cap is not None:
            args += ['--frames', str(cap)]
        return subprocess.run(args, text=True, capture_output=True, timeout=20)


def test(ref_frames, test_frames, cap, expected, mismatch):
    result = evaluate(ref_frames, test_frames, cap)
    assert result.returncode == 0, (result.returncode, result.stderr, result.stdout)
    assert f'Total frames:{expected}' in result.stdout, result.stdout
    assert ('reference/test lengths differ' in result.stderr) == mismatch, result.stderr
    print(f'OK ref={ref_frames} test={test_frames} cap={cap} -> {expected}')


def main():
    test(2, 2, None, 2, False)
    test(501, 500, None, 500, True)  # BasketballDrive frame-count scenario
    test(500, 501, None, 500, True)
    test(2, 1, None, 1, True)
    test(1, 2, None, 1, True)
    test(2, 1, 5, 1, True)
    test(1, 2, 5, 1, True)
    test(2, 1, 1, 1, False)  # cap reached without reading extra ref frame
    result = evaluate(0, 1)
    assert result.returncode != 0 and 'no frames read' in result.stderr, result.stderr
    print('OK empty input rejected')


if __name__ == '__main__':
    main()
