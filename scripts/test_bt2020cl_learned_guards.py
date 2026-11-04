#!/usr/bin/env python3
"""No-GPU adapter preflight tests: reject BT.2020 CL before learned inference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pixvqa_cvvdp.adapter import AdapterError, check_pair  # noqa: E402
from pixvqa_cvqahdr import adapter as nr  # noqa: E402


def expect_rejected(callback, reason: str) -> None:
    try:
        callback()
    except Exception as exc:
        assert reason in str(exc), str(exc)
    else:
        raise AssertionError(f'CL input not rejected: expected {reason!r}')


def main() -> None:
    good = {'width': 32, 'height': 32, 'r_frame_rate': '30/1',
            'nb_frames': '30', 'color_transfer': 'smpte2084',
            'color_primaries': 'bt2020', 'color_space': 'bt2020nc',
            'color_range': 'tv'}
    cl = dict(good, color_space='bt2020c')
    check_pair(good, good, 'pq', 0)
    expect_rejected(lambda: check_pair(cl, good, 'pq', 0), 'CL-to-NCL')
    expect_rejected(lambda: check_pair(good, cl, 'pq', 0), 'CL-to-NCL')
    for interpretation in ('hlg', 'sdr'):
        expect_rejected(lambda: check_pair(cl, good, interpretation, 0), 'CL-to-NCL')
    for matrix, should_fail in (('bt2020c', True), ('bt2020nc', False), ('unknown', False)):
        fake = SimpleNamespace(stdout=json.dumps({'streams':[{'color_space':matrix}]}))
        with patch.object(nr.subprocess, 'run', return_value=fake):
            if should_fail:
                expect_rejected(lambda: nr.reject_unsupported_cl('mock.mp4'), 'unsupported')
            else:
                nr.reject_unsupported_cl('mock.mp4')
    with patch.object(nr.subprocess, 'run', side_effect=FileNotFoundError('missing')):
        expect_rejected(lambda: nr.reject_unsupported_cl('mock.mp4'), 'could not verify')
    print('PASS CVVDP PQ/HLG/SDR CL guard; CVQA-HDR-NR CL/NCL/unknown and missing ffprobe preflight')


if __name__ == '__main__':
    main()
