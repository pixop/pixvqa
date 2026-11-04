#!/usr/bin/env python3
"""Strict CVVDP metadata and mixed-input frame-count contract tests (CPU only)."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def expect_error(fn, message: str) -> None:
    try:
        fn()
    except Exception as exc:
        assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError(f'expected failure containing {message!r}')


def meta(count: str | None) -> dict:
    return {
        'width': 16, 'height': 16, 'r_frame_rate': '24/1',
        'nb_frames': count, 'color_transfer': 'smpte2084',
        'color_primaries': 'bt2020', 'color_space': 'bt2020nc',
        'color_range': 'tv',
    }


def test_adapter_frame_policy() -> None:
    from pixvqa_cvvdp.adapter import check_pair
    check_pair(meta('20'), meta('20'), 'pq', 0)
    check_pair(meta('20'), meta('12'), 'pq', 10)
    for cap in (0, -1):
        expect_error(lambda: check_pair(meta('20'), meta('12'), 'pq', cap), 'frame count mismatch')
    expect_error(lambda: check_pair(meta('20'), meta('12'), 'pq', 13), 'test has 12 frames')
    expect_error(lambda: check_pair(meta('12'), meta('20'), 'pq', 13), 'reference has 12 frames')
    # Unknown compressed frame counts must be handled by the reader at runtime.
    check_pair(meta(None), meta('20'), 'pq', 0)
    check_pair(meta(None), meta('20'), 'pq', 10)
    print('OK   cvvdp-adapter-frame-count-policy')


def test_adapter_transfer_tags() -> None:
    from pixvqa_cvvdp.adapter import check_pair
    tagged = meta('12')
    untagged = dict(tagged, color_transfer='unspecified')
    mismatched = dict(tagged, color_transfer='arib-std-b67')
    for reference, test in ((tagged, untagged), (untagged, tagged)):
        expect_error(lambda: check_pair(reference, test, 'pq', 0), 'not explicitly limited-range BT.2020 NCL PQ')
    expect_error(lambda: check_pair(tagged, mismatched, 'pq', 0), 'not explicitly limited-range BT.2020 NCL PQ')
    cl = dict(tagged, color_space='bt2020c')
    expect_error(lambda: check_pair(cl, tagged, 'pq', 0), 'no validated CL-to-NCL')
    expect_error(lambda: check_pair(tagged, cl, 'pq', 0), 'no validated CL-to-NCL')
    ncl_full = dict(tagged, color_range='pc')
    expect_error(lambda: check_pair(tagged, ncl_full, 'pq', 0), 'limited-range')
    print('OK   cvvdp-strict-per-side-transfer-tags-and-matrix')


def test_mixed_reader_length_policy() -> None:
    # Inject lightweight modules: importing the actual pycvvdp/torch model is
    # unnecessary to exercise reader count handling.
    class Base:
        def __init__(self, **kwargs):
            pass

    class Reader:
        def __init__(self, path, *args, **kwargs):
            self.frames = counts[path]
            self.avg_fps = 24.0
            self.width = 16
            self.height = 16

    fake_pycvvdp = types.ModuleType('pycvvdp')
    fake_source = types.ModuleType('pycvvdp.video_source')
    fake_source.reshuffle_dims = lambda *a, **kw: None
    fake_source.video_source_dm = Base
    fake_yuv = types.ModuleType('pycvvdp.video_source_yuv')
    fake_yuv.YUVReader = Reader
    fake_exc = types.ModuleType('pycvvdp.vq_exception')
    fake_exc.vq_exception = RuntimeError
    fake_file = types.ModuleType('pycvvdp.video_source_file')
    fake_file.video_reader = Reader
    injected = {
        'pycvvdp': fake_pycvvdp, 'pycvvdp.video_source': fake_source,
        'pycvvdp.video_source_yuv': fake_yuv,
        'pycvvdp.vq_exception': fake_exc,
        'pycvvdp.video_source_file': fake_file,
    }
    counts = {'reference.yuv': 20, 'test.mkv': 12}
    with patch.dict(sys.modules, injected):
        sys.modules.pop('pixvqa_cvvdp.mixed_video_source', None)
        mod = importlib.import_module('pixvqa_cvvdp.mixed_video_source')
        def src(n):
            return mod.PixvqaMixedVideoSource(
                'test.mkv', 'reference.yuv', ref_is_raw=True, test_is_raw=False,
                display_photometry='standard_hdr_pq', frames=n, fps=24.0,
            )
        expect_error(lambda: src(-1).init_readers(), 'frame counts differ')
        expect_error(lambda: src(13).init_readers(), 'test has 12 frames')
        good = src(10)
        assert good.get_video_size() == (16, 16, 10)
        counts['reference.yuv'] = 12
        assert src(-1).get_video_size() == (16, 16, 12)
        counts['reference.yuv'] = -1
        assert src(-1).get_video_size() == (16, 16, 12)
        counts['test.mkv'] = -1
        expect_error(lambda: src(-1).init_readers(), 'Neither test nor reference')
        assert src(5).get_video_size() == (16, 16, 5)
        sys.modules.pop('pixvqa_cvvdp.mixed_video_source', None)
    print('OK   cvvdp-mixed-reader-frame-count-policy')


def test_mixed_output_uses_reader_count() -> None:
    api = (ROOT / 'python' / 'pixvqa_cvvdp' / 'python_api.py').read_text()
    assert '"frames_evaluated": int(vs.frames)' in api
    assert '"frames_evaluated": frames_evaluated(' not in api
    print('OK   cvvdp-mixed-result-frame-count-source')


if __name__ == '__main__':
    test_adapter_frame_policy()
    test_adapter_transfer_tags()
    test_mixed_reader_length_policy()
    test_mixed_output_uses_reader_count()
