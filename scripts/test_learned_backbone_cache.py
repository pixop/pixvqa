#!/usr/bin/env python3
"""Offline test of pinned prefetch, cache reuse and corrupt-cache rejection."""
import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'python'))
from pixvqa_learned_install.backbones import ensure_slowfast, SLOWFAST_NAME

with tempfile.TemporaryDirectory() as td:
    base = Path(td)
    payload = b'PK' + b'fake pretrained data' * 100
    src = base / 'source.pyth'
    src.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    cache = base / 'hub' / 'checkpoints'
    try:
        ensure_slowfast(cache, offline=True, expected=expected)
    except RuntimeError as e:
        assert 'not cached' in str(e)
    else:
        raise AssertionError('missing cache accepted offline')
    dst, installed = ensure_slowfast(cache, url=src.as_uri(), expected=expected)
    assert installed and dst.name == SLOWFAST_NAME and dst.read_bytes() == payload
    dst2, installed = ensure_slowfast(cache, offline=True, expected=expected)
    assert not installed and dst2 == dst
    dst.write_bytes(b'bad')
    try:
        ensure_slowfast(cache, offline=False, url=src.as_uri(), expected=expected)
    except RuntimeError as e:
        assert 'SHA-256 mismatch' in str(e)
    else:
        raise AssertionError('corrupt cache overwritten or accepted')
    dst.unlink()
    try:
        ensure_slowfast(cache, url=src.as_uri(), expected='a'*64)
    except RuntimeError as e:
        assert 'downloaded SHA-256 mismatch' in str(e)
    else:
        raise AssertionError('corrupt download accepted')
    assert not dst.exists()
    assert not list(cache.glob('*.part'))
print('PASS pinned backbone download, offline cache, corruption rejection and cleanup')
