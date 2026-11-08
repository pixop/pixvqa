"""Pin runtime-downloaded auxiliary weights before metric inference.

SimpleVQA calls pytorchvideo.models.hub.slowfast_r50(pretrained=True), which
uses torch.hub.get_dir()/checkpoints/SLOWFAST_8x8_R50.pyth.  Warm this exact
cache location during installation, rather than fetching during a benchmark.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

SLOWFAST_NAME = 'SLOWFAST_8x8_R50.pyth'
SLOWFAST_URL = 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth'
SLOWFAST_SHA256 = '454f39e1c1f985df2bee2aa27887ed53ff56e74ed8b8cca11203a1a1264d7cc2'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_slowfast(cache_dir: Path, *, offline: bool = False,
                    url: str = SLOWFAST_URL, expected: str = SLOWFAST_SHA256) -> tuple[Path, bool]:
    """Verify cached weights, otherwise download atomically under process lock.

    A separate lock file is retained intentionally: multiple installer processes
    synchronize on the same inode, and never observe partially written weights.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / SLOWFAST_NAME
    lock_path = cache_dir / (SLOWFAST_NAME + '.pixvqa.lock')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if dst.is_file():
            actual = digest(dst)
            if actual != expected:
                raise RuntimeError(f'SimpleVQA SlowFast cache SHA-256 mismatch: {dst} ({actual})')
            return dst, False
        if offline:
            raise RuntimeError(f'SimpleVQA SlowFast backbone not cached: {dst}; rerun installer online')
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.part',
                                             prefix='pixvqa-slowfast-', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                req = urllib.request.Request(url, headers={'User-Agent': 'PixVQA model installer'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
            actual = digest(tmp_path)
            if actual != expected:
                raise RuntimeError(f'SimpleVQA SlowFast downloaded SHA-256 mismatch: {actual}')
            os.replace(tmp_path, dst)
            return dst, True
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description='Verify/download SimpleVQA pretrained SlowFast backbone')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    import torch
    location, downloaded = ensure_slowfast(Path(torch.hub.get_dir()) / 'checkpoints', offline=args.offline)
    print(f"Learned setup: {'installed' if downloaded else 'verified'} SimpleVQA SlowFast backbone: {location}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
