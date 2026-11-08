#!/usr/bin/env python3
"""Regression: pinned CVVDP has a conflicting ffmpeg PyPI requirement.

CVVDP requires ffmpeg-python's API; `ffmpeg` is another distribution that
shadows its module name. The installer must not install that distribution.
"""
from pathlib import Path
import subprocess
import tarfile
import tempfile
import tomllib

root = Path(__file__).resolve().parents[1]
rev = '166f5ae95ecbd841ae727984cd237de213875399'
with tempfile.TemporaryDirectory() as td:
    temp = Path(td)
    archive = temp / 'tree.tar'
    with archive.open('wb') as out:
        subprocess.run(['git', '-C', str(root / 'deps/ColorVideoVDP'),
                        'archive', '--format=tar', rev], check=True, stdout=out)
    with tarfile.open(archive) as tf:
        pyproject = tf.extractfile('pyproject.toml').read().decode()
    before = tomllib.loads(pyproject)['project']['dependencies']
    assert 'ffmpeg>=1.4' in before
    assert 'ffmpeg-python>=0.2.0' in before
    bad = '   "ffmpeg>=1.4",\n'
    assert pyproject.count(bad) == 1
    after = tomllib.loads(pyproject.replace(bad, ''))['project']['dependencies']
    assert 'ffmpeg>=1.4' not in after
    assert 'ffmpeg-python>=0.2.0' in after
    assert 'PyEXR>=0.3.10' in after
    assert len(after) == len(before) - 1
    shell = (root / 'scripts/ensure_cvvdp_dependencies.sh').read_text()
    assert '--no-deps --no-build-isolation --force-reinstall' in shell
    assert 'pip check' in shell
    assert "('PyEXR', 'PyEXR==0.5.0')" in shell
    # Installing PyEXR first triggers pip's misleading transient error about
    # the stale ffmpeg requirement; metadata correction must precede pip repair.
    assert shell.index("from importlib.metadata import distribution") < shell.index('missing="$("$py"')
print('PASS pinned ColorVideoVDP dependency correction only removes conflicting ffmpeg')
