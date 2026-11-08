#!/usr/bin/env python3
"""Offline Google Drive downloader regression; tests atomicity and failure path."""
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pixvqa_learned_install.download import DownloadError, ensure_asset

with tempfile.TemporaryDirectory() as td:
    folder = Path(td)
    py = folder / 'python'
    script = '''#!/usr/bin/env python3
import sys
from pathlib import Path
args = sys.argv
if '--id' not in args or '-O' not in args:
    sys.exit(2)
out = Path(args[args.index('-O') + 1])
if 'FAIL_DOWNLOAD' in __import__('os').environ:
    out.write_text('<html>blocked</html>')
    sys.exit(1)
out.write_bytes(b'PK' + b'0' * 2000)
print('fake download progress')
'''
    py.write_text(script)
    py.chmod(0o755)
    os.environ['PIXVQA_LEARNED_DOWNLOADER_PYTHON'] = str(py)
    destination = folder / 'SimpleVQA' / 'ckpts' / 'UGC_BVQA_model.pth'
    content = b'PK' + b'0' * 2000
    expected = hashlib.sha256(content).hexdigest()
    status, changed = ensure_asset(destination, expected_sha256=expected,
                                   gdrive_id='fake-id', min_bytes=1000, kind='pth')
    assert (status, changed) == ('ok', True)
    assert destination.read_bytes() == content
    assert not destination.with_suffix('.pth.part').exists()
    status, changed = ensure_asset(destination, expected_sha256=expected,
                                   gdrive_id='fake-id', min_bytes=1000, kind='pth')
    assert (status, changed) == ('ok', False)
    destination.unlink()
    os.environ['FAIL_DOWNLOAD'] = '1'
    status, changed = ensure_asset(destination, expected_sha256=expected,
                                   gdrive_id='fake-id', min_bytes=1000, kind='pth')
    assert (status, changed) == ('download_failed', False)
    assert not destination.exists()
    assert not destination.with_suffix('.pth.part').exists()
    os.environ.pop('FAIL_DOWNLOAD')
print('PASS SimpleVQA Google Drive downloader success, reuse, and failure cleanup')
