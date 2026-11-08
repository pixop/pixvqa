#!/usr/bin/env python3
"""Child termination must never be misreported as a NumPy warning."""
import contextlib
import importlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'python'))
from pixvqa_hdr_chipqa import adapter
with tempfile.TemporaryDirectory() as td:
    path=Path(td)
    for code, message, needle in [
        (-9, 'warning only', 'SIGKILL (9)'),
        (1, 'Traceback: genuine missing symbol', 'status 1'),
        (137, '', 'status 137'),
    ]:
        with patch.object(adapter, 'resolve_python', return_value=sys.executable), patch.object(adapter.subprocess, 'run', return_value=subprocess.CompletedProcess([],code,'',message)):
            try:
                adapter.extract_features(path,path/'input.yuv',path/'feature.z',16,16,bit_depth=10,color_space='BT709')
            except adapter.InferenceError as e:
                assert needle in str(e),str(e)
                if code == -9: assert 'OOM' in str(e)
                if code == 137: assert 'no Python traceback' in str(e)
            else:
                raise AssertionError('failed child was accepted')
# np.object/np.str lookups must not emit FutureWarning.
import warnings
with warnings.catch_warnings(record=True) as events:
    warnings.simplefilter('always')
    importlib.import_module('pixvqa_hdr_chipqa.numpy_legacy_shim')
    assert not events,events
print('PASS HDR-ChipQA failure exit/signal diagnostics and silent NumPy shim')
