#!/usr/bin/env python3
"""Regression: repair pip transcript embedded by old HDR-ChipQA installer."""
import json
import sys
import tempfile
import tomllib
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'python'))
from pixvqa_learned_install.config_sync import merge_config
with tempfile.TemporaryDirectory() as t:
  dst = Path(t) / 'config' / 'learned_metrics.local.toml'
  dst.parent.mkdir(parents=True)
  bad = 'Collecting scikit-image<0.26\nDownloading file\n━━━━━━━━━━━━━━\nSuccessfully installed scikit-image\n' + str(Path(t) / 'infer/bin/python')
  dst.write_text('[learned]\npython = "../.venv/bin/python"\n\n[hdr-chipqa]\ninference_python = ' + json.dumps(bad) + '\n', encoding='utf-8')
  default = str(Path(t) / 'infer/bin/python')
  merge_config(root/'config/learned_metrics.example.toml',dst,{'hdr-chipqa.inference_python': default})
  obj=tomllib.loads(dst.read_text())
  assert obj['hdr-chipqa']['inference_python'] == default, obj
  assert 'Collecting' not in dst.read_text()
  dst.write_text('[learned]\npython = "/custom/working/venv/bin/python"\n\n[hdr-chipqa]\ninference_python = "/custom/chipqa/bin/python"\n')
  merge_config(root/'config/learned_metrics.example.toml',dst,{'hdr-chipqa.inference_python': default})
  assert tomllib.loads(dst.read_text())['hdr-chipqa']['inference_python'] == '/custom/chipqa/bin/python'
  # Previous pre-venv installer wrote a base interpreter under deps/.
  legacy = Path(t) / 'deps/python-3.12/bin/python3.12'
  dst.write_text('[learned]\npython = ' + json.dumps(str(legacy)) + '\n')
  merge_config(root/'config/learned_metrics.example.toml',dst,{'learned.python': '/managed/venv/bin/python'})
  assert tomllib.loads(dst.read_text())['learned']['python'] == '/managed/venv/bin/python'
  print('PASS contaminated interpreter path and legacy Python repaired; custom path preserved')
