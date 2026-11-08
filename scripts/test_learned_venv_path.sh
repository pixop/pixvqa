#!/usr/bin/env bash
# Ensure real symlink-backed virtualenv paths survive installer configuration.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd -P)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
python3 -m venv "$work/env"
"$work/env/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix'
physical="$(readlink -f "$work/env/bin/python")"
[[ "$physical" != "$work/env/bin/python" ]] || { echo 'test requires symlink-backed venv' >&2; exit 1; }
PYTHONPATH="$root/python" python3 - "$root" "$work" "$physical" <<'PY'
import pathlib, sys, tomllib
from pixvqa_learned_install.config_sync import merge_config
root, work, physical = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
config = work / 'learned_metrics.local.toml'
config.write_text(f'[learned]\npython = "{physical}"\n\n[hdr-chipqa]\ninference_python = "{physical}"\n')
venv = str(work / 'env/bin/python')
merge_config(root / 'config/learned_metrics.example.toml', config, {
  'learned.python': venv,
  'hdr-chipqa.inference_python': venv,
})
parsed = tomllib.loads(config.read_text())
assert parsed['learned']['python'] == venv
assert parsed['hdr-chipqa']['inference_python'] == venv
# Older installer saved a valid base interpreter rather than the venv.
base = str(pathlib.Path(sys.executable).resolve())
config.write_text(f'[learned]\npython = "{base}"\n')
merge_config(root / 'config/learned_metrics.example.toml', config, {'learned.python': venv})
assert tomllib.loads(config.read_text())['learned']['python'] == venv
config.write_text('[learned]\npython = "/custom/interpreter"\n')
merge_config(root / 'config/learned_metrics.example.toml', config, {'learned.python': venv})
assert tomllib.loads(config.read_text())['learned']['python'] == '/custom/interpreter'
print('PASS venv symlink preservation and generated-config repair')
PY
