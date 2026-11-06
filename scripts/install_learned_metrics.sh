#!/usr/bin/env bash
# Optional learned-backend setup invoked by install_pixvqa.sh after native QA.
# Safe to rerun: never overwrite an existing virtual environment or local config.
set -Eeuo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
env_dir="${PIXVQA_LEARNED_ENV_DIR:-$HOME/pixvqa-learned-env}"
cvvdp_src="${PIXVQA_CVVDP_SOURCE:-$HOME/src/ColorVideoVDP}"
config="$root/config/learned_metrics.local.toml"
commit=166f5ae95ecbd841ae727984cd237de213875399

[[ "$env_dir" = /* && "$cvvdp_src" = /* ]] || {
  echo 'Learned setup: PIXVQA_LEARNED_ENV_DIR and PIXVQA_CVVDP_SOURCE must be absolute paths.' >&2
  exit 2
}

if [[ -x "$env_dir/bin/python" && -f "$env_dir/COLORVIDEOVDP_COMMIT" ]] &&
   [[ "$(cat "$env_dir/COLORVIDEOVDP_COMMIT")" == "$commit" ]]; then
  printf 'Learned setup: reusing existing environment: %s\n' "$env_dir"
elif [[ -e "$env_dir" ]]; then
  echo "Learned setup: $env_dir already exists but is not a recognized pinned environment." >&2
  echo 'Set PIXVQA_LEARNED_ENV_DIR to a new directory or manage this environment manually.' >&2
  exit 2
else
  python="${PYTHON:-python3.12}"
  if ! command -v "$python" >/dev/null 2>&1 || ! "$python" -m venv --help >/dev/null 2>&1; then
    echo "Learned setup requires $python with venv support; install Python 3.12 and python3.12-venv, or set PYTHON." >&2
    exit 2
  fi
  if [[ ! -d "$cvvdp_src/.git" ]]; then
    if [[ -e "$cvvdp_src" ]]; then
      echo "Learned setup: $cvvdp_src exists but is not a Git checkout." >&2
      exit 2
    fi
    echo "Learned setup: cloning ColorVideoVDP into $cvvdp_src"
    mkdir -p "$(dirname -- "$cvvdp_src")"
    git clone https://github.com/gfxdisp/ColorVideoVDP.git "$cvvdp_src"
  fi
  if ! git -C "$cvvdp_src" cat-file -e "$commit^{commit}" 2>/dev/null; then
    echo "Learned setup: fetching pinned ColorVideoVDP commit $commit"
    git -C "$cvvdp_src" fetch origin "$commit"
  fi
  git -C "$cvvdp_src" cat-file -e "$commit^{commit}" 2>/dev/null || {
    echo 'Learned setup: the ColorVideoVDP checkout lacks the pinned commit.' >&2
    exit 2
  }
  PYTHON="$python" bash "$root/scripts/setup_learned_env.sh" "$env_dir" "$cvvdp_src"
fi

if [[ ! -e "$config" ]]; then
  # Using Python avoids fragile sed replacements, and rejects unsafe TOML paths.
  "$env_dir/bin/python" - "$root/config/learned_metrics.example.toml" "$config" "$env_dir" <<'PY'
import pathlib, sys
example, destination, env_dir = map(pathlib.Path, sys.argv[1:])
source = example.read_text()
import json
old = 'python = "~/pixvqa-learned-env/bin/python"'
new = 'python = ' + json.dumps(str(env_dir / 'bin/python'))
lines = source.splitlines(keepends=True)
matches = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == old]
if len(matches) != 1:
    raise SystemExit('Learned setup: unexpected example configuration format')
index = matches[0]
lines[index] = lines[index].replace(old, new, 1)
# Exclusive creation: never overwrite an existing local config.
with destination.open('x') as output:
    output.writelines(lines)
PY
  printf 'Learned setup: created %s\n' "$config"
else
  printf 'Learned setup: preserving existing %s\n' "$config"
fi

printf '\nLearned Python environment: %s\n' "$env_dir"
printf 'ColorVideoVDP: installed from pinned upstream commit.\n'
printf 'CompressedVQA-HDR NR: supply its repository, checkpoint, and calibration profile in %s.\n' "$config"
printf 'See docs/LEARNED_METRICS.md for examples.\n'
