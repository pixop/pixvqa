#!/usr/bin/env bash
# Optional learned-backend setup invoked by install_pixvqa.sh after native QA.
# Safe to rerun: never overwrite an existing virtual environment or local config.
#
# Defaults (overridable): repository-local .venv (cvvdp + cvqa-hdr-nr + SimpleVQA deps)
# and deps/ColorVideoVDP checkout.
# Set PIXVQA_LEARNED_ENV_DIR and/or PIXVQA_CVVDP_SOURCE to absolute paths to
# use another layout (for example an existing ~/pixvqa-learned-env).
# Optional: PIXVQA_LEARNED_CONFIG_PYTHON overrides the host interpreter used
# only to write learned_metrics.local.toml from the example (default: python3).
set -Eeuo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
default_env_dir="$root/.venv"
default_cvvdp_src="$root/deps/ColorVideoVDP"
env_dir="${PIXVQA_LEARNED_ENV_DIR:-$default_env_dir}"
cvvdp_src="${PIXVQA_CVVDP_SOURCE:-$default_cvvdp_src}"
config="$root/config/learned_metrics.local.toml"
commit=166f5ae95ecbd841ae727984cd237de213875399

[[ "$env_dir" = /* && "$cvvdp_src" = /* ]] || {
  echo 'Learned setup: PIXVQA_LEARNED_ENV_DIR and PIXVQA_CVVDP_SOURCE must be absolute paths.' >&2
  exit 2
}

canonical_python_from_config() {
  local py="$1"
  # Print the absolute path PixVQA resolves for [learned] python=, or empty.
  "$py" - "$config" <<'PY' 2>/dev/null || true
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    sys.exit(0)
text = path.read_text()
match = re.search(r'^\s*python\s*=\s*"((?:\\.|[^"\\])*)"\s*$', text, re.MULTILINE)
if not match:
    match = re.search(r"^\s*python\s*=\s*'((?:\\.|[^'\\])*)'\s*$", text, re.MULTILINE)
if not match:
    sys.exit(0)
raw = bytes(match.group(1), "utf-8").decode("unicode_escape")
cfg_dir = path.parent.resolve()
if not raw or raw.find("/") == -1:
    print("")
elif raw.startswith("~/"):
    home = pathlib.Path.home()
    print(str((home / raw[2:]).resolve()))
elif raw.startswith("/"):
    print(str(pathlib.Path(raw).resolve()))
else:
    print(str((cfg_dir / raw).resolve()))
PY
}

warn_config_python_mismatch() {
  [[ -f "$config" ]] || return 0
  local py_check="${PYTHON_CHECK:-python3}"
  if ! command -v "$py_check" >/dev/null 2>&1; then
    printf 'Learned setup: warning: %s not found; cannot validate %s against the selected env.\n' \
      "$py_check" "$config" >&2
    return 0
  fi
  local cfg_py resolved
  cfg_py="$(canonical_python_from_config "$py_check")"
  if [[ -z "$cfg_py" ]]; then
    printf 'Learned setup: warning: could not parse [learned].python in %s; verify it matches %s/bin/python.\n' \
      "$config" "$env_dir" >&2
    return 0
  fi
  resolved="$(readlink -f -- "$cfg_py" 2>/dev/null || true)"
  if [[ -z "$resolved" ]]; then
    printf 'Learned setup: warning: configured interpreter %s is missing; selected env is %s.\n' \
      "$cfg_py" "$env_dir" >&2
    return 0
  fi
  if [[ "$resolved" != "$(readlink -f -- "$env_dir/bin/python")" ]]; then
    printf 'Learned setup: warning: %s lists python=%s but this step selected %s/bin/python.\n' \
      "$config" "$cfg_py" "$env_dir" >&2
    printf 'Learned setup: (opt-in) stop using the old env, then align the config, or set PIXVQA_LEARNED_ENV_DIR to that env. See docs/LEARNED_METRICS.md.\n' >&2
  fi
}

if [[ -x "$env_dir/bin/python" && -f "$env_dir/COLORVIDEOVDP_COMMIT" ]] &&
   [[ "$(cat "$env_dir/COLORVIDEOVDP_COMMIT")" == "$commit" ]]; then
  printf 'Learned setup: reusing existing environment: %s\n' "$env_dir"
elif [[ -e "$env_dir" ]]; then
  echo "Learned setup: $env_dir already exists but is not a recognized pinned environment." >&2
  echo 'Set PIXVQA_LEARNED_ENV_DIR to a new directory or manage this environment manually.' >&2
  exit 2
else
  python="$(bash "$root/scripts/ensure_learned_python.sh")"
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
  # Runtime validation: venv entry scripts must resolve to an executable interpreter.
  py_bin="$env_dir/bin/python"
  if [[ ! -x "$py_bin" ]]; then
    echo "Learned setup: $py_bin is missing or not executable after venv creation." >&2
    exit 1
  fi
  if ! "$py_bin" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "Learned setup: $py_bin is not a working CPython 3.12 interpreter (check venv and mount)." >&2
    exit 1
  fi
fi

selected_python="$(readlink -f -- "$env_dir/bin/python")"

write_local_config() {
  local selected="$1"
  # Prefer the system interpreter for this small stdlib-only step so a stub
  # venv python (tests) or a broken venv still produces config. The learned
  # environment remains $env_dir for runtime and for setup_learned_env.sh.
  local pyw="${PIXVQA_LEARNED_CONFIG_PYTHON:-}"
  if [[ -z "$pyw" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      pyw=python3
    elif command -v python3.12 >/dev/null 2>&1; then
      pyw=python3.12
    else
      pyw="$env_dir/bin/python"
    fi
  fi
  # Using Python avoids fragile sed replacements, and rejects unsafe TOML paths.
  "$pyw" - "$root/config/learned_metrics.example.toml" "$config" "$selected" <<'PY'
import json, pathlib, re, sys

example, destination, selected = map(pathlib.Path, sys.argv[1:])
selected_str = str(selected)
source = example.read_text()

def replace_learned_python(text, new_value):
    pattern = re.compile(
        r'^(\s*python\s*=\s*)'
        r'(?:"[^"\\]*(?:\\.[^"\\]*)*"|'
        r"'[^'\\]*(?:\\.[^'\\]*)*')"
        r"(\s*)$",
        re.MULTILINE,
    )

    def repl(match):
        return match.group(1) + json.dumps(new_value) + match.group(2)

    updated, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise SystemExit('Learned setup: unexpected example configuration format ([learned].python)')
    return updated

updated = replace_learned_python(source, selected_str)
# Exclusive creation: never overwrite an existing local config.
with destination.open('x') as output:
    output.write(updated)
PY
}

if [[ ! -e "$config" ]]; then
  write_local_config "$selected_python"
  printf 'Learned setup: created %s\n' "$config"
else
  printf 'Learned setup: preserving existing %s\n' "$config"
  warn_config_python_mismatch
fi

printf '\nLearned Python environment: %s\n' "$env_dir"
printf 'ColorVideoVDP upstream checkout: %s (pinned commit; see scripts/setup_learned_env.sh).\n' "$cvvdp_src"
printf 'CompressedVQA-HDR NR: obtain the repository, checkpoint, and calibration profile separately; configure paths in %s (no automatic model download).\n' "$config"
printf 'SimpleVQA: use the same interpreter for `python -m pixvqa_simplevqa` (not registered in pixvqa --metrics yet); obtain upstream repo under deps/SimpleVQA and checkpoints separately.\n'
printf 'See docs/LEARNED_METRICS.md for layout, WSL/NTFS venv notes, licensing, and opt-in migration from an external env.\n'
