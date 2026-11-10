#!/usr/bin/env bash
# Optional learned-backend setup invoked by install_pixvqa.sh after native QA.
# Safe to rerun: reuses pinned venvs/repos; never overwrites verified assets or config keys.
#
# Installs/clones cvvdp, cvqa-hdr-nr, simplevqa, hdr-chipqa upstream trees; downloads
# NR/SimpleVQA checkpoints when licensing and network allow; verifies SHA-256 pins;
# provisions HDR-ChipQA sklearn 1.0.2 inference on private Python 3.10 (micromamba).
# See docs/LEARNED_METRICS.md and: python -m pixvqa_learned_install --help
set -Eeuo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck source=learned_venv_helpers.sh
source "$root/scripts/learned_venv_helpers.sh"
default_env_dir="$root/.venv"
default_cvvdp_src="$root/deps/ColorVideoVDP"
default_chipqa_infer="$root/.venv-chipqa-inference"
env_dir="${PIXVQA_LEARNED_ENV_DIR:-$default_env_dir}"
cvvdp_src="${PIXVQA_CVVDP_SOURCE:-$default_cvvdp_src}"
chipqa_infer_dir="${PIXVQA_HDR_CHIPQA_INFERENCE_ENV_DIR:-$default_chipqa_infer}"
config="$root/config/learned_metrics.local.toml"
commit=166f5ae95ecbd841ae727984cd237de213875399
offline="${PIXVQA_LEARNED_INSTALL_OFFLINE:-}"

[[ "$env_dir" = /* && "$cvvdp_src" = /* && "$chipqa_infer_dir" = /* ]] || {
  echo 'Learned setup: PIXVQA_LEARNED_ENV_DIR, PIXVQA_CVVDP_SOURCE, and PIXVQA_HDR_CHIPQA_INFERENCE_ENV_DIR must be absolute paths.' >&2
  exit 2
}

canonical_python_from_config() {
  local py="$1"
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
  resolved="$(realpath -m -- "$cfg_py" 2>/dev/null || true)"
  if [[ -z "$resolved" ]]; then
    printf 'Learned setup: warning: configured interpreter %s is missing; selected env is %s.\n' \
      "$cfg_py" "$env_dir" >&2
    return 0
  fi
  if [[ "$resolved" != "$(realpath -m -- "$env_dir/bin/python")" ]]; then
    printf 'Learned setup: warning: %s lists python=%s but this step selected %s/bin/python.\n' \
      "$config" "$cfg_py" "$env_dir" >&2
    printf 'Learned setup: align the config or set PIXVQA_LEARNED_ENV_DIR. See docs/LEARNED_METRICS.md.\n' >&2
  fi
}

# --- Shared learned .venv (CVVDP + NR + SimpleVQA deps) ---
if [[ -x "$env_dir/bin/python" && -f "$env_dir/COLORVIDEOVDP_COMMIT" ]] &&
   [[ "$(cat "$env_dir/COLORVIDEOVDP_COMMIT")" == "$commit" ]]; then
  printf 'Learned setup: reusing existing environment: %s\n' "$env_dir"
elif [[ -e "$env_dir" ]]; then
  if [[ "${PIXVQA_LEARNED_RESUME_INCOMPLETE:-}" != 1 ]]; then
    echo "Learned setup: $env_dir already exists but is not a recognized pinned environment." >&2
    echo 'Set PIXVQA_LEARNED_ENV_DIR to a new directory or set PIXVQA_LEARNED_RESUME_INCOMPLETE=1 to resume an isolated, interrupted bootstrap.' >&2
    exit 2
  fi
  assert_learned_venv_isolated "$env_dir" || exit 2
  [[ -d "$cvvdp_src/.git" ]] || { echo "Learned setup: pinned CVVDP checkout missing at $cvvdp_src" >&2; exit 2; }
  printf 'Learned setup: resuming interrupted pinned environment: %s\n' "$env_dir"
  PYTHON="$env_dir/bin/python" bash "$root/scripts/setup_learned_env.sh" --resume "$env_dir" "$cvvdp_src"
else
  python="$(bash "$root/scripts/ensure_learned_python.sh")"
  if [[ ! -d "$cvvdp_src/.git" ]]; then
    if [[ -e "$cvvdp_src" ]]; then
      echo "Learned setup: $cvvdp_src exists but is not a Git checkout." >&2
      exit 2
    fi
    if [[ -z "$offline" ]]; then
      echo "Learned setup: cloning ColorVideoVDP into $cvvdp_src"
      mkdir -p "$(dirname -- "$cvvdp_src")"
      git clone https://github.com/gfxdisp/ColorVideoVDP.git "$cvvdp_src"
    fi
  fi
  if [[ -d "$cvvdp_src/.git" ]]; then
    if ! git -C "$cvvdp_src" cat-file -e "$commit^{commit}" 2>/dev/null; then
      if [[ -z "$offline" ]]; then
        echo "Learned setup: fetching pinned ColorVideoVDP commit $commit"
        git -C "$cvvdp_src" fetch origin "$commit"
      fi
    fi
    git -C "$cvvdp_src" cat-file -e "$commit^{commit}" 2>/dev/null || {
      echo 'Learned setup: the ColorVideoVDP checkout lacks the pinned commit.' >&2
      exit 2
    }
    PYTHON="$python" bash "$root/scripts/setup_learned_env.sh" "$env_dir" "$cvvdp_src"
  elif [[ -n "$offline" ]]; then
    echo "Learned setup: offline mode and ColorVideoVDP checkout missing at $cvvdp_src" >&2
    exit 2
  else
    echo "Learned setup: ColorVideoVDP checkout required at $cvvdp_src" >&2
    exit 2
  fi
  py_bin="$env_dir/bin/python"
  if [[ ! -x "$py_bin" ]]; then
    echo "Learned setup: $py_bin is missing after venv creation." >&2
    exit 1
  fi
  if ! "$py_bin" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "Learned setup: $py_bin is not CPython 3.12." >&2
    exit 1
  fi
fi

# Resolve the directory only, never the Python symlink inside the venv.
selected_python="$(cd -- "$env_dir" && pwd -P)/bin/python"
assert_learned_venv_isolated "$env_dir" || exit 2

# Pinned ffmpeg+ffprobe for learned decoding (out-of-tree CLI build; idempotent).
bash "$root/scripts/ensure_learned_ffmpeg_cli.sh" "$env_dir" "$root"

# Reconcile CVVDP dependencies in both fresh and reused learned environments.
bash "$root/scripts/ensure_cvvdp_dependencies.sh" \
  "$selected_python" "$cvvdp_src" "$commit"

# The old venv stamp proves only the CVVDP bootstrap version, not that every
# learned backend was provisioned. Repair SimpleVQA imports on every rerun.
bash "$root/scripts/ensure_simplevqa_dependencies.sh" \
  "$selected_python" "$root/python/requirements/learned-env.txt"

# --- HDR-ChipQA dual interpreters ---
feature_python="$selected_python"
if [[ -n "${PIXVQA_HDR_CHIPQA_FEATURE_PYTHON:-}" ]]; then
  feature_python="$(realpath -m -- "$PIXVQA_HDR_CHIPQA_FEATURE_PYTHON")"
fi
inference_python="$(bash "$root/scripts/setup_hdr_chipqa_envs.sh" "$feature_python" "$chipqa_infer_dir")"

# --- Upstream repos, checkpoints, config merge, validation ---
# Google's large-file confirmation flow is not reliably handled by urllib.
# Provision the dedicated downloader inside the existing learned .venv.
if [[ -z "$offline" ]]; then
  if ! "$selected_python" -c 'import gdown' >/dev/null 2>&1; then
    echo 'Learned setup: installing pinned Google Drive downloader for SimpleVQA' >&2
    "$selected_python" -m pip --disable-pip-version-check install --no-cache-dir 'gdown==5.2.0' >&2
  fi
fi
export PIXVQA_LEARNED_DOWNLOADER_PYTHON="$selected_python"
# PyTorchVideo's upstream SimpleVQA implementation loads an auxiliary
# SlowFast checkpoint on first inference. Verify/download it now, into the
# same torch.hub cache used at runtime, with a hash pin and interprocess lock.
PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}" "$selected_python" \
  -m pixvqa_learned_install.backbones ${offline:+--offline}
# The asset/config module uses stdlib tomllib (Python >=3.11). Ubuntu 22.04
# provides Python 3.10 as `python3`; use the managed learned Python 3.12.
pyw="${PIXVQA_LEARNED_CONFIG_PYTHON:-$selected_python}"
if [[ ! -x "$pyw" ]] && ! command -v "$pyw" >/dev/null 2>&1; then
  echo "Learned setup: config interpreter unavailable: $pyw" >&2
  exit 2
fi
if ! "$pyw" -c 'import sys, tomllib; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  echo "Learned setup: config interpreter must provide Python >=3.11 and tomllib: $pyw" >&2
  exit 2
fi

install_args=(--root "$root" --learned-python "$selected_python"
  --feature-python "$feature_python" --inference-python "$inference_python")
if [[ -n "$offline" ]]; then
  install_args+=(--offline)
fi

set +e
PYTHONPATH="$root/python" "$pyw" -m pixvqa_learned_install install "${install_args[@]}"
install_status=$?
set -e

warn_config_python_mismatch

printf '\nLearned Python environment: %s\n' "$env_dir"
printf 'HDR-ChipQA feature Python: %s\n' "$feature_python"
printf 'HDR-ChipQA inference Python: %s\n' "$inference_python"
printf 'Configuration: %s\n' "$config"
printf 'See docs/LEARNED_METRICS.md for licensing, manual checkpoints, and limitations.\n'

if [[ "$install_status" -ne 0 ]]; then
  echo 'Learned setup: one or more backends are not fully installed — see messages above.' >&2
  exit "$install_status"
fi
