#!/usr/bin/env bash
# Ensure CPython 3.10 for HDR-ChipQA SVR inference (sklearn 1.0.2). Prints absolute
# interpreter on stdout; never modifies system Python.
set -euo pipefail

chipqa_python_valid() {
  local py="$1"
  [[ -x "$py" ]] || command -v "$py" >/dev/null 2>&1 || return 1
  "$py" -c 'import sys, venv; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)' >/dev/null 2>&1
}

if [[ -n "${PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON:-}" ]]; then
  py="${PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON}"
  if chipqa_python_valid "$py"; then
    "$py" -c 'import sys; print(sys.executable)'
    exit 0
  fi
  echo "Learned setup: PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON=$py is not CPython 3.10." >&2
  exit 2
fi

if chipqa_python_valid python3.10; then
  python3.10 -c 'import sys; print(sys.executable)'
  exit 0
fi

root="${PIXVQA_DEPS_ROOT:-$HOME/pixvqa-deps}"
[[ -n "$root" && "$root" = /* && "$root" != / ]] || {
  echo 'Learned setup: PIXVQA_DEPS_ROOT must be an absolute directory.' >&2
  exit 2
}
prefix="$root/python310"
if chipqa_python_valid "$prefix/bin/python"; then
  printf '%s\n' "$prefix/bin/python"
  exit 0
fi
if [[ -e "$prefix" ]]; then
  echo "Learned setup: $prefix exists but Python 3.10 is unavailable; inspect before retrying." >&2
  exit 2
fi

if command -v micromamba >/dev/null 2>&1; then
  manager="$(command -v micromamba)"
elif command -v conda >/dev/null 2>&1; then
  manager="$(command -v conda)"
else
  tools_dir="$root/tools"
  manager="$tools_dir/bin/micromamba"
  if [[ ! -x "$manager" ]]; then
    command -v curl >/dev/null 2>&1 && command -v tar >/dev/null 2>&1 || {
      echo 'Learned setup: curl and tar are required to bootstrap micromamba for Python 3.10.' >&2
      exit 2
    }
    mkdir -p "$tools_dir"
    archive="$(mktemp "$tools_dir/.micromamba.XXXXXXXX.tar.bz2")"
    trap 'rm -f -- "$archive"' EXIT
    echo 'Learned setup: fetching micromamba to provision private Python 3.10 for HDR-ChipQA.' >&2
    curl -fLsS https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$archive"
    tar -xjf "$archive" -C "$tools_dir" bin/micromamba
  fi
fi

mkdir -p "$root"
echo "Learned setup: provisioning Python 3.10 into $prefix (HDR-ChipQA inference only)." >&2
if [[ "${manager##*/}" == conda ]]; then
  "$manager" create -y -p "$prefix" --override-channels -c conda-forge 'python=3.10' >&2
else
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$root/micromamba-root}" \
    "$manager" create -y -p "$prefix" --override-channels -c conda-forge 'python=3.10' >&2
fi
chipqa_python_valid "$prefix/bin/python" || {
  echo "Learned setup: provisioned Python 3.10 invalid at $prefix/bin/python" >&2
  exit 1
}
printf '%s\n' "$prefix/bin/python"
