#!/usr/bin/env bash
# Ensure Python 3.12 with venv for the optional learned-metric environment.
# Prints the selected absolute interpreter path on stdout. All progress goes
# to stderr so callers can safely use command substitution.
set -euo pipefail

learned_python_valid() {
  local py="$1"
  [[ -x "$py" ]] || command -v "$py" >/dev/null 2>&1 || return 1
  "$py" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1
}

if [[ -n "${PYTHON:-}" ]]; then
  if ! learned_python_valid "$PYTHON"; then
    echo "Learned setup: PYTHON=$PYTHON is not a working Python 3.12 with venv and ensurepip." >&2
    exit 2
  fi
  "${PYTHON}" -c 'import sys; print(sys.executable)'
  exit 0
fi

if learned_python_valid python3.12; then
  python3.12 -c 'import sys; print(sys.executable)'
  exit 0
fi

# Avoid changing system Python. Provision a dedicated interpreter prefix,
# separate from the application .venv created by setup_learned_env.sh.
root="${PIXVQA_DEPS_ROOT:-$HOME/pixvqa-deps}"
[[ -n "$root" && "$root" = /* && "$root" != / ]] || {
  echo 'Learned setup: PIXVQA_DEPS_ROOT must be an absolute directory.' >&2
  exit 2
}
prefix="$root/python312"
if learned_python_valid "$prefix/bin/python"; then
  printf '%s\n' "$prefix/bin/python"
  exit 0
fi
if [[ -e "$prefix" ]]; then
  echo "Learned setup: $prefix exists but Python 3.12 is unavailable or incomplete; inspect it before retrying." >&2
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
      echo 'Learned setup: curl and tar are required to bootstrap micromamba.' >&2
      exit 2
    }
    mkdir -p "$tools_dir"
    archive="$(mktemp "$tools_dir/.micromamba.XXXXXXXX.tar.bz2")"
    trap 'rm -f -- "$archive"' EXIT
    echo 'Learned setup: fetching micromamba to install a private Python 3.12.' >&2
    curl -fLsS https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$archive"
    tar -xjf "$archive" -C "$tools_dir" bin/micromamba
  fi
fi

mkdir -p "$root"
echo "Learned setup: provisioning Python 3.12 into $prefix" >&2
if [[ "${manager##*/}" == conda ]]; then
  "$manager" create -y -p "$prefix" --override-channels -c conda-forge 'python=3.12' >&2
else
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$root/micromamba-root}" \
    "$manager" create -y -p "$prefix" --override-channels -c conda-forge 'python=3.12' >&2
fi
if ! learned_python_valid "$prefix/bin/python"; then
  echo "Learned setup: provisioned Python is invalid at $prefix/bin/python." >&2
  exit 1
fi
printf '%s\n' "$prefix/bin/python"
