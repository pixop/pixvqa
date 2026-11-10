#!/usr/bin/env bash
# Fail closed when a reused learned .venv does not match python/requirements/learned-env.txt.
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 LEARNED_PYTHON LOCKFILE" >&2; exit 2; }
py="$1"
lock="$2"
[[ -x "$py" && -f "$lock" ]] || {
  echo 'Learned env pins: missing interpreter or lockfile' >&2
  exit 2
}
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
env_dir="$(cd -- "$(dirname -- "$py")/.." && pwd -P)"
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
exec "$py" -m pixvqa_learned_install.env_pins "$lock" --env-dir "$env_dir"
