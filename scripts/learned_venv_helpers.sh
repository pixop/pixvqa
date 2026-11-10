# Shared helpers for creating the learned-metrics venv (sourced by setup and tests).
# On WSL, repos on Windows-backed mounts (9p/drvfs/fuse) can break default venv symlinks;
# use `python -m venv --copies` unless PIXVQA_LEARNED_VENV_USE_SYMLINKS is set.

learned_venv_on_wsl_network_fs() {
  local env_dir="$1"
  command -v findmnt >/dev/null 2>&1 || return 1
  local parent fstype src
  parent="$(dirname -- "$env_dir")"
  [ -d "$parent" ] || return 1
  parent="$(cd "$parent" && pwd)"
  fstype="$(findmnt -n -o FSTYPE -T "$parent" 2>/dev/null || true)"
  src="$(findmnt -n -o SOURCE -T "$parent" 2>/dev/null || true)"
  [ -n "$fstype" ] || [ -n "$src" ] || return 1
  # WSL /mnt/<drive>: FSTYPE is typically 9p; SOURCE may be a Windows path without "9p" in it.
  [ "$fstype" = 9p ] || case "$fstype" in fuse*) true ;; *) false ;; esac ||
    case "$src" in *fuse*|*9p*|*drvfs*) true ;; *) false ;; esac
}

learned_venv_create_flags() {
  local env_dir="$1"
  if [ -n "${PIXVQA_LEARNED_VENV_USE_SYMLINKS:-}" ]; then
    printf '%s\n' ""
    return
  fi
  if learned_venv_on_wsl_network_fs "$env_dir"; then
    printf '%s\n' "--copies"
  else
    printf '%s\n' ""
  fi
}

# Fail fast when the learned venv is a wrapper, system Python, or site-packages leak
# into a forbidden prefix (set PIXVQA_ISOLATION_FORBID_PREFIXES as colon-separated paths).
assert_learned_venv_isolated() {
  local env_dir="$1"
  local py="$env_dir/bin/python"
  [[ -x "$py" ]] || {
    echo "Learned venv isolation: $py is missing or not executable." >&2
    return 1
  }
  if file -b "$py" 2>/dev/null | grep -qiE 'shell script|text executable'; then
    if head -1 "$py" 2>/dev/null | grep -qE '^#!.*sh'; then
      echo "Learned venv isolation: $py is a shell wrapper, not a venv interpreter." >&2
      return 1
    fi
  fi
  if [[ -n "${PIP_TARGET:-}" ]]; then
    echo "Learned venv isolation: PIP_TARGET is set ($PIP_TARGET); refusing pip installs." >&2
    return 1
  fi
  local forbid_csv="${PIXVQA_ISOLATION_FORBID_PREFIXES:-}"
  env_dir="$(cd -- "$env_dir" && pwd -P)"
  FORBID_CSV="$forbid_csv" "$py" - "$env_dir" <<'PY'
import importlib
import os
import pathlib
import sys

env_dir = pathlib.Path(sys.argv[1]).resolve()
prefix = pathlib.Path(sys.prefix).resolve()
base = pathlib.Path(sys.base_prefix).resolve()
if prefix != env_dir:
    raise SystemExit(f"sys.prefix {prefix} != env_dir {env_dir}")
if prefix == base:
    raise SystemExit(f"sys.prefix == sys.base_prefix ({prefix}); not an isolated venv")
cfg = env_dir / "pyvenv.cfg"
if not cfg.is_file():
    raise SystemExit(f"missing pyvenv.cfg in {env_dir}")
include = "false"
for line in cfg.read_text(encoding="utf-8").splitlines():
    if line.strip().startswith("include-system-site-packages"):
        include = line.split("=", 1)[1].strip().lower()
        break
if include in {"1", "true", "yes", "on"}:
    raise SystemExit("include-system-site-packages is enabled")
for raw in os.environ.get("FORBID_CSV", "").split(":"):
    raw = raw.strip()
    if not raw:
        continue
    bad = pathlib.Path(raw).resolve()
    if str(prefix).startswith(str(bad)) and prefix != bad:
        raise SystemExit(f"venv prefix {prefix} is under forbidden path {bad}")
    if str(prefix) == str(bad):
        raise SystemExit(f"venv prefix equals forbidden path {bad}")
md = importlib.import_module("importlib.metadata")
pip_loc = pathlib.Path(md.distribution("pip").locate_file("")).resolve()
if not str(pip_loc).startswith(str(env_dir)):
    raise SystemExit(f"pip location {pip_loc} is outside venv {env_dir}")
for raw in os.environ.get("FORBID_CSV", "").split(":"):
    raw = raw.strip()
    if not raw:
        continue
    bad = pathlib.Path(raw).resolve()
    if str(pip_loc).startswith(str(bad)):
        raise SystemExit(f"pip location {pip_loc} leaks into forbidden prefix {bad}")
PY
}

learned_pip_cache_flag() {
  if [ -n "${PIXVQA_PIP_USE_CACHE:-}" ]; then
    printf '%s\n' ""
  else
    printf '%s\n' "--no-cache-dir"
  fi
}

# Out-of-tree FFmpeg CLI builds fail on WSL drvfs/9p checkouts; keep artifacts on Linux fs.
learned_ffmpeg_cli_root() {
  local checkout_root="$1" default_root="$2"
  if [ -n "${PIXVQA_LEARNED_FFMPEG_CLI_ROOT:-}" ]; then
    printf '%s\n' "${PIXVQA_LEARNED_FFMPEG_CLI_ROOT}"
    return
  fi
  if learned_venv_on_wsl_network_fs "$checkout_root"; then
    local home_root
    home_root="$(cd -- "$HOME" && pwd -P)"
    printf '%s\n' "$home_root/pixvqa-deps/learned-ffmpeg-cli"
    return
  fi
  printf '%s\n' "$default_root"
}
