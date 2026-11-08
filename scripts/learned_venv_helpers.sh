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
