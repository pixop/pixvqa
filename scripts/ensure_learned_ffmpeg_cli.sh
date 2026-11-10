#!/usr/bin/env bash
# Provision pinned ffmpeg+ffprobe into a learned-metrics venv bin directory.
# Idempotent; honors PIXVQA_LEARNED_INSTALL_OFFLINE; never overwrites a validated
# venv copy until a replacement build passes smoke checks.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: ensure_learned_ffmpeg_cli.sh LEARNED_ENV_DIR [REPO_ROOT]

Install or refresh pinned FFmpeg n8.1.3 ffmpeg and ffprobe into LEARNED_ENV_DIR/bin.
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }

env_dir="$1"
root="${2:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
offline="${PIXVQA_LEARNED_INSTALL_OFFLINE:-}"
FFMPEG_TAG=n8.1.3
FFMPEG_COMMIT=1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7
stamp_name=PIXVQA_LEARNED_FFMPEG_COMMIT

[[ "$env_dir" = /* ]] || {
  echo 'Learned FFmpeg CLI: LEARNED_ENV_DIR must be an absolute path.' >&2
  exit 2
}

# shellcheck disable=SC1091
source "$root/scripts/deps_layout.sh"
pixvqa_apply_deps_layout "$root" || exit 1
# shellcheck disable=SC1091
source "$root/scripts/git_retry.sh"
# shellcheck disable=SC1091
source "$root/scripts/learned_venv_helpers.sh"

if (( PIXVQA_DEPS_ISOLATED )); then
  default_cli_root="$PIXVQA_DEPS_ROOT/learned-ffmpeg-cli"
else
  default_cli_root="$root/deps/learned-ffmpeg-cli"
fi
cli_root="$(learned_ffmpeg_cli_root "$root" "$default_cli_root")"
install_prefix="$cli_root/prefix"
# Dedicated CLI source tree (never shares in-source config.h with native libavcodec build).
src="$cli_root/ffmpeg-src"

libdir_for() {
  local bin_path="$1"
  local env_root
  env_root="$(cd -- "$(dirname -- "$bin_path")/.." && pwd -P)"
  if [[ -f "$env_root/PIXVQA_LEARNED_FFMPEG_PREFIX" ]]; then
    printf '%s/lib\n' "$(tr -d '\r\n' < "$env_root/PIXVQA_LEARNED_FFMPEG_PREFIX")"
    return 0
  fi
  printf '%s\n' "$(cd -- "$(dirname -- "$bin_path")/.." && pwd -P)/lib"
}

validate_pair() {
  local ffmpeg_bin="$1" ffprobe_bin="$2"
  local libdir="${3:-$(libdir_for "$ffmpeg_bin")}"
  [[ -x "$ffmpeg_bin" && -x "$ffprobe_bin" && -d "$libdir" ]] || return 1
  env LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$ffmpeg_bin" -hide_banner -version 2>/dev/null | grep -qF "$FFMPEG_TAG" || return 1
  env LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$ffprobe_bin" -hide_banner -version 2>/dev/null | grep -qF "$FFMPEG_TAG" || return 1
  return 0
}

write_wrapper() {
  local name="$1" target="$2" wrapper="$3"
  cat > "$wrapper" <<EOF
#!/usr/bin/env bash
# PixVQA learned FFmpeg wrapper — executes the staged ${FFMPEG_TAG} binary with its private lib dir.
set -euo pipefail
libdir="${install_prefix}/lib"
exec env LD_LIBRARY_PATH="\${libdir}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" "$target" "\$@"
EOF
  chmod 755 "$wrapper"
}

venv_ffmpeg="$env_dir/bin/ffmpeg"
venv_ffprobe="$env_dir/bin/ffprobe"
venv_stamp="$env_dir/$stamp_name"
if [[ -f "$venv_stamp" && "$(cat "$venv_stamp")" == "$FFMPEG_COMMIT" ]] &&
   validate_pair "$venv_ffmpeg" "$venv_ffprobe"; then
  printf 'Learned FFmpeg CLI: reusing %s/bin/ffmpeg and ffprobe\n' "$env_dir"
  exit 0
fi

clone_pinned_ffmpeg() {
  if git -C "$src" rev-parse HEAD >/dev/null 2>&1; then
    local got
    got="$(git -C "$src" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$got" == "$FFMPEG_COMMIT" ]]; then
      return 0
    fi
    echo "Learned FFmpeg CLI: $src exists but is not at pinned commit $FFMPEG_COMMIT." >&2
    exit 2
  fi
  if [[ -e "$src" ]]; then
    echo "Learned FFmpeg CLI: $src exists but is not a Git checkout." >&2
    exit 2
  fi
  if [[ -n "$offline" ]]; then
    echo "Learned FFmpeg CLI: offline and FFmpeg source missing at $src" >&2
    exit 2
  fi
  mkdir -p "$(dirname -- "$src")"
  pixvqa_git_clone "$FFMPEG_TAG" https://git.ffmpeg.org/ffmpeg.git "$src" ||
    pixvqa_git_clone "$FFMPEG_TAG" https://github.com/FFmpeg/FFmpeg.git "$src" || {
      echo 'Learned FFmpeg CLI: could not clone pinned FFmpeg source.' >&2
      exit 1
    }
  local got
  got="$(git -C "$src" rev-parse HEAD)"
  [[ "$got" == "$FFMPEG_COMMIT" ]] || {
    echo "Learned FFmpeg CLI: cloned FFmpeg HEAD $got differs from pin $FFMPEG_COMMIT." >&2
    exit 2
  }
}

staged_ffmpeg="$install_prefix/bin/ffmpeg"
staged_ffprobe="$install_prefix/bin/ffprobe"
staged_stamp="$cli_root/FFMPEG_COMMIT"

need_build=1
if [[ -f "$staged_stamp" ]] &&
   [[ "$(cat "$staged_stamp")" == "$FFMPEG_COMMIT" ]] &&
   validate_pair "$staged_ffmpeg" "$staged_ffprobe"; then
  need_build=0
fi

if (( need_build )); then
  if [[ -n "$offline" ]]; then
    echo "Learned FFmpeg CLI: offline and staged build missing under $install_prefix/bin." >&2
    exit 2
  fi
  clone_pinned_ffmpeg
  mkdir -p "$cli_root"
  bash "$root/scripts/build_learned_ffmpeg_cli.sh" --prefix "$install_prefix"
fi

validate_pair "$staged_ffmpeg" "$staged_ffprobe" "$install_prefix/lib" || {
  echo 'Learned FFmpeg CLI: staged binaries failed validation; refusing to install into venv.' >&2
  exit 1
}

mkdir -p "$env_dir/bin"
tmp_ffmpeg="$(mktemp "$env_dir/bin/.ffmpeg.XXXXXXXX")"
tmp_ffprobe="$(mktemp "$env_dir/bin/.ffprobe.XXXXXXXX")"
trap 'rm -f "$tmp_ffmpeg" "$tmp_ffprobe"' EXIT
write_wrapper ffmpeg "$staged_ffmpeg" "$tmp_ffmpeg"
write_wrapper ffprobe "$staged_ffprobe" "$tmp_ffprobe"
validate_pair "$tmp_ffmpeg" "$tmp_ffprobe" "$install_prefix/lib" || {
  echo 'Learned FFmpeg CLI: venv wrappers failed validation.' >&2
  exit 1
}
mv -f "$tmp_ffmpeg" "$venv_ffmpeg"
mv -f "$tmp_ffprobe" "$venv_ffprobe"
printf '%s\n' "$FFMPEG_COMMIT" > "$venv_stamp"
printf '%s\n' "$install_prefix" > "$env_dir/PIXVQA_LEARNED_FFMPEG_PREFIX"
trap - EXIT

printf 'Learned FFmpeg CLI: installed %s/bin/ffmpeg and ffprobe (FFmpeg %s)\n' "$env_dir" "$FFMPEG_TAG"
