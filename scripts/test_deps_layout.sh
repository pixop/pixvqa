#!/usr/bin/env bash
# Path checks for the guided installer dependency layout.
# Does not install apt packages, download sources, or modify $HOME/dav1d-1.5.1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

bash -n install_pixvqa.sh
bash -n scripts/deps_layout.sh
bash -n scripts/build_zimg.sh
bash -n scripts/build_libvmaf.sh
bash -n scripts/libvmaf_flags.sh

grep -q 'PIXVQA_DEPS_ROOT' install_pixvqa.sh || fail 'installer does not read PIXVQA_DEPS_ROOT'
grep -q 'scripts/deps_layout.sh' install_pixvqa.sh || fail 'installer does not source the layout helper'
if grep -Eq 'archive\.ubuntu\.com|\.deb' install_pixvqa.sh; then
  fail 'installer downloads Ubuntu binary packages'
fi
grep -q 'PIXVQA_MANUAL_ROOT' docs/INSTALL.md || fail 'manual procedure has no dependency root'
grep -q 'apt packages' docs/INSTALL.md || fail 'manual procedure does not say apt stays shared'
grep -q 'PIXVQA_DEPS_ROOT="\$PWD/deps"' README.md || fail 'README omits the isolated install command'

real_home="${HOME}"
snapshot() {
  local path="$1"
  if [[ -e "$path" ]]; then
    stat -c '%d:%i:%Y' "$path" 2>/dev/null || stat -f '%d:%i:%m' "$path"
  else
    printf 'absent\n'
  fi
}
before_dav1d="$(snapshot "$real_home/dav1d-1.5.1")"
before_ffmpeg="$(snapshot "$real_home/ffmpeg-n8.1.3")"

fake_home="$(mktemp -d)"
src="$(mktemp -d)"
mkdir -p "$src/deps"
isolated="$(mktemp -d)/pixvqa-deps"
cleanup() { rm -rf "$fake_home" "$src" "$(dirname "$isolated")"; }
trap cleanup EXIT

(
  unset PIXVQA_DEPS_ROOT LIBVMAF_SRC LIBVMAF_BUILD ZIMG_PREFIX ZIMG_SRC CUDA_HOME
  export HOME="$fake_home"
  # shellcheck disable=SC1091
  source "$ROOT/scripts/deps_layout.sh"
  pixvqa_apply_deps_layout "$src"
  [[ "$PIXVQA_DEPS_ISOLATED" == 0 ]]
  [[ "$DAV1D_PREFIX" == "$fake_home/dav1d-1.5.1" ]]
  [[ "$FFMPEG_PREFIX" == "$fake_home/ffmpeg-n8.1.3" ]]
  [[ "$DEPS_SRC" == "$fake_home/pixvqa-deps/src" ]]
  [[ "$GCC13" == "$fake_home/pixvqa-deps/gcc13" ]]
  [[ "$MAMBA_ROOT_PREFIX" == "$fake_home/pixvqa-deps/micromamba-root" ]]
  [[ "$PIXVQA_TOOLS_DIR" == "$fake_home/pixvqa-deps/tools" ]]
  [[ "$NVCODEC_PREFIX" == "$fake_home/pixvqa-deps/nv-codec-headers" ]]
  [[ "$LIBVMAF_PREFIX" == "$src/deps/libvmaf-74f930d" ]]
  [[ -z "${ZIMG_PREFIX:-}" ]]
  [[ -z "${CUDA_HOME:-}" ]]
  [[ ! -e "$fake_home/dav1d-1.5.1" ]]
)
echo "OK   legacy defaults"

(
  unset CUDA_HOME
  export HOME="$fake_home"
  export PIXVQA_DEPS_ROOT="$isolated"
  # shellcheck disable=SC1091
  source "$ROOT/scripts/deps_layout.sh"
  pixvqa_apply_deps_layout "$src"
  [[ "$PIXVQA_DEPS_ISOLATED" == 1 ]]
  root="$(cd -- "$PIXVQA_DEPS_ROOT" && pwd -P)"
  [[ "$DAV1D_PREFIX" == "$root/dav1d-1.5.1" ]]
  [[ "$FFMPEG_PREFIX" == "$root/ffmpeg-n8.1.3" ]]
  [[ "$DEPS_SRC" == "$root/src" ]]
  [[ "$GCC13" == "$root/gcc13" ]]
  [[ "$MAMBA_ROOT_PREFIX" == "$root/micromamba-root" ]]
  [[ "$PIXVQA_TOOLS_DIR" == "$root/tools" ]]
  [[ "$NVCODEC_PREFIX" == "$root/nv-codec-headers" ]]
  [[ "$LIBVMAF_PREFIX" == "$root/libvmaf-74f930d" ]]
  [[ "$LIBVMAF_SRC" == "$root/src/vmaf-74f930d" ]]
  [[ "$LIBVMAF_BUILD" == "$root/build/libvmaf-74f930d" ]]
  [[ "$ZIMG_PREFIX" == "$root/zimg-3.0.6" ]]
  [[ "$ZIMG_SRC" == "$root/src/zimg" ]]
  [[ "$ZIMG_AUTOTOOLS_DEBS" == "$root/src/host-debs" ]]
  [[ "$DAV1D_PREFIX" != "$fake_home/dav1d-1.5.1" ]]
  [[ "$FFMPEG_PREFIX" != "$fake_home/ffmpeg-n8.1.3" ]]
  [[ "$DAV1D_PREFIX" != "$real_home/dav1d-1.5.1" ]]
  [[ -z "${CUDA_HOME:-}" ]]
  [[ ! -e "$DAV1D_PREFIX" && ! -e "$FFMPEG_PREFIX" ]]
)
echo "OK   isolated root"

(
  export HOME="$fake_home"
  export PIXVQA_DEPS_ROOT="$fake_home"
  # shellcheck disable=SC1091
  source "$ROOT/scripts/deps_layout.sh"
  if pixvqa_apply_deps_layout "$src"; then
    fail 'isolated root accepted the home directory'
  fi
  export PIXVQA_DEPS_ROOT="$fake_home/dav1d-1.5.1"
  if pixvqa_apply_deps_layout "$src"; then
    fail 'isolated root accepted the legacy dav1d prefix'
  fi
  [[ ! -e "$fake_home/dav1d-1.5.1" ]]
  export PIXVQA_DEPS_ROOT=""
  if pixvqa_apply_deps_layout "$src"; then
    fail 'empty PIXVQA_DEPS_ROOT was accepted'
  fi
)
echo "OK   isolated root rejects home prefixes"

(
  unset ZIMG_PREFIX
  # shellcheck disable=SC1091
  source "$ROOT/scripts/libvmaf_flags.sh"
  if [[ -f "$ROOT/deps/zimg-3.0.6/lib/pkgconfig/zimg.pc" || -f "$ROOT/deps/zimg-3.0.6/lib64/pkgconfig/zimg.pc" ]]; then
    got="$(pixvqa_zimg_pc_dir "$ROOT")"
    [[ "$got" == "$ROOT/deps/zimg-3.0.6/"* ]]
  fi
  other="$(mktemp -d)"
  mkdir -p "$other/lib/pkgconfig"
  printf 'Name: zimg\nVersion: 3.0.6\n' > "$other/lib/pkgconfig/zimg.pc"
  export ZIMG_PREFIX="$other"
  got="$(pixvqa_zimg_pc_dir "$ROOT")"
  [[ "$got" == "$other/lib/pkgconfig" ]]
  rm -rf "$other"
)
echo "OK   explicit zimg prefix is not replaced by the checkout"

fns="$(mktemp)"
sed -n '36,138p' scripts/build_zimg.sh > "$fns"
bash -n "$fns"
set +e
out="$(bash -c '
  set -euo pipefail
  source "$1"
  autotools_ready() { return 1; }
  ubuntu_version_id() { printf "22.04\n"; }
  curl() { printf "DOWNLOADED\n"; exit 99; }
  ensure_autotools
' bash "$fns" 2>&1)"
status=$?
set -e
rm -f "$fns"
printf '%s\n' "$out" | grep -q 'refusing Ubuntu 24.04 binary autotools on Ubuntu 22.04' || fail "22.04 refusal missing: $out"
printf '%s\n' "$out" | grep -q 'DOWNLOADED' && fail '22.04 branch downloaded packages'
[[ "$status" == 1 ]] || fail "22.04 branch exited $status"
echo "OK   Ubuntu 22.04 does not download Ubuntu 24.04 autotools"

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  printf 'Host OS: %s %s\n' "${ID:-unknown}" "${VERSION_ID:-unknown}"
fi

after_dav1d="$(snapshot "$real_home/dav1d-1.5.1")"
after_ffmpeg="$(snapshot "$real_home/ffmpeg-n8.1.3")"
[[ "$before_dav1d" == "$after_dav1d" ]] || fail 'HOME dav1d prefix changed'
[[ "$before_ffmpeg" == "$after_ffmpeg" ]] || fail 'HOME FFmpeg prefix changed'
echo "OK   existing home prefixes unchanged"
echo "All dependency-layout checks passed"
