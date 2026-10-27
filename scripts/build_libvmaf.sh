#!/usr/bin/env bash
# Build the HDR-capable Netflix libvmaf source pin into deps/libvmaf-74f930d.
# This is not a tagged release. The last tagged release remains v3.2.1
# (commit f85a853692a8c730d0270cd733c8bb30b5b93b7c), still buildable with
# scripts/build_libvmaf_3.2.1.sh. Do not describe this pin as v3.3.
# zimg is required. CUDA stays off. There is no fallback to libvmaf 3.2.1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN_COMMIT="74f930d17de8c58e7c2ce55e2b897b0030ebd8ef"
PREVIOUS_TAG="v3.2.1"
PREVIOUS_COMMIT="f85a853692a8c730d0270cd733c8bb30b5b93b7c"
SRC="${LIBVMAF_SRC:-$ROOT/deps/src/vmaf-74f930d}"
PREFIX="${LIBVMAF_PREFIX:-$ROOT/deps/libvmaf-74f930d}"
BUILD="${LIBVMAF_BUILD:-$ROOT/deps/build/libvmaf-74f930d}"
ZIMG_PREFIX="${ZIMG_PREFIX:-$ROOT/deps/zimg-3.0.6}"

# Optional explicit compiler prefix. There is no default user directory.
if [[ -n "${GCC_ENV:-}" ]]; then
  export PATH="${GCC_ENV}/bin:${PATH}"
fi

for tool in git meson ninja nasm pkg-config gcc g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "build_libvmaf: missing required tool '$tool' on PATH" >&2
    exit 1
  fi
done

bash "$ROOT/scripts/build_zimg.sh"

ZPC=""
if [[ -f "$ZIMG_PREFIX/lib64/pkgconfig/zimg.pc" ]]; then
  ZPC="$ZIMG_PREFIX/lib64/pkgconfig"
elif [[ -f "$ZIMG_PREFIX/lib/pkgconfig/zimg.pc" ]]; then
  ZPC="$ZIMG_PREFIX/lib/pkgconfig"
elif [[ -f "$ZIMG_PREFIX/lib/x86_64-linux-gnu/pkgconfig/zimg.pc" ]]; then
  ZPC="$ZIMG_PREFIX/lib/x86_64-linux-gnu/pkgconfig"
else
  echo "build_libvmaf: zimg.pc missing under $ZIMG_PREFIX" >&2
  exit 1
fi
export PKG_CONFIG_PATH="${ZPC}${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
if ! pkg-config --exists zimg; then
  echo "build_libvmaf: pkg-config cannot see zimg" >&2
  exit 1
fi
ZVER="$(pkg-config --modversion zimg)"

mkdir -p "$(dirname "$SRC")"
if [[ ! -d "$SRC/.git" ]]; then
  git init -q -b main "$SRC"
  git -C "$SRC" remote add origin https://github.com/Netflix/vmaf.git
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/git_retry.sh"
pixvqa_git_fetch "$SRC" fetch --depth 1 origin "$PIN_COMMIT"
git -C "$SRC" checkout --detach --force FETCH_HEAD
ACTUAL="$(git -C "$SRC" rev-parse HEAD)"
echo "libvmaf source HEAD=$ACTUAL"
if [[ "$ACTUAL" != "$PIN_COMMIT" ]]; then
  echo "build_libvmaf: checkout $ACTUAL != $PIN_COMMIT" >&2
  exit 1
fi
echo "Verified libvmaf pin $PIN_COMMIT (previous tagged release $PREVIOUS_TAG $PREVIOUS_COMMIT)"

rm -rf "$BUILD"
meson setup "$BUILD" "$SRC/libvmaf" \
  --prefix="$PREFIX" \
  --buildtype=release \
  -Denable_tests=false \
  -Denable_docs=false \
  -Denable_tools=true \
  -Denable_cuda=false \
  -Denable_zimg=true \
  -Dbuilt_in_models=true \
  -Denable_float=false \
  -Ddefault_library=static

meson compile -C "$BUILD" -j"$(nproc)"
meson install -C "$BUILD"

# Prefer the pkg-config file installed by THIS pinned build. Meson commonly
# uses lib/x86_64-linux-gnu on Ubuntu; do not silently resolve /usr/local.
PC_FILE="$(find "$PREFIX" -type f -name libvmaf.pc -print -quit)"
if [[ -z "$PC_FILE" ]]; then
  echo "build_libvmaf: libvmaf.pc missing under $PREFIX" >&2
  exit 1
fi
PC_DIR="$(dirname "$PC_FILE")"
LIB_A="$(find "$PREFIX" -type f -name libvmaf.a -print -quit)"
if [[ -z "$LIB_A" ]]; then
  echo "build_libvmaf: libvmaf.a missing under $PREFIX" >&2
  exit 1
fi
export PKG_CONFIG_PATH="$PC_DIR${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
resolved_pc="$(pkg-config --variable=pcfiledir libvmaf)"
if [[ "$resolved_pc" != "$PC_DIR" ]]; then
  echo "build_libvmaf: incorrect pkg-config resolution: $resolved_pc (expected $PC_DIR)" >&2
  exit 1
fi
LIBS="$(pkg-config --libs --static libvmaf)"
echo "libvmaf pkg-config version: $(pkg-config --modversion libvmaf)"
echo "libvmaf cflags: $(pkg-config --cflags libvmaf)"
echo "libvmaf libs: $LIBS"
if [[ "$LIBS" != *"-lzimg"* ]]; then
  echo "build_libvmaf: static libvmaf does not pull -lzimg; refusing install" >&2
  exit 1
fi
if ! grep -a -q zimg_filter_graph_build "$LIB_A"; then
  echo "build_libvmaf: libvmaf.a has no zimg_filter_graph_build; conversion was not compiled in" >&2
  exit 1
fi
if ! grep -q vmaf_picture_convert "$PREFIX/include/libvmaf/picture.h"; then
  echo "build_libvmaf: installed picture.h has no vmaf_picture_convert" >&2
  exit 1
fi
CLI_VER="$("$PREFIX/bin/vmaf" --version 2>&1 || true)"
echo "vmaf cli: $CLI_VER"
{
  echo "libvmaf_commit=$PIN_COMMIT"
  echo "libvmaf_previous_tag=$PREVIOUS_TAG"
  echo "libvmaf_previous_commit=$PREVIOUS_COMMIT"
  echo "libvmaf_pkgconfig_version=$(pkg-config --modversion libvmaf)"
  echo "libvmaf_cli_version=$CLI_VER"
  echo "zimg_version=$ZVER"
  echo "zimg_commit=f819b14e8f39d1282400b0d9543e8ef73c1b2bbd"
  echo "zimg_tag=release-3.0.6"
  echo "meson_enable_zimg=true"
  echo "meson_enable_cuda=false"
  echo "meson_enable_float=false"
  echo "meson_default_library=static"
  echo "note=source pin, not a tagged release newer than v3.2.1"
} > "$PREFIX/pixvqa-libvmaf-stamp.txt"
echo "libvmaf install OK: $PREFIX"
