#!/usr/bin/env bash
# Build pinned Netflix libvmaf 3.2.1 (CPU only) into deps/libvmaf-3.2.1.
# Tag/commit are verified before configure. CUDA is never enabled.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN_TAG="v3.2.1"
PIN_COMMIT="f85a853692a8c730d0270cd733c8bb30b5b93b7c"
SRC="${LIBVMAF_SRC:-$ROOT/deps/src/vmaf}"
PREFIX="${LIBVMAF_PREFIX:-$ROOT/deps/libvmaf-3.2.1}"
BUILD="${LIBVMAF_BUILD:-$ROOT/deps/build/libvmaf-3.2.1}"

if [[ -n "${GCC_ENV:-}" ]]; then
  export PATH="${GCC_ENV}/bin:${PATH}"
elif [[ -d "${HOME}/gcc-env/bin" ]]; then
  export PATH="${HOME}/gcc-env/bin:${PATH}"
fi

for tool in git meson ninja nasm pkg-config; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "build_libvmaf: missing required tool '$tool' on PATH" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/deps/src"
if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth 1 --branch "$PIN_TAG" https://github.com/Netflix/vmaf.git "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin "refs/tags/${PIN_TAG}:refs/tags/${PIN_TAG}" || true
  git -C "$SRC" checkout -f "$PIN_TAG"
fi

ACTUAL="$(git -C "$SRC" rev-parse HEAD)"
echo "libvmaf source HEAD=$ACTUAL"
if [[ "$ACTUAL" != "$PIN_COMMIT" ]]; then
  echo "build_libvmaf: tag $PIN_TAG resolved to $ACTUAL, expected $PIN_COMMIT" >&2
  exit 1
fi
echo "Verified $PIN_TAG == $PIN_COMMIT"

rm -rf "$BUILD"
meson setup "$BUILD" "$SRC/libvmaf" \
  --prefix="$PREFIX" \
  --buildtype=release \
  -Denable_tests=false \
  -Denable_docs=false \
  -Denable_tools=true \
  -Denable_cuda=false \
  -Dbuilt_in_models=true \
  -Denable_float=false \
  -Ddefault_library=static

meson compile -C "$BUILD" -j"$(nproc)"
meson install -C "$BUILD"

PC_DIR=""
if [[ -f "$PREFIX/lib64/pkgconfig/libvmaf.pc" ]]; then
  PC_DIR="$PREFIX/lib64/pkgconfig"
elif [[ -f "$PREFIX/lib/pkgconfig/libvmaf.pc" ]]; then
  PC_DIR="$PREFIX/lib/pkgconfig"
fi
export PKG_CONFIG_PATH="${PC_DIR}${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
echo "libvmaf version: $(pkg-config --modversion libvmaf)"
echo "libvmaf cflags: $(pkg-config --cflags libvmaf)"
echo "libvmaf libs: $(pkg-config --libs --static libvmaf)"
"$PREFIX/bin/vmaf" --version
echo "libvmaf install OK: $PREFIX"
