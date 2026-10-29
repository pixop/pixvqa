#!/usr/bin/env bash
# Emit GNU Make assignments for one PixVQA build variant (cpu|swdec|cuda|full).
# Sources scripts/deps_layout.sh and scripts/libvmaf_flags.sh; does not build deps.
set -euo pipefail
VARIANT="${1:?usage: make_config.sh cpu|swdec|cuda|full}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# The shared layout helper picks defaults; explicit Make environment overrides win.
user_ffmpeg="${FFMPEG_PREFIX:-}"
user_dav1d="${DAV1D_PREFIX:-}"
user_gcc13="${GCC13:-}"
user_vmaf="${LIBVMAF_PREFIX:-}"
user_zimg="${ZIMG_PREFIX:-}"
user_zimg_src="${ZIMG_SRC:-}"
# shellcheck disable=SC1091
source "$ROOT/scripts/deps_layout.sh"
pixvqa_apply_deps_layout "$ROOT" || exit 1
[[ -z "$user_ffmpeg" ]] || FFMPEG_PREFIX="$user_ffmpeg"
[[ -z "$user_dav1d" ]] || DAV1D_PREFIX="$user_dav1d"
[[ -z "$user_gcc13" ]] || GCC13="$user_gcc13"
[[ -z "$user_vmaf" ]] || LIBVMAF_PREFIX="$user_vmaf"
[[ -z "$user_zimg" ]] || ZIMG_PREFIX="$user_zimg"
[[ -z "$user_zimg_src" ]] || ZIMG_SRC="$user_zimg_src"

# shellcheck disable=SC1091
source "$ROOT/scripts/libvmaf_flags.sh"
pixvqa_detect_libvmaf "$ROOT"

make_quote() {
  local s=$1
  s=${s//\\/\\\\}
  s=${s//$/\$\$}
  s=${s//#/\\#}
  printf '%s' "$s"
}

emit() {
  local name=$1 value=$2
  printf '%s := %s\n' "$name" "$(make_quote "$value")"
}

pixvqa_find_dav1d_lib() {
  if [[ -n "${DAV1D_LIB:-}" ]]; then
    printf '%s\n' "$DAV1D_LIB"
    return 0
  fi
  local candidate pc
  if [[ -n "${DAV1D_PREFIX:-}" ]] && command -v pkg-config >/dev/null 2>&1; then
    pc="$(PKG_CONFIG_PATH="${DAV1D_PREFIX}/lib/pkgconfig:${DAV1D_PREFIX}/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}" \
      pkg-config --variable=libdir dav1d 2>/dev/null || true)"
    if [[ -n "$pc" && -f "$pc/libdav1d.so" ]]; then
      printf '%s\n' "$pc"
      return 0
    fi
  fi
  for candidate in \
      "${DAV1D_PREFIX:-}/lib/x86_64-linux-gnu" \
      "${DAV1D_PREFIX:-}/lib" \
      "${DAV1D_PREFIX:-}/lib64"; do
    if [[ -n "$candidate" && -f "$candidate/libdav1d.so" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
}

pixvqa_resolve_gcc_env() {
  if [[ -n "${GCC_ENV:-}" ]]; then
    printf '%s\n' "$GCC_ENV"
    return 0
  fi
  if [[ -x /usr/bin/g++ ]]; then
    printf '/usr\n'
    return 0
  fi
  printf '\n'
}

pixvqa_resolve_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    printf '%s\n' "$CUDA_HOME"
    return 0
  fi
  for candidate in /usr/local/cuda-12.8 /usr/local/cuda; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '\n'
}

pixvqa_resolve_cudart() {
  local cuda_home=$1
  local candidate
  if [[ -n "${CUDART_LIBDIR:-}" ]]; then
    printf '%s\n' "$CUDART_LIBDIR"
    return 0
  fi
  for candidate in \
      "$cuda_home/targets/x86_64-linux/lib" \
      "$cuda_home/lib64" \
      "$cuda_home/lib" \
      "$cuda_home/lib/x64"; do
    if [[ -f "$candidate/libcudart.so" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '\n'
}

pixvqa_resolve_libcuda_dir() {
  if [[ -n "${PIXVQA_LIBCUDA_DIR:-}" ]]; then
    printf '%s\n' "$PIXVQA_LIBCUDA_DIR"
    return 0
  fi
  if [[ -d /usr/lib/wsl/lib ]]; then
    printf '/usr/lib/wsl/lib\n'
    return 0
  fi
  if [[ -e /usr/lib/x86_64-linux-gnu/libcuda.so || -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]]; then
    printf '/usr/lib/x86_64-linux-gnu\n'
    return 0
  fi
  printf '\n'
}

fail_msg() {
  echo "make_config: $*" >&2
  exit 1
}

GCC_ENV_RESOLVED="$(pixvqa_resolve_gcc_env)"
DAV1D_LIB_RESOLVED="$(pixvqa_find_dav1d_lib)"
CUDA_HOME_RESOLVED="$(pixvqa_resolve_cuda_home)"
CUDART_LIBDIR_RESOLVED=""
LIBCUDA_DIR_RESOLVED="$(pixvqa_resolve_libcuda_dir)"
FULL_LINK_CXX="${PIXVQA_FULL_LINK_CXX:-/usr/bin/g++}"

BASE_CXXFLAGS="-O3 -Wall -Wextra -std=c++17 -Iinclude -pthread -march=native"
LINK_LIBS=""
LDFLAGS=""
VMAF_SRCS=""

if [[ "$PIXVQA_HAS_VMAF" == "1" ]]; then
  BASE_CXXFLAGS+=" -DPIXVQA_HAS_VMAF=1 $PIXVQA_VMAF_CFLAGS"
  LINK_LIBS="$PIXVQA_VMAF_LIBS"
  VMAF_SRCS="src/vmaf.cpp"
fi

case "$VARIANT" in
  cpu)
    BUILD=build/cpu
    CXX="${CXX:-g++}"
    if [[ -n "$GCC_ENV_RESOLVED" ]]; then
      CXX="$GCC_ENV_RESOLVED/bin/g++"
    fi
    [[ -x "$CXX" ]] || fail_msg "no C++ compiler ($CXX); set GCC_ENV or CXX"
    CXXFLAGS="$BASE_CXXFLAGS"
    BUILD_KIND=cpu
    ;;
  swdec)
    BUILD=build/swdec
    [[ -n "$FFMPEG_PREFIX" && -f "$FFMPEG_PREFIX/lib/libavcodec.so" ]] \
      || fail_msg "FFMPEG_PREFIX must point at FFmpeg n8.1.3 (lib/libavcodec.so)"
    [[ -n "$GCC_ENV_RESOLVED" && -x "$GCC_ENV_RESOLVED/bin/g++" ]] \
      || fail_msg "set GCC_ENV to a prefix with bin/g++ (e.g. GCC_ENV=/usr)"
    CXX="${CXX:-$GCC_ENV_RESOLVED/bin/g++}"
    CXXFLAGS="$BASE_CXXFLAGS -I$FFMPEG_PREFIX/include -DPIXVQA_HAS_FFMPEG=1"
    LDFLAGS="$FFMPEG_PREFIX/lib/libavformat.so $FFMPEG_PREFIX/lib/libavcodec.so $FFMPEG_PREFIX/lib/libavutil.so -Wl,--disable-new-dtags,-rpath,$FFMPEG_PREFIX/lib -pthread"
    if [[ -n "$DAV1D_LIB_RESOLVED" ]]; then
      LDFLAGS+=" -L$DAV1D_LIB_RESOLVED -Wl,--disable-new-dtags,-rpath,$DAV1D_LIB_RESOLVED -Wl,-rpath-link,$DAV1D_LIB_RESOLVED"
    fi
    LDFLAGS+=" $LINK_LIBS"
    BUILD_KIND=cpu
    ;;
  cuda)
    BUILD=build/cuda
    [[ -n "$GCC13" && -x "$GCC13/bin/x86_64-conda-linux-gnu-g++" ]] \
      || fail_msg "set GCC13 to conda-forge gxx_linux-64=13.4 (nvcc host compiler)"
    [[ -n "$CUDA_HOME_RESOLVED" ]] || fail_msg "set CUDA_HOME to a toolkit with nvcc"
    CUDART_LIBDIR_RESOLVED="$(pixvqa_resolve_cudart "$CUDA_HOME_RESOLVED")"
    [[ -n "$CUDART_LIBDIR_RESOLVED" ]] || fail_msg "libcudart.so not found under CUDA_HOME"
    [[ -n "$LIBCUDA_DIR_RESOLVED" ]] || fail_msg "set PIXVQA_LIBCUDA_DIR or install NVIDIA driver libs"
    CXX="$GCC13/bin/g++"
    CXXFLAGS="$BASE_CXXFLAGS -DPIXVQA_HAS_CUDA=1"
    LDFLAGS="-L$CUDART_LIBDIR_RESOLVED -lcudart -L$LIBCUDA_DIR_RESOLVED -lcuda"
    LDFLAGS+=" -Wl,--disable-new-dtags,-rpath,$CUDART_LIBDIR_RESOLVED:$GCC13/lib:$LIBCUDA_DIR_RESOLVED"
    LDFLAGS+=" $LINK_LIBS -L$GCC13/lib -lstdc++"
    NVCC="$CUDA_HOME_RESOLVED/bin/nvcc"
    BUILD_KIND=cuda
    ;;
  full)
    BUILD=build/full
    [[ -n "$FFMPEG_PREFIX" && -f "$FFMPEG_PREFIX/lib/libavcodec.so" ]] \
      || fail_msg "FFMPEG_PREFIX must point at FFmpeg n8.1.3"
    [[ -n "$GCC13" && -x "$GCC13/bin/g++" ]] \
      || fail_msg "set GCC13 for CUDA host compile and libstdc++ link"
    [[ -n "$CUDA_HOME_RESOLVED" ]] || fail_msg "set CUDA_HOME"
    CUDART_LIBDIR_RESOLVED="$(pixvqa_resolve_cudart "$CUDA_HOME_RESOLVED")"
    [[ -n "$CUDART_LIBDIR_RESOLVED" ]] || fail_msg "libcudart.so not found"
    [[ -n "$LIBCUDA_DIR_RESOLVED" ]] || fail_msg "NVIDIA driver library directory unknown"
    [[ -x "$FULL_LINK_CXX" ]] || fail_msg "final link driver missing: $FULL_LINK_CXX"
    CXX="$GCC13/bin/g++"
    FULL_RPATH="$CUDART_LIBDIR_RESOLVED:$GCC13/lib:$FFMPEG_PREFIX/lib:$LIBCUDA_DIR_RESOLVED"
    if [[ -n "$DAV1D_LIB_RESOLVED" ]]; then
      FULL_RPATH="$DAV1D_LIB_RESOLVED:$FULL_RPATH"
    fi
    CXXFLAGS="$BASE_CXXFLAGS -I$FFMPEG_PREFIX/include -I$CUDA_HOME_RESOLVED/include -I$CUDA_HOME_RESOLVED/targets/x86_64-linux/include -DPIXVQA_HAS_CUDA=1 -DPIXVQA_HAS_FFMPEG=1 -DPIXVQA_HAS_NVDEC=1"
    LDFLAGS="-L$GCC13/lib -L$CUDART_LIBDIR_RESOLVED -lcudart -L$LIBCUDA_DIR_RESOLVED -lcuda"
    LDFLAGS+=" $FFMPEG_PREFIX/lib/libavformat.so $FFMPEG_PREFIX/lib/libavcodec.so $FFMPEG_PREFIX/lib/libavutil.so"
    LDFLAGS+=" -Wl,--disable-new-dtags,-rpath,$FULL_RPATH"
    if [[ -n "$DAV1D_LIB_RESOLVED" ]]; then
      LDFLAGS+=" -L$DAV1D_LIB_RESOLVED -Wl,-rpath-link,$DAV1D_LIB_RESOLVED"
    fi
    LDFLAGS+=" $LINK_LIBS -L$GCC13/lib -lstdc++"
    NVCC="$CUDA_HOME_RESOLVED/bin/nvcc"
    BUILD_KIND=cuda
    ;;
  *)
    fail_msg "unknown variant $VARIANT"
    ;;
esac

LIB_SRCS="$ROOT/src/yuv_io.cpp $ROOT/src/psnr.cpp $ROOT/src/ssim_ref.cpp $ROOT/src/msssim.cpp \
$ROOT/src/thread_pool.cpp $ROOT/src/xpsnr.cpp $ROOT/src/hdr.cpp $ROOT/src/pu21.cpp"
case "$VARIANT" in
  swdec|full) LIB_SRCS+=" $ROOT/src/decode.cpp" ;;
esac
if [[ "$PIXVQA_HAS_VMAF" == "1" ]]; then
  LIB_SRCS+=" $ROOT/src/vmaf.cpp"
fi

CONFIG_ID="$(printf '%s\0' "$VARIANT" "$CXX" "$CXXFLAGS" "$LDFLAGS" "${NVCC:-}" "${GCC13:-}" "$FFMPEG_PREFIX" "$DAV1D_LIB_RESOLVED" "$LINK_LIBS" | sha256sum | awk '{print $1}')"

UP="${VARIANT^^}"

{
  emit VARIANT "$VARIANT"
  emit BUILD "$BUILD"
  emit CONFIG_ID "$CONFIG_ID"
  emit "${UP}_LIB_SRCS" "$LIB_SRCS"
  emit "${UP}_CXX" "$CXX"
  emit "${UP}_FULL_LINK_CXX" "$FULL_LINK_CXX"
  emit "${UP}_CXXFLAGS" "$CXXFLAGS"
  emit "${UP}_LDFLAGS" "$LDFLAGS"
  emit "${UP}_BUILD_KIND" "$BUILD_KIND"
  emit "${UP}_FFMPEG_PREFIX" "${FFMPEG_PREFIX:-}"
  emit "${UP}_DAV1D_LIB" "$DAV1D_LIB_RESOLVED"
  emit "${UP}_GCC13" "${GCC13:-}"
  emit "${UP}_CUDA_HOME" "${CUDA_HOME_RESOLVED:-}"
  emit "${UP}_CUDART_LIBDIR" "${CUDART_LIBDIR_RESOLVED:-}"
  emit "${UP}_LIBCUDA_DIR" "${LIBCUDA_DIR_RESOLVED:-}"
  emit "${UP}_NVCC" "${NVCC:-}"
  emit "${UP}_LINK_LIBS" "$LINK_LIBS"
  emit "${UP}_HAS_VMAF" "$PIXVQA_HAS_VMAF"
}
