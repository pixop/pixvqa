#!/usr/bin/env bash
# FFmpeg decode builds. Links the LGPL n8.1.3 libraries in FFMPEG_PREFIX.
# Software decode: build/swdec/pixvqa
# Full CUDA + software + NVDEC: build/full/pixvqa
# Neither replaces build/cpu/pixvqa or build/cuda/pixvqa.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -z "${FFMPEG_PREFIX:-}" ]]; then
  echo "Set FFMPEG_PREFIX to an FFmpeg n8.1.3 LGPL shared-library prefix." >&2
  echo "Example: FFMPEG_PREFIX=/path/to/ffmpeg-n8.1.3-lgpl GCC_ENV=/path/to/gcc bash build_decode.sh" >&2
  exit 1
fi
if [[ ! -f "$FFMPEG_PREFIX/lib/libavcodec.so" ]]; then
  echo "FFMPEG_PREFIX=$FFMPEG_PREFIX has no lib/libavcodec.so." >&2
  exit 1
fi
if [[ ! -f "$FFMPEG_PREFIX/include/libavcodec/avcodec.h" ]]; then
  echo "FFMPEG_PREFIX=$FFMPEG_PREFIX has no include/libavcodec/avcodec.h." >&2
  exit 1
fi

if [[ -z "${GCC_ENV:-}" ]]; then
  echo "Set GCC_ENV to a GCC prefix with bin/g++ for the software-decode build." >&2
  echo "Example: FFMPEG_PREFIX=/path/to/ffmpeg-n8.1.3-lgpl GCC_ENV=/path/to/gcc bash build_decode.sh" >&2
  exit 1
fi
if [[ ! -x "$GCC_ENV/bin/g++" ]]; then
  echo "GCC_ENV=$GCC_ENV has no executable bin/g++." >&2
  exit 1
fi
export PATH="${GCC_ENV}/bin:${PATH}"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "C++ compiler '$CXX' was not found on PATH after adding GCC_ENV/bin." >&2
  exit 1
fi

# shellcheck source=scripts/libvmaf_flags.sh
source "$ROOT/scripts/libvmaf_flags.sh"
pixvqa_detect_libvmaf "$ROOT"
CXXFLAGS="-O3 -Wall -Wextra -std=c++17 -Iinclude -I$FFMPEG_PREFIX/include -pthread -march=native -DPIXVQA_HAS_FFMPEG=1"
LDFLAGS="-L$FFMPEG_PREFIX/lib -lavformat -lavcodec -lavutil -Wl,-rpath,$FFMPEG_PREFIX/lib -pthread"
# libdav1d is a dependency of libavcodec, not of PixVQA itself. The linker still
# has to see it, and the binary has to find it at run time.
if [[ -n "${DAV1D_LIB:-}" ]]; then
  LDFLAGS+=" -L$DAV1D_LIB -Wl,-rpath,$DAV1D_LIB -Wl,-rpath-link,$DAV1D_LIB"
fi
srcs=(src/yuv_io.cpp src/psnr.cpp src/ssim_ref.cpp src/msssim.cpp
      src/thread_pool.cpp src/xpsnr.cpp src/hdr.cpp src/pu21.cpp src/decode.cpp)
if [[ "$PIXVQA_HAS_VMAF" == "1" ]]; then
  echo "libvmaf: enabled (CPU VMAF)"
  CXXFLAGS+=" -DPIXVQA_HAS_VMAF=1 $PIXVQA_VMAF_CFLAGS"
  LDFLAGS+=" $PIXVQA_VMAF_LIBS"
  srcs+=(src/vmaf.cpp)
else
  echo "libvmaf: disabled (optional). Run: bash scripts/build_libvmaf.sh"
fi
OUT=build/swdec
mkdir -p "$OUT"
echo "Using $($CXX --version | head -1)"
echo "FFmpeg prefix $FFMPEG_PREFIX"
echo "GCC_ENV $GCC_ENV"

objs=()
for src in "${srcs[@]}"; do
  obj="$OUT/$(basename "${src%.cpp}").o"
  echo "CXX $src"
  $CXX $CXXFLAGS -c "$src" -o "$obj"
  objs+=("$obj")
done
echo "CXX src/ssim_simd.cpp (AVX2)"
$CXX $CXXFLAGS -mavx2 -mfma -c src/ssim_simd.cpp -o "$OUT/ssim_simd.o"
objs+=("$OUT/ssim_simd.o")
echo "CXX src/msssim_avx2.cpp (AVX2, no FMA contract)"
$CXX $CXXFLAGS -mavx2 -ffp-contract=off -fno-fast-math -c src/msssim_avx2.cpp -o "$OUT/msssim_avx2.o"
objs+=("$OUT/msssim_avx2.o")
echo "LINK $OUT/pixvqa"
$CXX $CXXFLAGS -DPIXVQA_BUILD_KIND=\"cpu\" src/main.cpp "${objs[@]}" -o "$OUT/pixvqa" $LDFLAGS
echo "Software decode build OK: $OUT/pixvqa"

ASAN_REQ="${PIXVQA_DECODE_ASAN:-${PIXVQA_DECODE_ASAN:-}}"
if [[ "$ASAN_REQ" == "1" ]]; then
  echo "ASan software decode build"
  ASAN_DIR=build/swdec-asan
  mkdir -p "$ASAN_DIR"
  ASAN="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined"
  asan_objs=()
  asan_srcs=("${srcs[@]}" src/ssim_simd.cpp src/msssim_avx2.cpp)
  for src in "${asan_srcs[@]}"; do
    extra=""
    case "$src" in
      *ssim_simd.cpp) extra="-mavx2 -mfma" ;;
      *msssim_avx2.cpp) extra="-mavx2 -ffp-contract=off -fno-fast-math" ;;
    esac
    obj="$ASAN_DIR/$(basename "${src%.cpp}").o"
    $CXX $CXXFLAGS $ASAN $extra -c "$src" -o "$obj"
    asan_objs+=("$obj")
  done
  $CXX $CXXFLAGS $ASAN -DPIXVQA_BUILD_KIND=\"cpu\" src/main.cpp "${asan_objs[@]}" -o "$ASAN_DIR/pixvqa" $LDFLAGS
  echo "ASan decode build OK: $ASAN_DIR/pixvqa"
fi

if [[ -n "${GCC13:-}" && -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  if [[ ! -f build/cuda/cuda_metrics.o ]]; then
    echo "build/cuda/cuda_metrics.o is missing. Run bash build_cuda.sh before the full build." >&2
    exit 1
  fi
  if [[ ! -x "$GCC13/bin/g++" ]]; then
    echo "GCC13=$GCC13 has no executable bin/g++." >&2
    exit 1
  fi
  FULL=build/full
  mkdir -p "$FULL"
  DEC_CXX="$GCC13/bin/g++"
  echo "CXX src/decode.cpp (CUDA + NVDEC)"
  $DEC_CXX -O3 -Wall -Wextra -std=c++17 -Iinclude -I"$FFMPEG_PREFIX/include" -I"$CUDA_HOME/include" -I"$CUDA_HOME/targets/x86_64-linux/include" -pthread -march=native \
    -DPIXVQA_HAS_CUDA=1 -DPIXVQA_HAS_FFMPEG=1 -DPIXVQA_HAS_NVDEC=1 \
    -c src/decode.cpp -o "$FULL/decode.o"
  echo "CXX hdr/pu21/ssim/msssim (CUDA link)"
  for src in hdr.cpp pu21.cpp ssim_ref.cpp msssim.cpp; do
    $DEC_CXX -O3 -Wall -Wextra -std=c++17 -Iinclude -pthread -march=native \
      -DPIXVQA_HAS_CUDA=1 -c "src/$src" -o "build/cuda/${src%.cpp}.o"
  done
  echo "LINK $FULL/pixvqa"
  FULL_CXXFLAGS="-O3 -Wall -Wextra -std=c++17 -Iinclude -I$FFMPEG_PREFIX/include -I$CUDA_HOME/include -I$CUDA_HOME/targets/x86_64-linux/include -pthread -march=native -DPIXVQA_HAS_CUDA=1 -DPIXVQA_HAS_FFMPEG=1 -DPIXVQA_HAS_NVDEC=1"
  FULL_OBJS=(build/cuda/yuv_io.o build/cuda/psnr.o build/cuda/ssim_ref.o
             build/cuda/msssim.o build/cuda/thread_pool.o build/cuda/xpsnr.o build/cuda/hdr.o build/cuda/pu21.o
             build/cuda/ssim_simd.o build/cuda/msssim_avx2.o
             build/cuda/cuda_metrics.o "$FULL/decode.o")
  if [[ -z "${PIXVQA_LIBCUDA_DIR:-}" ]]; then
    if [[ -d /usr/lib/wsl/lib ]]; then
      PIXVQA_LIBCUDA_DIR=/usr/lib/wsl/lib
    elif [[ -e /usr/lib/x86_64-linux-gnu/libcuda.so || -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]]; then
      PIXVQA_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu
    else
      echo "Set PIXVQA_LIBCUDA_DIR to the directory that contains libcuda.so.1." >&2
      exit 1
    fi
  fi
  # Link using the native OS toolchain: FFmpeg depends on the Ubuntu glibc ABI.
  # CUDA object files were produced with GCC13, but the final driver must not
  # link against the older Conda sysroot (GLIBC_2.35 failure).
  FULL_LINK_CXX="${PIXVQA_FULL_LINK_CXX:-/usr/bin/g++}"
  [[ -x "$FULL_LINK_CXX" ]] || { echo "Missing system linker driver: $FULL_LINK_CXX" >&2; exit 1; }
  CUDART_LIBDIR="${CUDART_LIBDIR:-${CUDA_HOME}/targets/x86_64-linux/lib}"
  [[ -f "$CUDART_LIBDIR/libcudart.so" ]] || { echo "Missing CUDA runtime library: $CUDART_LIBDIR/libcudart.so" >&2; exit 1; }
  FULL_RPATH="$CUDART_LIBDIR:$GCC13/lib:$FFMPEG_PREFIX/lib:$PIXVQA_LIBCUDA_DIR"
  if [[ -n "${DAV1D_LIB:-}" ]]; then
    FULL_RPATH="$DAV1D_LIB:$FULL_RPATH"
  fi
  FULL_LIBS=(-L"$GCC13/lib" -L"$CUDART_LIBDIR" -lcudart -L"$PIXVQA_LIBCUDA_DIR" -lcuda
             -L"$FFMPEG_PREFIX/lib" -lavformat -lavcodec -lavutil
             -Wl,-rpath,"$FULL_RPATH")
  if [[ -n "${DAV1D_LIB:-}" ]]; then
    FULL_LIBS+=(-L"$DAV1D_LIB" -Wl,-rpath-link,"$DAV1D_LIB")
  fi
  if [[ "$PIXVQA_HAS_VMAF" == "1" ]]; then
    FULL_CXXFLAGS+=" -DPIXVQA_HAS_VMAF=1 $PIXVQA_VMAF_CFLAGS"
    FULL_OBJS+=(build/cuda/vmaf.o)
    FULL_LIBS+=($PIXVQA_VMAF_LIBS)
  fi
  echo "Final link compiler: $FULL_LINK_CXX (system glibc)"
  "$FULL_LINK_CXX" $FULL_CXXFLAGS -DPIXVQA_BUILD_KIND=\"cuda\" src/main.cpp "${FULL_OBJS[@]}" \
    -o "$FULL/pixvqa" "${FULL_LIBS[@]}"
  echo "Full build OK: $FULL/pixvqa"
fi
