#!/usr/bin/env bash
# CUDA Pixop VQA without FFmpeg. Output: build/cuda/pixvqa
# The CPU-only binary stays at build/cpu/pixvqa. This script does not replace it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
GCC13="${GCC13:-}"
if [[ -z "$GCC13" || ! -x "$GCC13/bin/x86_64-conda-linux-gnu-g++" ]]; then
  echo "Set GCC13 to a GCC 13 prefix. nvcc 12.8 needs that host compiler." >&2
  echo "Example: micromamba create -y -p \"\$GCC13\" -c conda-forge gxx_linux-64=13.4" >&2
  exit 1
fi
ln -sfn x86_64-conda-linux-gnu-gcc "$GCC13/bin/gcc"
ln -sfn x86_64-conda-linux-gnu-g++ "$GCC13/bin/g++"
ln -sfn x86_64-conda-linux-gnu-gcc "$GCC13/bin/cc"
ln -sfn x86_64-conda-linux-gnu-g++ "$GCC13/bin/c++"
export PATH="$GCC13/bin:/usr/bin:/bin:${PATH}"
CUDA_HOME="${CUDA_HOME:-}"
if [[ -z "$CUDA_HOME" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "Set CUDA_HOME to a CUDA toolkit that provides nvcc." >&2
  exit 1
fi
export PATH="$CUDA_HOME/bin:${PATH}"
# shellcheck source=scripts/libvmaf_flags.sh
source "$ROOT/scripts/libvmaf_flags.sh"
pixvqa_detect_libvmaf "$ROOT"
CXX="${CXX:-$GCC13/bin/g++}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
CXXFLAGS="-O3 -Wall -Wextra -std=c++17 -Iinclude -pthread -march=native -DPIXVQA_HAS_CUDA=1"
LINK_LIBS=""
srcs=(src/yuv_io.cpp src/psnr.cpp src/ssim_ref.cpp src/msssim.cpp
      src/thread_pool.cpp src/xpsnr.cpp src/hdr.cpp src/pu21.cpp)
if [[ "$PIXVQA_HAS_VMAF" == "1" ]]; then
  echo "libvmaf: enabled (CPU VMAF; no CUDA VMAF)"
  CXXFLAGS+=" -DPIXVQA_HAS_VMAF=1 $PIXVQA_VMAF_CFLAGS"
  LINK_LIBS="$PIXVQA_VMAF_LIBS"
  srcs+=(src/vmaf.cpp)
else
  echo "libvmaf: disabled (optional). Run: bash scripts/build_libvmaf.sh"
fi
OUT=build/cuda
mkdir -p "$OUT"
echo "Using $($CXX --version | head -1)"
echo "Using $($NVCC --version | tail -1)"

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

echo "NVCC src/cuda_metrics.cu (sm_120)"
$NVCC -O3 -std=c++17 -Iinclude --compiler-bindir "$GCC13/bin" \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120 \
  -c src/cuda_metrics.cu -o "$OUT/cuda_metrics.o"

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
echo "LINK $OUT/pixvqa"
$CXX $CXXFLAGS -DPIXVQA_BUILD_KIND=\"cuda\" src/main.cpp "${objs[@]}" "$OUT/cuda_metrics.o" \
  -o "$OUT/pixvqa" \
  -L"$CUDA_HOME/lib" -lcudart -L"$PIXVQA_LIBCUDA_DIR" -lcuda \
  -Wl,-rpath,"$CUDA_HOME/lib:$GCC13/lib:$PIXVQA_LIBCUDA_DIR" \
  $LINK_LIBS
echo "CUDA build OK: $OUT/pixvqa"
echo "CPU-only binary: build/cpu/pixvqa (bash build.sh)."
