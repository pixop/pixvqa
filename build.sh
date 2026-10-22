#!/usr/bin/env bash
# CPU-only Pixop VQA. No FFmpeg and no CUDA. Output: build/cpu/pixvqa
# Links pinned libvmaf 3.2.1 from deps/libvmaf-3.2.1 when present (optional).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ -n "${GCC_ENV:-}" ]]; then
  export PATH="${GCC_ENV}/bin:${PATH}"
fi
# shellcheck source=scripts/libvmaf_flags.sh
source "$ROOT/scripts/libvmaf_flags.sh"
pixvqa_detect_libvmaf "$ROOT"

CXX="${CXX:-g++}"
CXXFLAGS="-O3 -Wall -Wextra -std=c++17 -Iinclude -pthread -march=native"
LINK_LIBS=""
srcs=(src/yuv_io.cpp src/psnr.cpp src/ssim_ref.cpp src/msssim.cpp
      src/thread_pool.cpp src/xpsnr.cpp src/hdr.cpp src/pu21.cpp)
if [[ "$PIXVQA_HAS_VMAF" == "1" ]]; then
  echo "libvmaf: enabled ($PIXVQA_VMAF_CFLAGS)"
  CXXFLAGS+=" -DPIXVQA_HAS_VMAF=1 $PIXVQA_VMAF_CFLAGS"
  LINK_LIBS="$PIXVQA_VMAF_LIBS"
  srcs+=(src/vmaf.cpp)
else
  echo "libvmaf: disabled (optional). Run: bash scripts/build_libvmaf.sh"
fi

OUT=build/cpu
mkdir -p "$OUT"
echo "Using $($CXX --version | head -1)"

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
$CXX $CXXFLAGS -DPIXVQA_BUILD_KIND=\"cpu\" src/main.cpp "${objs[@]}" -o "$OUT/pixvqa" $LINK_LIBS
echo "LINK $OUT/selftest"
$CXX $CXXFLAGS src/selftest.cpp "${objs[@]}" -o "$OUT/selftest" $LINK_LIBS
echo "RUN selftest"
"$OUT/selftest"
echo "RUN IO regressions"
python3 "$ROOT/scripts/test_io_regressions.py"
echo "RUN golden-log compatibility"
python3 "$ROOT/scripts/test_golden_logs.py"
echo "RUN production-parser compatibility"
python3 "$ROOT/scripts/test_production_parsers.py"
echo "Build OK: $OUT/pixvqa"
