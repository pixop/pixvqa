#pragma once

#include "pixvqa/types.hpp"

namespace pixvqa {

// Classical single-scale SSIM.
// 11x11 Gaussian, sigma 1.5, K1=0.01, K2=0.03, L = peak = (1<<bit_depth)-1.
// The score is the mean of the luminance×contrast-structure map over the
// valid 11-tap interior. It is not clamped to [0,1].
// full=true → luminance × CS; full=false → CS only.
// YUV avg uses yuv_sample_weight_avg (4:2:0 is 4Y+U+V over 6).
// The caller selects scalar or tiled AVX2 after a CPUID check. Full-frame
// AVX2 remains available to internal callers and is not the public default.
// Five-scale MS-SSIM is a separate entry point and is not selected by SSIM.
double ssim_mean_scalar(PlaneView x, PlaneView y, double L, bool full, int threads = 1);
// Same 11x11 Gaussian, border, and formula as ssim_mean_scalar, on binary64 samples.
double ssim_mean_doubles(const double* x, int xstride, const double* y, int ystride, int width,
                         int height, double L, bool full);
double ssim_mean_avx2(PlaneView x, PlaneView y, double L, bool full, int threads = 1);
double ssim_mean_avx2_fullframe(PlaneView x, PlaneView y, double L, bool full, int threads = 1);
double ssim_mean_avx2_tiled(PlaneView x, PlaneView y, double L, bool full, int threads = 1);

inline double ssim_mean(PlaneView x, PlaneView y, double L, bool full, SsimKernelMode mode,
                        int threads = 1) {
  switch (mode) {
    case SsimKernelMode::Scalar:
      return ssim_mean_scalar(x, y, L, full, threads);
    case SsimKernelMode::Avx2Full:
      return ssim_mean_avx2_fullframe(x, y, L, full, threads);
    case SsimKernelMode::Avx2:
    default:
      return ssim_mean_avx2_tiled(x, y, L, full, threads);
  }
}

PlaneScore compute_ssim_frame(const Frame& ref, const Frame& test, const VideoFormat& fmt,
                              double L, SsimKernelMode mode, int threads);

// Strip height for tiled AVX2 (valid output rows per strip). 0 → default.
void pixvqa_set_strip_height(int rows);
int pixvqa_get_strip_height();

// When PIXVQA_PROFILE=1, dumps aggregated AVX2 SSIM stage timings to stderr.
void pixvqa_dump_ssim_profile();

}  // namespace pixvqa
