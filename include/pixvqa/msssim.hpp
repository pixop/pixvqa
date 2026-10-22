#pragma once

#include "pixvqa/types.hpp"

namespace pixvqa {

// Five-scale product MS-SSIM. Separate from single-scale SSIM.
// Five published weights, not renormalized. Fewer scales only when a
// coarser level would be smaller than the 11-tap window.

struct MsssimScaleInfo {
  int height = 0;
  int width = 0;
  double cs = 0;    // spatial mean of the contrast/structure map, before ReLU
  double ssim = 0;  // spatial mean of the full SSIM map, before ReLU
};

struct MsssimPlaneInfo {
  int scales = 0;
  double score = 0;
  MsssimScaleInfo scale[5]{};
};

// Published weights. Decimal sum is 1.0001. Do not renormalize.
inline constexpr double kMsssimWeight[5] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
inline constexpr int kMsssimWin = 11;

int msssim_scales(int height, int width);

// CPU entry. AVX2 double is the default. PIXVQA_MSSSIM_BACKEND=scalar selects
// the reference implementation. The variable is read on the first call.
double msssim_plane(PlaneView ref, PlaneView dist, double peak, int threads,
                         MsssimPlaneInfo* detail);

double msssim_plane_scalar(PlaneView ref, PlaneView dist, double peak, int threads,
                                MsssimPlaneInfo* detail);
// Same five-scale product, weights, pooling, and borders as msssim_plane_scalar.
double msssim_plane_doubles(const double* ref, const double* dist, int width, int height, double peak,
                            int threads);
double msssim_plane_avx2(PlaneView ref, PlaneView dist, double peak, int threads,
                              MsssimPlaneInfo* detail);

PlaneScore compute_msssim_frame(const Frame& ref, const Frame& test, const VideoFormat& fmt,
                                     double peak, int threads);

}  // namespace pixvqa
