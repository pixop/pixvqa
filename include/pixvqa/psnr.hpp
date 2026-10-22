#pragma once

#include "pixvqa/types.hpp"

namespace pixvqa {

double plane_mse(PlaneView a, PlaneView b);
PsnrFrame compute_psnr_frame(const Frame& ref, const Frame& test, const VideoFormat& fmt, double peak);

// Sequence summary from accumulated per-frame plane MSEs (arithmetic mean).
PsnrFrame psnr_from_mean_mse(double mse_y, double mse_u, double mse_v, ChromaFormat fmt, double peak);

}  // namespace pixvqa
