#include "pixvqa/psnr.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace pixvqa {

double plane_mse(PlaneView a, PlaneView b) {
  if (a.width != b.width || a.height != b.height) {
    throw std::runtime_error("plane_mse shape mismatch");
  }
  long double acc = 0.0;
  const size_t n = static_cast<size_t>(a.width) * a.height;
  for (int y = 0; y < a.height; ++y) {
    const uint16_t* pa = a.data + static_cast<size_t>(y) * a.stride;
    const uint16_t* pb = b.data + static_cast<size_t>(y) * b.stride;
    for (int x = 0; x < a.width; ++x) {
      const double d = static_cast<double>(pa[x]) - static_cast<double>(pb[x]);
      acc += d * d;
    }
  }
  return static_cast<double>(acc / static_cast<long double>(n));
}

static double mse_to_psnr(double mse, double peak) {
  if (mse <= 0.0) return std::numeric_limits<double>::infinity();
  return 10.0 * std::log10((peak * peak) / mse);
}

PsnrFrame compute_psnr_frame(const Frame& ref, const Frame& test, const VideoFormat& fmt,
                             double peak) {
  PsnrFrame out;
  out.mse.y = plane_mse(ref.y.view(), test.y.view());
  out.mse.u = plane_mse(ref.u.view(), test.u.view());
  out.mse.v = plane_mse(ref.v.view(), test.v.view());
  out.mse.avg = yuv_sample_weight_avg(out.mse.y, out.mse.u, out.mse.v, fmt.chroma);
  out.psnr.y = mse_to_psnr(out.mse.y, peak);
  out.psnr.u = mse_to_psnr(out.mse.u, peak);
  out.psnr.v = mse_to_psnr(out.mse.v, peak);
  out.psnr.avg = mse_to_psnr(out.mse.avg, peak);
  return out;
}

PsnrFrame psnr_from_mean_mse(double mse_y, double mse_u, double mse_v, ChromaFormat fmt,
                             double peak) {
  PsnrFrame out;
  out.mse.y = mse_y;
  out.mse.u = mse_u;
  out.mse.v = mse_v;
  out.mse.avg = yuv_sample_weight_avg(mse_y, mse_u, mse_v, fmt);
  out.psnr.y = mse_to_psnr(out.mse.y, peak);
  out.psnr.u = mse_to_psnr(out.mse.u, peak);
  out.psnr.v = mse_to_psnr(out.mse.v, peak);
  out.psnr.avg = mse_to_psnr(out.mse.avg, peak);
  return out;
}

}  // namespace pixvqa
