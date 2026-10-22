#include "pixvqa/cpu_dispatch.hpp"
#include "pixvqa/msssim.hpp"
#include "pixvqa/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pixvqa {

namespace {

int pool_out(int n) {
  const int pad = n & 1;
  return (n + 2 * pad - 2) / 2 + 1;
}

void gaussian_kernel(double k[kMsssimWin]) {
  const double sigma = 1.5;
  double sum = 0.0;
  for (int i = 0; i < kMsssimWin; ++i) {
    const double x = static_cast<double>(i - (kMsssimWin / 2));
    k[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
    sum += k[i];
  }
  for (int i = 0; i < kMsssimWin; ++i) k[i] /= sum;
}

void to_double(PlaneView src, std::vector<double>& dst) {
  dst.resize(static_cast<size_t>(src.height) * src.width);
  for (int y = 0; y < src.height; ++y) {
    const uint16_t* row = src.data + static_cast<size_t>(y) * src.stride;
    double* out = dst.data() + static_cast<size_t>(y) * src.width;
    for (int x = 0; x < src.width; ++x) out[x] = static_cast<double>(row[x]);
  }
}

// Non-overlapping 2x2 mean. Odd axes are zero-padded on both sides first,
// matching pytorch-msssim's avg_pool2d(kernel=2, padding=size%2).
void avg_pool2(const std::vector<double>& src, int h, int w, std::vector<double>& dst, int& oh,
               int& ow) {
  const int pad_h = h & 1;
  const int pad_w = w & 1;
  oh = pool_out(h);
  ow = pool_out(w);
  dst.assign(static_cast<size_t>(oh) * ow, 0.0);
  for (int y = 0; y < oh; ++y) {
    for (int x = 0; x < ow; ++x) {
      double sum = 0.0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const int sy = y * 2 + dy - pad_h;
          const int sx = x * 2 + dx - pad_w;
          if (sy >= 0 && sy < h && sx >= 0 && sx < w) {
            sum += src[static_cast<size_t>(sy) * w + sx];
          }
        }
      }
      dst[static_cast<size_t>(y) * ow + x] = sum * 0.25;
    }
  }
}

void filter_separable(const std::vector<double>& src, int h, int w, const double k[kMsssimWin],
                      std::vector<double>& dst, int threads) {
  const int ow = w - (kMsssimWin - 1);
  const int oh = h - (kMsssimWin - 1);
  std::vector<double> tmp(static_cast<size_t>(h) * ow);
  auto horiz = [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const double* row = src.data() + static_cast<size_t>(y) * w;
      double* out = tmp.data() + static_cast<size_t>(y) * ow;
      for (int x = 0; x < ow; ++x) {
        double s = 0.0;
        for (int t = 0; t < kMsssimWin; ++t) s += k[t] * row[x + t];
        out[x] = s;
      }
    }
  };
  if (threads <= 1 || h < 32) horiz(0, h);
  else parallel_for_ranges(threads, 0, h, horiz);

  dst.resize(static_cast<size_t>(oh) * ow);
  auto vert = [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      double* out = dst.data() + static_cast<size_t>(y) * ow;
      for (int x = 0; x < ow; ++x) {
        double s = 0.0;
        for (int t = 0; t < kMsssimWin; ++t) {
          s += k[t] * tmp[static_cast<size_t>(y + t) * ow + x];
        }
        out[x] = s;
      }
    }
  };
  if (threads <= 1 || oh < 32) vert(0, oh);
  else parallel_for_ranges(threads, 0, oh, vert);
}

bool msssim_profile_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char* p = pix_getenv("MSSSIM_PROFILE");
    enabled = (p && p[0] == '1') ? 1 : 0;
  }
  return enabled != 0;
}

bool msssim_force_scalar() {
  static int scalar = -1;
  if (scalar < 0) {
    const char* p = pix_getenv("MSSSIM_BACKEND");
    scalar = (p && std::strcmp(p, "scalar") == 0) ? 1 : 0;
  }
  return scalar != 0;
}

struct StageUs {
  double convert = 0;
  double products = 0;
  double filter = 0;
  double formula = 0;
  double pool = 0;
};

}  // namespace

int msssim_scales(int height, int width) {
  int scales = 0;
  int h = height;
  int w = width;
  for (int i = 0; i < 5; ++i) {
    if (h < kMsssimWin || w < kMsssimWin) break;
    ++scales;
    if (i == 4) break;
    h = pool_out(h);
    w = pool_out(w);
  }
  return scales;
}

double msssim_plane_scalar(PlaneView ref, PlaneView dist, double peak, int threads,
                                MsssimPlaneInfo* detail) {
  if (ref.width != dist.width || ref.height != dist.height) {
    throw std::runtime_error("msssim plane shape mismatch");
  }
  if (threads < 1) threads = 1;
  const int scales = msssim_scales(ref.height, ref.width);
  if (scales < 1) {
    throw std::runtime_error("plane too small for MS-SSIM (need at least 11x11)");
  }
  double kernel[kMsssimWin];
  gaussian_kernel(kernel);
  const double c1 = (0.01 * peak) * (0.01 * peak);
  const double c2 = (0.03 * peak) * (0.03 * peak);
  const bool profile = msssim_profile_enabled();
  StageUs stage;
  using Clock = std::chrono::steady_clock;
  auto stamp = []() { return Clock::now(); };
  auto add_us = [](double& slot, Clock::time_point a, Clock::time_point b) {
    slot += std::chrono::duration<double, std::micro>(b - a).count();
  };

  std::vector<double> cur_x, cur_y, next;
  auto t0 = stamp();
  to_double(ref, cur_x);
  to_double(dist, cur_y);
  if (profile) add_us(stage.convert, t0, stamp());
  int h = ref.height;
  int w = ref.width;
  double terms[5];
  MsssimPlaneInfo local;
  local.scales = scales;

  for (int scale = 0; scale < scales; ++scale) {
    std::vector<double> mux, muy, xx, yy, xy;
    t0 = stamp();
    std::vector<double> x2(cur_x.size()), y2(cur_y.size()), xy_src(cur_x.size());
    for (size_t i = 0; i < cur_x.size(); ++i) {
      x2[i] = cur_x[i] * cur_x[i];
      y2[i] = cur_y[i] * cur_y[i];
      xy_src[i] = cur_x[i] * cur_y[i];
    }
    if (profile) add_us(stage.products, t0, stamp());
    t0 = stamp();
    filter_separable(cur_x, h, w, kernel, mux, threads);
    filter_separable(cur_y, h, w, kernel, muy, threads);
    filter_separable(x2, h, w, kernel, xx, threads);
    filter_separable(y2, h, w, kernel, yy, threads);
    filter_separable(xy_src, h, w, kernel, xy, threads);
    if (profile) add_us(stage.filter, t0, stamp());
    const int oh = h - (kMsssimWin - 1);
    const int ow = w - (kMsssimWin - 1);
    const double n = static_cast<double>(oh) * static_cast<double>(ow);
    double sum_cs = 0.0;
    double sum_ssim = 0.0;
    const size_t count = static_cast<size_t>(oh) * ow;
    t0 = stamp();
    for (size_t i = 0; i < count; ++i) {
      const double vx = xx[i] - mux[i] * mux[i];
      const double vy = yy[i] - muy[i] * muy[i];
      const double cov = xy[i] - mux[i] * muy[i];
      const double cs = (2.0 * cov + c2) / (vx + vy + c2);
      const double lum = (2.0 * mux[i] * muy[i] + c1) / (mux[i] * mux[i] + muy[i] * muy[i] + c1);
      sum_cs += cs;
      sum_ssim += lum * cs;
    }
    if (profile) add_us(stage.formula, t0, stamp());
    const double cs = sum_cs / n;
    const double ssim = sum_ssim / n;
    local.scale[scale] = MsssimScaleInfo{h, w, cs, ssim};
    const bool last = scale + 1 == scales;
    const double used = std::max(0.0, last ? ssim : cs);
    terms[scale] = used;
    if (!last) {
      int nh = 0, nw = 0;
      t0 = stamp();
      avg_pool2(cur_x, h, w, next, nh, nw);
      cur_x.swap(next);
      avg_pool2(cur_y, h, w, next, nh, nw);
      cur_y.swap(next);
      if (profile) add_us(stage.pool, t0, stamp());
      h = nh;
      w = nw;
    }
  }
  if (profile) {
    std::fprintf(stderr,
                 "msssim_profile scalar %dx%d threads %d convert_ms %.3f products_ms %.3f "
                 "filter_ms %.3f formula_ms %.3f pool_ms %.3f\n",
                 ref.width, ref.height, threads, stage.convert / 1000.0, stage.products / 1000.0,
                 stage.filter / 1000.0, stage.formula / 1000.0, stage.pool / 1000.0);
  }

  double score = 1.0;
  for (int i = 0; i < scales; ++i) score *= std::pow(terms[i], kMsssimWeight[i]);
  local.score = score;
  if (detail) *detail = local;
  return score;
}

double msssim_plane(PlaneView ref, PlaneView dist, double peak, int threads,
                         MsssimPlaneInfo* detail) {
  if (msssim_force_scalar() || !cpu_has_avx2()) {
    return msssim_plane_scalar(ref, dist, peak, threads, detail);
  }
  return msssim_plane_avx2(ref, dist, peak, threads, detail);
}

double msssim_plane_doubles(const double* ref, const double* dist, int width, int height, double peak,
                            int threads) {
  if (threads < 1) threads = 1;
  const int scales = msssim_scales(height, width);
  if (scales < 1) {
    throw std::runtime_error("plane too small for MS-SSIM (need at least 11x11)");
  }
  double kernel[kMsssimWin];
  gaussian_kernel(kernel);
  const double c1 = (0.01 * peak) * (0.01 * peak);
  const double c2 = (0.03 * peak) * (0.03 * peak);
  const size_t n0 = static_cast<size_t>(height) * width;
  std::vector<double> cur_x(ref, ref + n0);
  std::vector<double> cur_y(dist, dist + n0);
  int h = height;
  int w = width;
  double terms[5];
  std::vector<double> next;
  for (int scale = 0; scale < scales; ++scale) {
    std::vector<double> x2(cur_x.size()), y2(cur_y.size()), xy_src(cur_x.size());
    for (size_t i = 0; i < cur_x.size(); ++i) {
      x2[i] = cur_x[i] * cur_x[i];
      y2[i] = cur_y[i] * cur_y[i];
      xy_src[i] = cur_x[i] * cur_y[i];
    }
    std::vector<double> mux, muy, xx, yy, xy;
    filter_separable(cur_x, h, w, kernel, mux, threads);
    filter_separable(cur_y, h, w, kernel, muy, threads);
    filter_separable(x2, h, w, kernel, xx, threads);
    filter_separable(y2, h, w, kernel, yy, threads);
    filter_separable(xy_src, h, w, kernel, xy, threads);
    const int oh = h - (kMsssimWin - 1);
    const int ow = w - (kMsssimWin - 1);
    const double n = static_cast<double>(oh) * static_cast<double>(ow);
    double sum_cs = 0.0;
    double sum_ssim = 0.0;
    const size_t count = static_cast<size_t>(oh) * ow;
    for (size_t i = 0; i < count; ++i) {
      const double vx = xx[i] - mux[i] * mux[i];
      const double vy = yy[i] - muy[i] * muy[i];
      const double cov = xy[i] - mux[i] * muy[i];
      const double cs = (2.0 * cov + c2) / (vx + vy + c2);
      const double lum = (2.0 * mux[i] * muy[i] + c1) / (mux[i] * mux[i] + muy[i] * muy[i] + c1);
      sum_cs += cs;
      sum_ssim += lum * cs;
    }
    const bool last = scale + 1 == scales;
    terms[scale] = std::max(0.0, last ? sum_ssim / n : sum_cs / n);
    if (!last) {
      int nh = 0, nw = 0;
      avg_pool2(cur_x, h, w, next, nh, nw);
      cur_x.swap(next);
      avg_pool2(cur_y, h, w, next, nh, nw);
      cur_y.swap(next);
      h = nh;
      w = nw;
    }
  }
  double score = 1.0;
  for (int i = 0; i < scales; ++i) score *= std::pow(terms[i], kMsssimWeight[i]);
  return score;
}

PlaneScore compute_msssim_frame(const Frame& ref, const Frame& test, const VideoFormat& fmt,
                                     double peak, int threads) {
  if (threads < 1) threads = 1;
  PlaneScore out;
  out.y = msssim_plane(ref.y.view(), test.y.view(), peak, threads, nullptr);
  out.u = msssim_plane(ref.u.view(), test.u.view(), peak, threads, nullptr);
  out.v = msssim_plane(ref.v.view(), test.v.view(), peak, threads, nullptr);
  out.avg = yuv_sample_weight_avg(out.y, out.u, out.v, fmt.chroma);
  return out;
}

}  // namespace pixvqa
