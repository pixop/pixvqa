#include "pixvqa/ssim.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace pixvqa {

namespace {

// Normalized 11-tap Gaussian, sigma=1.5 (matches scipy/numpy reference).
constexpr int kRadius = 5;
constexpr int kTaps = 11;

const double* kernel1d() {
  static double k[kTaps];
  static std::once_flag once;
  std::call_once(once, []() {
    double sum = 0.0;
    for (int i = 0; i < kTaps; ++i) {
      const double x = static_cast<double>(i - kRadius);
      k[i] = std::exp(-(x * x) / (2.0 * 1.5 * 1.5));
      sum += k[i];
    }
    for (int i = 0; i < kTaps; ++i) k[i] /= sum;
  });
  return k;
}

void blur_separable_valid(const double* src, int w, int h, int stride, double* dst_valid) {
  // dst_valid is (h-10) * (w-10), row-major.
  const double* k = kernel1d();
  const int ow = w - 2 * kRadius;
  const int oh = h - 2 * kRadius;
  std::vector<double> tmp(static_cast<size_t>(h) * w);

  // Horizontal, nearest border.
  for (int y = 0; y < h; ++y) {
    const double* row = src + static_cast<size_t>(y) * stride;
    double* out = tmp.data() + static_cast<size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      double acc = 0.0;
      for (int t = -kRadius; t <= kRadius; ++t) {
        int xx = x + t;
        if (xx < 0) xx = 0;
        if (xx >= w) xx = w - 1;
        acc += k[t + kRadius] * row[xx];
      }
      out[x] = acc;
    }
  }

  // Vertical + crop valid region.
  for (int y = kRadius; y < h - kRadius; ++y) {
    double* out = dst_valid + static_cast<size_t>(y - kRadius) * ow;
    for (int x = kRadius; x < w - kRadius; ++x) {
      double acc = 0.0;
      for (int t = -kRadius; t <= kRadius; ++t) {
        int yy = y + t;
        if (yy < 0) yy = 0;
        if (yy >= h) yy = h - 1;
        acc += k[t + kRadius] * tmp[static_cast<size_t>(yy) * w + x];
      }
      out[x - kRadius] = acc;
    }
  }
  (void)oh;
}

}  // namespace

double ssim_mean_scalar(PlaneView x, PlaneView y, double L, bool full, int /*threads*/) {
  if (x.width != y.width || x.height != y.height) {
    throw std::runtime_error("ssim shape mismatch");
  }
  const int w = x.width;
  const int h = x.height;
  if (w < kTaps || h < kTaps) {
    throw std::runtime_error("plane too small for 11x11 SSIM window");
  }
  const int ow = w - 2 * kRadius;
  const int oh = h - 2 * kRadius;
  const size_t n = static_cast<size_t>(ow) * oh;

  std::vector<double> xf(static_cast<size_t>(h) * w);
  std::vector<double> yf(static_cast<size_t>(h) * w);
  std::vector<double> xx(static_cast<size_t>(h) * w);
  std::vector<double> yy(static_cast<size_t>(h) * w);
  std::vector<double> xy(static_cast<size_t>(h) * w);
  for (int r = 0; r < h; ++r) {
    const uint16_t* px = x.data + static_cast<size_t>(r) * x.stride;
    const uint16_t* py = y.data + static_cast<size_t>(r) * y.stride;
    for (int c = 0; c < w; ++c) {
      const size_t i = static_cast<size_t>(r) * w + c;
      const double a = static_cast<double>(px[c]);
      const double b = static_cast<double>(py[c]);
      xf[i] = a;
      yf[i] = b;
      xx[i] = a * a;
      yy[i] = b * b;
      xy[i] = a * b;
    }
  }

  std::vector<double> mu1(n), mu2(n), s1(n), s2(n), s12(n);
  blur_separable_valid(xf.data(), w, h, w, mu1.data());
  blur_separable_valid(yf.data(), w, h, w, mu2.data());
  blur_separable_valid(xx.data(), w, h, w, s1.data());
  blur_separable_valid(yy.data(), w, h, w, s2.data());
  blur_separable_valid(xy.data(), w, h, w, s12.data());

  const double c1 = (0.01 * L) * (0.01 * L);
  const double c2 = (0.03 * L) * (0.03 * L);

  long double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double m1 = mu1[i];
    const double m2 = mu2[i];
    const double v1 = s1[i] - m1 * m1;
    const double v2 = s2[i] - m2 * m2;
    const double v12 = s12[i] - m1 * m2;
    const double cs = (2.0 * v12 + c2) / (v1 + v2 + c2);
    if (full) {
      const double lum = (2.0 * m1 * m2 + c1) / (m1 * m1 + m2 * m2 + c1);
      acc += lum * cs;
    } else {
      acc += cs;
    }
  }
  return static_cast<double>(acc / static_cast<long double>(n));
}

double ssim_mean_doubles(const double* x, int xstride, const double* y, int ystride, int w, int h,
                         double L, bool full) {
  if (w < kTaps || h < kTaps) {
    throw std::runtime_error("plane too small for 11x11 SSIM window");
  }
  const int ow = w - 2 * kRadius;
  const int oh = h - 2 * kRadius;
  const size_t n = static_cast<size_t>(ow) * oh;
  std::vector<double> xf(static_cast<size_t>(h) * w);
  std::vector<double> yf(static_cast<size_t>(h) * w);
  std::vector<double> xx(static_cast<size_t>(h) * w);
  std::vector<double> yy(static_cast<size_t>(h) * w);
  std::vector<double> xy(static_cast<size_t>(h) * w);
  for (int r = 0; r < h; ++r) {
    const double* px = x + static_cast<size_t>(r) * xstride;
    const double* py = y + static_cast<size_t>(r) * ystride;
    for (int c = 0; c < w; ++c) {
      const size_t i = static_cast<size_t>(r) * w + c;
      const double a = px[c];
      const double b = py[c];
      xf[i] = a;
      yf[i] = b;
      xx[i] = a * a;
      yy[i] = b * b;
      xy[i] = a * b;
    }
  }
  std::vector<double> mu1(n), mu2(n), s1(n), s2(n), s12(n);
  blur_separable_valid(xf.data(), w, h, w, mu1.data());
  blur_separable_valid(yf.data(), w, h, w, mu2.data());
  blur_separable_valid(xx.data(), w, h, w, s1.data());
  blur_separable_valid(yy.data(), w, h, w, s2.data());
  blur_separable_valid(xy.data(), w, h, w, s12.data());
  const double c1 = (0.01 * L) * (0.01 * L);
  const double c2 = (0.03 * L) * (0.03 * L);
  long double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double m1 = mu1[i];
    const double m2 = mu2[i];
    const double v1 = s1[i] - m1 * m1;
    const double v2 = s2[i] - m2 * m2;
    const double v12 = s12[i] - m1 * m2;
    const double cs = (2.0 * v12 + c2) / (v1 + v2 + c2);
    if (full) {
      const double lum = (2.0 * m1 * m2 + c1) / (m1 * m1 + m2 * m2 + c1);
      acc += lum * cs;
    } else {
      acc += cs;
    }
  }
  return static_cast<double>(acc / static_cast<long double>(n));
}

PlaneScore compute_ssim_frame(const Frame& ref, const Frame& test, const VideoFormat& fmt,
                              double L, SsimKernelMode mode, int threads) {
  PlaneScore out;
  if (threads < 1) threads = 1;
  out.y = ssim_mean(ref.y.view(), test.y.view(), L, /*full=*/true, mode, threads);
  out.u = ssim_mean(ref.u.view(), test.u.view(), L, /*full=*/true, mode, threads);
  out.v = ssim_mean(ref.v.view(), test.v.view(), L, /*full=*/true, mode, threads);
  out.avg = yuv_sample_weight_avg(out.y, out.u, out.v, fmt.chroma);
  return out;
}

}  // namespace pixvqa
