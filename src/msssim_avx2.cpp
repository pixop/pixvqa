#include "pixvqa/msssim.hpp"
#include "pixvqa/thread_pool.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

void grow(std::vector<double>& v, size_t n) {
  if (v.size() < n) v.resize(n);
}

// Reused across frames in this process. MS-SSIM planes run one at a time.
struct Scratch {
  std::vector<double> x, y, ax, ay, x2, y2, xy;
  std::vector<double> tmp, mux, muy, xx, yy, xyf;
  void ensure_plane(int h, int w) {
    const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w);
    grow(x, n);
    grow(y, n);
    grow(ax, n);
    grow(ay, n);
    grow(x2, n);
    grow(y2, n);
    grow(xy, n);
  }
  void ensure_maps(int h, int w) {
    const int ow = w - (kMsssimWin - 1);
    const int oh = h - (kMsssimWin - 1);
    grow(tmp, static_cast<size_t>(h) * static_cast<size_t>(ow));
    const size_t no = static_cast<size_t>(oh) * static_cast<size_t>(ow);
    grow(mux, no);
    grow(muy, no);
    grow(xx, no);
    grow(yy, no);
    grow(xyf, no);
  }
};

Scratch& scratch() {
  static Scratch s;
  return s;
}

bool profile_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char* p = pix_getenv("MSSSIM_PROFILE");
    enabled = (p && p[0] == '1') ? 1 : 0;
  }
  return enabled != 0;
}

void convert_plane(PlaneView src, double* dst, int threads) {
  auto body = [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const uint16_t* row = src.data + static_cast<size_t>(y) * src.stride;
      double* out = dst + static_cast<size_t>(y) * src.width;
      for (int x = 0; x < src.width; ++x) out[x] = static_cast<double>(row[x]);
    }
  };
  if (threads <= 1 || src.height < 32) body(0, src.height);
  else parallel_for_ranges(threads, 0, src.height, body);
}

void products(const double* src_x, const double* src_y, int h, int w, double* x2, double* y2,
              double* xy, int threads) {
  auto body = [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const double* xr = src_x + static_cast<size_t>(y) * w;
      const double* yr = src_y + static_cast<size_t>(y) * w;
      double* a = x2 + static_cast<size_t>(y) * w;
      double* b = y2 + static_cast<size_t>(y) * w;
      double* c = xy + static_cast<size_t>(y) * w;
      for (int i = 0; i < w; ++i) {
        const double xv = xr[i];
        const double yv = yr[i];
        a[i] = xv * xv;
        b[i] = yv * yv;
        c[i] = xv * yv;
      }
    }
  };
  if (threads <= 1 || h < 32) body(0, h);
  else parallel_for_ranges(threads, 0, h, body);
}

void horiz_rows(const double* src, int w, int ow, int y0, int y1, const __m256d kv[kMsssimWin],
                const double k[kMsssimWin], double* tmp) {
  for (int y = y0; y < y1; ++y) {
    const double* row = src + static_cast<size_t>(y) * w;
    double* out = tmp + static_cast<size_t>(y) * ow;
    int x = 0;
    for (; x + 4 <= ow; x += 4) {
      __m256d acc = _mm256_setzero_pd();
      for (int t = 0; t < kMsssimWin; ++t) {
        const __m256d v = _mm256_loadu_pd(row + x + t);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(kv[t], v));
      }
      _mm256_storeu_pd(out + x, acc);
    }
    for (; x < ow; ++x) {
      double s = 0.0;
      for (int t = 0; t < kMsssimWin; ++t) s += k[t] * row[x + t];
      out[x] = s;
    }
  }
}

void vert_rows(const double* tmp, int ow, int y0, int y1, const __m256d kv[kMsssimWin],
               const double k[kMsssimWin], double* dst) {
  for (int y = y0; y < y1; ++y) {
    double* out = dst + static_cast<size_t>(y) * ow;
    int x = 0;
    for (; x + 4 <= ow; x += 4) {
      __m256d acc = _mm256_setzero_pd();
      for (int t = 0; t < kMsssimWin; ++t) {
        const __m256d v = _mm256_loadu_pd(tmp + (static_cast<size_t>(y + t) * ow + x));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(kv[t], v));
      }
      _mm256_storeu_pd(out + x, acc);
    }
    for (; x < ow; ++x) {
      double s = 0.0;
      for (int t = 0; t < kMsssimWin; ++t) {
        s += k[t] * tmp[static_cast<size_t>(y + t) * ow + x];
      }
      out[x] = s;
    }
  }
}

void filter_plane(const double* src, int h, int w, const __m256d kv[kMsssimWin], const double k[kMsssimWin],
                  double* tmp, double* dst, int threads) {
  const int ow = w - (kMsssimWin - 1);
  const int oh = h - (kMsssimWin - 1);
  auto horiz = [&](int y0, int y1) { horiz_rows(src, w, ow, y0, y1, kv, k, tmp); };
  if (threads <= 1 || h < 32) horiz(0, h);
  else parallel_for_ranges(threads, 0, h, horiz);
  auto vert = [&](int y0, int y1) { vert_rows(tmp, ow, y0, y1, kv, k, dst); };
  if (threads <= 1 || oh < 32) vert(0, oh);
  else parallel_for_ranges(threads, 0, oh, vert);
}

void pool_plane(const double* src, int h, int w, double* dst, int threads, int& oh, int& ow) {
  const int pad_h = h & 1;
  const int pad_w = w & 1;
  oh = pool_out(h);
  ow = pool_out(w);
  auto body = [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
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
  };
  if (threads <= 1 || oh < 32) body(0, oh);
  else parallel_for_ranges(threads, 0, oh, body);
}

}  // namespace

double msssim_plane_avx2(PlaneView ref, PlaneView dist, double peak, int threads,
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
  __m256d kv[kMsssimWin];
  for (int i = 0; i < kMsssimWin; ++i) kv[i] = _mm256_set1_pd(kernel[i]);
  const double c1 = (0.01 * peak) * (0.01 * peak);
  const double c2 = (0.03 * peak) * (0.03 * peak);
  const bool profile = profile_enabled();
  using Clock = std::chrono::steady_clock;
  double convert_us = 0, products_us = 0, filter_us = 0, formula_us = 0, pool_us = 0;
  auto stamp = []() { return Clock::now(); };
  auto add_us = [](double& slot, Clock::time_point a, Clock::time_point b) {
    slot += std::chrono::duration<double, std::micro>(b - a).count();
  };

  Scratch& sc = scratch();
  sc.ensure_plane(ref.height, ref.width);
  double* cur_x = sc.x.data();
  double* cur_y = sc.y.data();
  double* alt_x = sc.ax.data();
  double* alt_y = sc.ay.data();
  auto t0 = stamp();
  convert_plane(ref, cur_x, threads);
  convert_plane(dist, cur_y, threads);
  if (profile) add_us(convert_us, t0, stamp());

  int h = ref.height;
  int w = ref.width;
  double terms[5];
  MsssimPlaneInfo local;
  local.scales = scales;

  for (int scale = 0; scale < scales; ++scale) {
    sc.ensure_maps(h, w);
    t0 = stamp();
    products(cur_x, cur_y, h, w, sc.x2.data(), sc.y2.data(), sc.xy.data(), threads);
    if (profile) add_us(products_us, t0, stamp());
    t0 = stamp();
    filter_plane(cur_x, h, w, kv, kernel, sc.tmp.data(), sc.mux.data(), threads);
    filter_plane(cur_y, h, w, kv, kernel, sc.tmp.data(), sc.muy.data(), threads);
    filter_plane(sc.x2.data(), h, w, kv, kernel, sc.tmp.data(), sc.xx.data(), threads);
    filter_plane(sc.y2.data(), h, w, kv, kernel, sc.tmp.data(), sc.yy.data(), threads);
    filter_plane(sc.xy.data(), h, w, kv, kernel, sc.tmp.data(), sc.xyf.data(), threads);
    if (profile) add_us(filter_us, t0, stamp());

    const int oh = h - (kMsssimWin - 1);
    const int ow = w - (kMsssimWin - 1);
    const double n = static_cast<double>(oh) * static_cast<double>(ow);
    const size_t count = static_cast<size_t>(oh) * static_cast<size_t>(ow);
    const double* mux = sc.mux.data();
    const double* muy = sc.muy.data();
    const double* xx = sc.xx.data();
    const double* yy = sc.yy.data();
    const double* xyf = sc.xyf.data();
    double sum_cs = 0.0;
    double sum_ssim = 0.0;
    const __m256d c1v = _mm256_set1_pd(c1);
    const __m256d c2v = _mm256_set1_pd(c2);
    const __m256d two = _mm256_set1_pd(2.0);
    t0 = stamp();
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
      const __m256d muxv = _mm256_loadu_pd(mux + i);
      const __m256d muyv = _mm256_loadu_pd(muy + i);
      const __m256d xxv = _mm256_loadu_pd(xx + i);
      const __m256d yyv = _mm256_loadu_pd(yy + i);
      const __m256d xyv = _mm256_loadu_pd(xyf + i);
      const __m256d mux2 = _mm256_mul_pd(muxv, muxv);
      const __m256d muy2 = _mm256_mul_pd(muyv, muyv);
      const __m256d vx = _mm256_sub_pd(xxv, mux2);
      const __m256d vy = _mm256_sub_pd(yyv, muy2);
      const __m256d cov = _mm256_sub_pd(xyv, _mm256_mul_pd(muxv, muyv));
      const __m256d cs = _mm256_div_pd(_mm256_add_pd(_mm256_mul_pd(two, cov), c2v),
                                       _mm256_add_pd(_mm256_add_pd(vx, vy), c2v));
      const __m256d lum = _mm256_div_pd(
          _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(two, muxv), muyv), c1v),
          _mm256_add_pd(_mm256_add_pd(mux2, muy2), c1v));
      const __m256d ssimv = _mm256_mul_pd(lum, cs);
      alignas(32) double csl[4], ssl[4];
      _mm256_store_pd(csl, cs);
      _mm256_store_pd(ssl, ssimv);
      sum_cs += csl[0];
      sum_cs += csl[1];
      sum_cs += csl[2];
      sum_cs += csl[3];
      sum_ssim += ssl[0];
      sum_ssim += ssl[1];
      sum_ssim += ssl[2];
      sum_ssim += ssl[3];
    }
    for (; i < count; ++i) {
      const double vx = xx[i] - mux[i] * mux[i];
      const double vy = yy[i] - muy[i] * muy[i];
      const double cov = xyf[i] - mux[i] * muy[i];
      const double cs = (2.0 * cov + c2) / (vx + vy + c2);
      const double lum = (2.0 * mux[i] * muy[i] + c1) / (mux[i] * mux[i] + muy[i] * muy[i] + c1);
      sum_cs += cs;
      sum_ssim += lum * cs;
    }
    if (profile) add_us(formula_us, t0, stamp());
    const double cs = sum_cs / n;
    const double ssim = sum_ssim / n;
    local.scale[scale] = MsssimScaleInfo{h, w, cs, ssim};
    const bool last = scale + 1 == scales;
    terms[scale] = std::max(0.0, last ? ssim : cs);
    if (!last) {
      int nh = 0, nw = 0;
      t0 = stamp();
      pool_plane(cur_x, h, w, alt_x, threads, nh, nw);
      pool_plane(cur_y, h, w, alt_y, threads, nh, nw);
      if (profile) add_us(pool_us, t0, stamp());
      std::swap(cur_x, alt_x);
      std::swap(cur_y, alt_y);
      h = nh;
      w = nw;
    }
  }
  if (profile) {
    std::fprintf(stderr,
                 "msssim_profile avx2 %dx%d threads %d convert_ms %.3f products_ms %.3f "
                 "filter_ms %.3f formula_ms %.3f pool_ms %.3f\n",
                 ref.width, ref.height, threads, convert_us / 1000.0, products_us / 1000.0,
                 filter_us / 1000.0, formula_us / 1000.0, pool_us / 1000.0);
  }

  double score = 1.0;
  for (int i = 0; i < scales; ++i) score *= std::pow(terms[i], kMsssimWeight[i]);
  local.score = score;
  if (detail) *detail = local;
  return score;
}

}  // namespace pixvqa
