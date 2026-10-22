#include "pixvqa/ssim.hpp"
#include "pixvqa/thread_pool.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace pixvqa {

namespace {

constexpr int kRadius = 5;
constexpr int kTaps = 11;
constexpr int kDefaultStrip = 8;

alignas(32) static double g_kernel[kTaps];
static std::once_flag g_kern_once;
static int g_strip_height = kDefaultStrip;
static int g_strip_auto = 0;  // 1 = adaptive per plane/scale

bool slide_enabled();

int choose_strip(int oh, int threads) {
  if (!g_strip_auto) {
    int strip = g_strip_height;
    if (strip < 1) strip = kDefaultStrip;
    if (strip > oh) strip = oh;
    return strip;
  }
  // 48-frame sweep: 8 valid rows was the fastest independent-strip height at 8 threads.
  // With the sliding window, the same height also keeps worker spans balanced.
  int strip = 8;
  if (threads <= 1 && !slide_enabled()) strip = 128;
  if (strip > oh) strip = oh;
  if (strip < 1) strip = 1;
  return strip;
}

void init_kernel_once() {
  double sum = 0.0;
  for (int i = 0; i < kTaps; ++i) {
    const double x = static_cast<double>(i - kRadius);
    g_kernel[i] = std::exp(-(x * x) / (2.0 * 1.5 * 1.5));
    sum += g_kernel[i];
  }
  for (int i = 0; i < kTaps; ++i) g_kernel[i] /= sum;
}

inline void ensure_kernel() { std::call_once(g_kern_once, init_kernel_once); }

inline double hsum256d(__m256d v) {
  __m128d lo = _mm256_castpd256_pd128(v);
  __m128d hi = _mm256_extractf128_pd(v, 1);
  __m128d s = _mm_add_pd(lo, hi);
  __m128d sh = _mm_unpackhi_pd(s, s);
  s = _mm_add_sd(s, sh);
  return _mm_cvtsd_f64(s);
}

struct ProfileAccum {
  double convert_ms = 0;
  double blur_h_ms = 0;
  double blur_v_ms = 0;
  double map_ms = 0;
  double alloc_ms = 0;
  int calls = 0;
};

static std::mutex g_prof_mu;
static ProfileAccum g_prof_total;

bool profile_enabled() {
  const char* e = pix_getenv("PROFILE");
  return e && e[0] == '1';
}

void profile_add(const ProfileAccum& a) {
  if (!profile_enabled()) return;
  std::lock_guard<std::mutex> lock(g_prof_mu);
  g_prof_total.convert_ms += a.convert_ms;
  g_prof_total.blur_h_ms += a.blur_h_ms;
  g_prof_total.blur_v_ms += a.blur_v_ms;
  g_prof_total.map_ms += a.map_ms;
  g_prof_total.alloc_ms += a.alloc_ms;
  g_prof_total.calls += a.calls;
}

// CPU-time sums for the sliding-window kernel (nanoseconds, across workers).
static std::atomic<uint64_t> g_ns_convert{0};
static std::atomic<uint64_t> g_ns_h{0};
static std::atomic<uint64_t> g_ns_v{0};
static std::atomic<uint64_t> g_ns_map{0};

// Default on. PIXVQA_SLIDE=0 restores independent per-strip tasks (checkpoint A2).
bool slide_enabled() {
  const char* e = pix_getenv("SLIDE");
  if (!e || e[0] == '\0') return true;
  return e[0] != '0';
}

using Clock = std::chrono::steady_clock;

struct RawBuf {
  std::unique_ptr<double[]> data;
  size_t cap = 0;
  void ensure(size_t n) {
    if (n <= cap) return;
    data.reset(new double[n]);  // uninitialized on purpose
    cap = n;
  }
  double* get() { return data.get(); }
};

// ---- Horizontal blur (shared) ------------------------------------------------

void blur_h_row_avx2(const double* src, double* dst, int w) {
  ensure_kernel();
  const __m256d k0 = _mm256_set1_pd(g_kernel[0]);
  const __m256d k1 = _mm256_set1_pd(g_kernel[1]);
  const __m256d k2 = _mm256_set1_pd(g_kernel[2]);
  const __m256d k3 = _mm256_set1_pd(g_kernel[3]);
  const __m256d k4 = _mm256_set1_pd(g_kernel[4]);
  const __m256d k5 = _mm256_set1_pd(g_kernel[5]);
  const __m256d k6 = _mm256_set1_pd(g_kernel[6]);
  const __m256d k7 = _mm256_set1_pd(g_kernel[7]);
  const __m256d k8 = _mm256_set1_pd(g_kernel[8]);
  const __m256d k9 = _mm256_set1_pd(g_kernel[9]);
  const __m256d k10 = _mm256_set1_pd(g_kernel[10]);

  int x = 0;
  for (; x < kRadius && x < w; ++x) {
    double acc = 0.0;
    for (int t = -kRadius; t <= kRadius; ++t) {
      int xx = x + t;
      if (xx < 0) xx = 0;
      if (xx >= w) xx = w - 1;
      acc += g_kernel[t + kRadius] * src[xx];
    }
    dst[x] = acc;
  }

  const int x_interior_end = w - kRadius;
  for (; x + 4 <= x_interior_end; x += 4) {
    __m256d acc = _mm256_setzero_pd();
    acc = _mm256_fmadd_pd(k0, _mm256_loadu_pd(src + x - 5), acc);
    acc = _mm256_fmadd_pd(k1, _mm256_loadu_pd(src + x - 4), acc);
    acc = _mm256_fmadd_pd(k2, _mm256_loadu_pd(src + x - 3), acc);
    acc = _mm256_fmadd_pd(k3, _mm256_loadu_pd(src + x - 2), acc);
    acc = _mm256_fmadd_pd(k4, _mm256_loadu_pd(src + x - 1), acc);
    acc = _mm256_fmadd_pd(k5, _mm256_loadu_pd(src + x), acc);
    acc = _mm256_fmadd_pd(k6, _mm256_loadu_pd(src + x + 1), acc);
    acc = _mm256_fmadd_pd(k7, _mm256_loadu_pd(src + x + 2), acc);
    acc = _mm256_fmadd_pd(k8, _mm256_loadu_pd(src + x + 3), acc);
    acc = _mm256_fmadd_pd(k9, _mm256_loadu_pd(src + x + 4), acc);
    acc = _mm256_fmadd_pd(k10, _mm256_loadu_pd(src + x + 5), acc);
    _mm256_storeu_pd(dst + x, acc);
  }
  for (; x < x_interior_end; ++x) {
    double acc = 0.0;
    for (int t = -kRadius; t <= kRadius; ++t) acc += g_kernel[t + kRadius] * src[x + t];
    dst[x] = acc;
  }
  for (; x < w; ++x) {
    double acc = 0.0;
    for (int t = -kRadius; t <= kRadius; ++t) {
      int xx = x + t;
      if (xx < 0) xx = 0;
      if (xx >= w) xx = w - 1;
      acc += g_kernel[t + kRadius] * src[xx];
    }
    dst[x] = acc;
  }
}

void convert_row(const uint16_t* px, const uint16_t* py, int w, double* xf, double* yf, double* xx,
                 double* yy, double* xy) {
  int c = 0;
  for (; c + 4 <= w; c += 4) {
    for (int k = 0; k < 4; ++k) {
      const double a = static_cast<double>(px[c + k]);
      const double b = static_cast<double>(py[c + k]);
      xf[c + k] = a;
      yf[c + k] = b;
      xx[c + k] = a * a;
      yy[c + k] = b * b;
      xy[c + k] = a * b;
    }
  }
  for (; c < w; ++c) {
    const double a = static_cast<double>(px[c]);
    const double b = static_cast<double>(py[c]);
    xf[c] = a;
    yf[c] = b;
    xx[c] = a * a;
    yy[c] = b * b;
    xy[c] = a * b;
  }
}

// Vertical blur one valid row. rows[0] is center-5 ... rows[10] is center+5, each length w.
void blur_v_from_rows(const double* rows[11], int w, double* dst_valid) {
  ensure_kernel();
  const __m256d k0 = _mm256_set1_pd(g_kernel[0]);
  const __m256d k1 = _mm256_set1_pd(g_kernel[1]);
  const __m256d k2 = _mm256_set1_pd(g_kernel[2]);
  const __m256d k3 = _mm256_set1_pd(g_kernel[3]);
  const __m256d k4 = _mm256_set1_pd(g_kernel[4]);
  const __m256d k5 = _mm256_set1_pd(g_kernel[5]);
  const __m256d k6 = _mm256_set1_pd(g_kernel[6]);
  const __m256d k7 = _mm256_set1_pd(g_kernel[7]);
  const __m256d k8 = _mm256_set1_pd(g_kernel[8]);
  const __m256d k9 = _mm256_set1_pd(g_kernel[9]);
  const __m256d k10 = _mm256_set1_pd(g_kernel[10]);

  const double* r0 = rows[0];
  const double* r1 = rows[1];
  const double* r2 = rows[2];
  const double* r3 = rows[3];
  const double* r4 = rows[4];
  const double* r5 = rows[5];
  const double* r6 = rows[6];
  const double* r7 = rows[7];
  const double* r8 = rows[8];
  const double* r9 = rows[9];
  const double* r10 = rows[10];

  int x = kRadius;
  for (; x + 4 <= w - kRadius; x += 4) {
    __m256d acc = _mm256_setzero_pd();
    acc = _mm256_fmadd_pd(k0, _mm256_loadu_pd(r0 + x), acc);
    acc = _mm256_fmadd_pd(k1, _mm256_loadu_pd(r1 + x), acc);
    acc = _mm256_fmadd_pd(k2, _mm256_loadu_pd(r2 + x), acc);
    acc = _mm256_fmadd_pd(k3, _mm256_loadu_pd(r3 + x), acc);
    acc = _mm256_fmadd_pd(k4, _mm256_loadu_pd(r4 + x), acc);
    acc = _mm256_fmadd_pd(k5, _mm256_loadu_pd(r5 + x), acc);
    acc = _mm256_fmadd_pd(k6, _mm256_loadu_pd(r6 + x), acc);
    acc = _mm256_fmadd_pd(k7, _mm256_loadu_pd(r7 + x), acc);
    acc = _mm256_fmadd_pd(k8, _mm256_loadu_pd(r8 + x), acc);
    acc = _mm256_fmadd_pd(k9, _mm256_loadu_pd(r9 + x), acc);
    acc = _mm256_fmadd_pd(k10, _mm256_loadu_pd(r10 + x), acc);
    _mm256_storeu_pd(dst_valid + (x - kRadius), acc);
  }
  for (; x < w - kRadius; ++x) {
    double acc = 0.0;
    for (int t = 0; t < kTaps; ++t) acc += g_kernel[t] * rows[t][x];
    dst_valid[x - kRadius] = acc;
  }
}

// Vertical blur one valid row into dst[ow], reading H-blurred plane with stride w.
// center_y is absolute source row in [radius, h-radius).
void blur_v_one_row(const double* hplane, int w, int center_y, double* dst_valid) {
  const double* rows[11];
  for (int t = 0; t < kTaps; ++t) {
    rows[t] = hplane + static_cast<size_t>(center_y - kRadius + t) * w;
  }
  blur_v_from_rows(rows, w, dst_valid);
}

double map_row_sum(const double* mu1, const double* mu2, const double* s1, const double* s2,
                   const double* s12, int ow, double c1, double c2, bool full) {
  const __m256d vc1 = _mm256_set1_pd(c1);
  const __m256d vc2 = _mm256_set1_pd(c2);
  const __m256d vtwo = _mm256_set1_pd(2.0);
  __m256d vacc = _mm256_setzero_pd();
  double tail = 0.0;
  int i = 0;
  for (; i + 4 <= ow; i += 4) {
    __m256d m1 = _mm256_loadu_pd(mu1 + i);
    __m256d m2 = _mm256_loadu_pd(mu2 + i);
    __m256d v1 = _mm256_sub_pd(_mm256_loadu_pd(s1 + i), _mm256_mul_pd(m1, m1));
    __m256d v2 = _mm256_sub_pd(_mm256_loadu_pd(s2 + i), _mm256_mul_pd(m2, m2));
    __m256d v12 = _mm256_sub_pd(_mm256_loadu_pd(s12 + i), _mm256_mul_pd(m1, m2));
    __m256d cs = _mm256_div_pd(_mm256_fmadd_pd(vtwo, v12, vc2),
                               _mm256_add_pd(_mm256_add_pd(v1, v2), vc2));
    __m256d score = cs;
    if (full) {
      __m256d lum =
          _mm256_div_pd(_mm256_fmadd_pd(vtwo, _mm256_mul_pd(m1, m2), vc1),
                        _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(m1, m1), _mm256_mul_pd(m2, m2)),
                                      vc1));
      score = _mm256_mul_pd(lum, cs);
    }
    vacc = _mm256_add_pd(vacc, score);
  }
  for (; i < ow; ++i) {
    const double m1 = mu1[i];
    const double m2 = mu2[i];
    const double v1 = s1[i] - m1 * m1;
    const double v2 = s2[i] - m2 * m2;
    const double v12 = s12[i] - m1 * m2;
    const double cs = (2.0 * v12 + c2) / (v1 + v2 + c2);
    if (full) {
      const double lum = (2.0 * m1 * m2 + c1) / (m1 * m1 + m2 * m2 + c1);
      tail += lum * cs;
    } else {
      tail += cs;
    }
  }
  return hsum256d(vacc) + tail;
}

// ---- Phase-1 full-frame workspace -------------------------------------------

struct FullWorkspace {
  RawBuf xf, yf, xx, yy, xy;
  RawBuf h1, h2, h3, h4, h5;
  RawBuf mu1, mu2, s1, s2, s12;
  int cap_w = 0, cap_h = 0;
  void ensure(int w, int h) {
    if (w <= cap_w && h <= cap_h) return;
    const size_t plane = static_cast<size_t>(w) * h;
    const size_t valid = static_cast<size_t>(std::max(0, w - 2 * kRadius)) *
                         static_cast<size_t>(std::max(0, h - 2 * kRadius));
    xf.ensure(plane);
    yf.ensure(plane);
    xx.ensure(plane);
    yy.ensure(plane);
    xy.ensure(plane);
    h1.ensure(plane);
    h2.ensure(plane);
    h3.ensure(plane);
    h4.ensure(plane);
    h5.ensure(plane);
    mu1.ensure(valid);
    mu2.ensure(valid);
    s1.ensure(valid);
    s2.ensure(valid);
    s12.ensure(valid);
    cap_w = w;
    cap_h = h;
  }
};

FullWorkspace& tls_full_ws() {
  thread_local FullWorkspace ws;
  return ws;
}

void blur_v_rows_avx2(const double* tmp, int w, int h, int y0, int y1, double* out_valid) {
  const int y_lo = std::max(y0, kRadius);
  const int y_hi = std::min(y1, h - kRadius);
  const int ow = w - 2 * kRadius;
  for (int y = y_lo; y < y_hi; ++y) {
    blur_v_one_row(tmp, w, y, out_valid + static_cast<size_t>(y - kRadius) * ow);
  }
}

// ---- Phase-2 strip workspace ------------------------------------------------

struct StripWorkspace {
  RawBuf src[5];  // xf,yf,xx,yy,xy — nrows * w
  RawBuf hbuf[5];
  RawBuf mu1, mu2, s1, s2, s12;  // one row ow
  int cap_w = 0;
  int cap_nrows = 0;
  int cap_ow = 0;

  void ensure(int w, int nrows, int ow) {
    if (w <= cap_w && nrows <= cap_nrows && ow <= cap_ow) return;
    const size_t plane = static_cast<size_t>(std::max(w, 1)) * static_cast<size_t>(std::max(nrows, 1));
    for (int c = 0; c < 5; ++c) {
      src[c].ensure(plane);
      hbuf[c].ensure(plane);
    }
    mu1.ensure(static_cast<size_t>(std::max(ow, 1)));
    mu2.ensure(static_cast<size_t>(std::max(ow, 1)));
    s1.ensure(static_cast<size_t>(std::max(ow, 1)));
    s2.ensure(static_cast<size_t>(std::max(ow, 1)));
    s12.ensure(static_cast<size_t>(std::max(ow, 1)));
    cap_w = w;
    cap_nrows = nrows;
    cap_ow = ow;
  }
};

StripWorkspace& tls_strip_ws() {
  thread_local StripWorkspace ws;
  return ws;
}

// Process valid rows [v0, v1). Returns sum of SSIM/CS over those rows (not mean).
double process_strip(PlaneView x, PlaneView y, int v0, int v1, double c1, double c2, bool full) {
  const int w = x.width;
  const int h = x.height;
  const int ow = w - 2 * kRadius;
  // Source rows needed for H: [v0, v1 + 2*radius)
  const int src0 = v0;
  const int src1 = v1 + 2 * kRadius;
  const int nrows = src1 - src0;
  if (nrows <= 0 || v0 >= v1) return 0.0;

  StripWorkspace& ws = tls_strip_ws();
  ws.ensure(w, nrows, ow);

  double* s[5] = {ws.src[0].get(), ws.src[1].get(), ws.src[2].get(), ws.src[3].get(),
                  ws.src[4].get()};
  double* hb[5] = {ws.hbuf[0].get(), ws.hbuf[1].get(), ws.hbuf[2].get(), ws.hbuf[3].get(),
                   ws.hbuf[4].get()};

  // Convert + H for all five channels together (cache-friendly per row).
  for (int ly = 0; ly < nrows; ++ly) {
    const int sy = src0 + ly;
    // Valid SSIM centers only use sy in [0,h); strip construction guarantees that.
    const uint16_t* px = x.data + static_cast<size_t>(sy) * x.stride;
    const uint16_t* py = y.data + static_cast<size_t>(sy) * y.stride;
    const size_t off = static_cast<size_t>(ly) * w;
    convert_row(px, py, w, s[0] + off, s[1] + off, s[2] + off, s[3] + off, s[4] + off);
    for (int c = 0; c < 5; ++c) {
      blur_h_row_avx2(s[c] + off, hb[c] + off, w);
    }
  }

  // Vertical + map. Local row index for absolute source y is (y - src0).
  double sum = 0.0;
  for (int v = v0; v < v1; ++v) {
    const int local_cy = (v + kRadius) - src0;  // in [radius, nrows-radius)
    blur_v_one_row(hb[0], w, local_cy, ws.mu1.get());
    blur_v_one_row(hb[1], w, local_cy, ws.mu2.get());
    blur_v_one_row(hb[2], w, local_cy, ws.s1.get());
    blur_v_one_row(hb[3], w, local_cy, ws.s2.get());
    blur_v_one_row(hb[4], w, local_cy, ws.s12.get());
    sum += map_row_sum(ws.mu1.get(), ws.mu2.get(), ws.s1.get(), ws.s2.get(), ws.s12.get(), ow, c1,
                       c2, full);
  }
  (void)h;
  return sum;
}

// 11-row ring of horizontally filtered moments. A worker walks a contiguous
// span of strips and reuses the 10-row overlap instead of recomputing it.
struct SlideWorkspace {
  RawBuf ring[5];
  RawBuf row[5];
  RawBuf mu[5];
  int cap_w = 0;
  int cap_ow = 0;

  void ensure(int w, int ow) {
    if (w <= cap_w && ow <= cap_ow) return;
    const size_t ring_n = static_cast<size_t>(std::max(w, 1)) * static_cast<size_t>(kTaps);
    for (int c = 0; c < 5; ++c) {
      ring[c].ensure(ring_n);
      row[c].ensure(static_cast<size_t>(std::max(w, 1)));
      mu[c].ensure(static_cast<size_t>(std::max(ow, 1)));
    }
    cap_w = w;
    cap_ow = ow;
  }
};

SlideWorkspace& tls_slide_ws() {
  thread_local SlideWorkspace ws;
  return ws;
}

inline uint64_t ns_between(Clock::time_point a, Clock::time_point b) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

// Fill partial[s0, s1) with the same per-strip sums as process_strip.
// Horizontal rows inside the span are filtered once.
void process_span(PlaneView x, PlaneView y, int s0, int s1, int strip, int oh, double c1,
                  double c2, bool full, double* partial) {
  if (s0 >= s1) return;
  const int w = x.width;
  const int ow = w - 2 * kRadius;
  const int v0 = s0 * strip;
  const int v1 = std::min(oh, s1 * strip);
  if (v0 >= v1) return;

  SlideWorkspace& ws = tls_slide_ws();
  ws.ensure(w, ow);
  double* ring[5];
  double* rowb[5];
  double* mu[5];
  for (int c = 0; c < 5; ++c) {
    ring[c] = ws.ring[c].get();
    rowb[c] = ws.row[c].get();
    mu[c] = ws.mu[c].get();
  }

  const bool prof = profile_enabled();
  uint64_t ns_c = 0, ns_h = 0, ns_v = 0, ns_m = 0;
  double strip_sum = 0.0;
  int cur_s = s0;
  int boundary = std::min(v1, (cur_s + 1) * strip);

  const int sy_end = v1 + 2 * kRadius;
  for (int sy = v0; sy < sy_end; ++sy) {
    Clock::time_point t0{}, t1{}, t2{};
    if (prof) t0 = Clock::now();
    const uint16_t* px = x.data + static_cast<size_t>(sy) * x.stride;
    const uint16_t* py = y.data + static_cast<size_t>(sy) * y.stride;
    convert_row(px, py, w, rowb[0], rowb[1], rowb[2], rowb[3], rowb[4]);
    if (prof) t1 = Clock::now();
    const int slot = sy % kTaps;
    for (int c = 0; c < 5; ++c) {
      blur_h_row_avx2(rowb[c], ring[c] + static_cast<size_t>(slot) * w, w);
    }
    if (prof) {
      t2 = Clock::now();
      ns_c += ns_between(t0, t1);
      ns_h += ns_between(t1, t2);
    }
    if (sy < v0 + 2 * kRadius) continue;
    const int v = sy - 2 * kRadius;
    if (v >= v1) break;

    Clock::time_point tv0{}, tv1{}, tm1{};
    if (prof) tv0 = Clock::now();
    for (int c = 0; c < 5; ++c) {
      const double* taps[11];
      for (int t = 0; t < kTaps; ++t) {
        const int src_row = v + t;
        taps[t] = ring[c] + static_cast<size_t>(src_row % kTaps) * w;
      }
      blur_v_from_rows(taps, w, mu[c]);
    }
    if (prof) tv1 = Clock::now();
    strip_sum += map_row_sum(mu[0], mu[1], mu[2], mu[3], mu[4], ow, c1, c2, full);
    if (prof) {
      tm1 = Clock::now();
      ns_v += ns_between(tv0, tv1);
      ns_m += ns_between(tv1, tm1);
    }
    if (v + 1 == boundary) {
      partial[cur_s] = strip_sum;
      strip_sum = 0.0;
      ++cur_s;
      boundary = std::min(v1, (cur_s + 1) * strip);
    }
  }
  if (cur_s != s1) throw std::runtime_error("sliding span did not cover every strip");
  if (prof) {
    g_ns_convert.fetch_add(ns_c, std::memory_order_relaxed);
    g_ns_h.fetch_add(ns_h, std::memory_order_relaxed);
    g_ns_v.fetch_add(ns_v, std::memory_order_relaxed);
    g_ns_map.fetch_add(ns_m, std::memory_order_relaxed);
  }
}

}  // namespace

void pixvqa_set_strip_height(int rows) {
  if (rows < 0) {
    g_strip_auto = 1;
    g_strip_height = kDefaultStrip;
    return;
  }
  g_strip_auto = 0;
  if (rows == 0) g_strip_height = kDefaultStrip;
  else g_strip_height = rows;
}

int pixvqa_get_strip_height() { return g_strip_auto ? -1 : g_strip_height; }

void pixvqa_dump_ssim_profile() {
  if (!profile_enabled()) return;
  std::lock_guard<std::mutex> lock(g_prof_mu);
  const double tot = g_prof_total.convert_ms + g_prof_total.blur_h_ms + g_prof_total.blur_v_ms +
                     g_prof_total.map_ms + g_prof_total.alloc_ms;
  const double sc = static_cast<double>(g_ns_convert.load()) / 1e6;
  const double sh = static_cast<double>(g_ns_h.load()) / 1e6;
  const double sv = static_cast<double>(g_ns_v.load()) / 1e6;
  const double sm = static_cast<double>(g_ns_map.load()) / 1e6;
  std::fprintf(stderr,
               "[PIXVQA_PROFILE] ssim_calls=%d alloc=%.1fms convert=%.1fms blur_h=%.1fms "
               "blur_v=%.1fms map=%.1fms total=%.1fms strip=%d slide=%d "
               "slide_convert=%.1fms slide_h=%.1fms slide_v=%.1fms slide_map=%.1fms\n",
               g_prof_total.calls, g_prof_total.alloc_ms, g_prof_total.convert_ms,
               g_prof_total.blur_h_ms, g_prof_total.blur_v_ms, g_prof_total.map_ms, tot,
               g_strip_height, slide_enabled() ? 1 : 0, sc, sh, sv, sm);
}

double ssim_mean_avx2_fullframe(PlaneView x, PlaneView y, double L, bool full, int threads) {
  if (x.width != y.width || x.height != y.height) throw std::runtime_error("ssim shape mismatch");
  const int w = x.width;
  const int h = x.height;
  if (w < kTaps || h < kTaps) throw std::runtime_error("plane too small for 11x11 SSIM window");
  if (threads < 1) threads = 1;

  const int ow = w - 2 * kRadius;
  const int oh = h - 2 * kRadius;
  const size_t n = static_cast<size_t>(ow) * oh;

  ProfileAccum local_prof{};
  local_prof.calls = 1;
  auto ta = Clock::now();
  FullWorkspace& ws = tls_full_ws();
  ws.ensure(w, h);
  auto tb = Clock::now();
  local_prof.alloc_ms = std::chrono::duration<double, std::milli>(tb - ta).count();

  parallel_for_ranges(threads, 0, h, [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      convert_row(x.data + static_cast<size_t>(r) * x.stride, y.data + static_cast<size_t>(r) * y.stride,
                  w, ws.xf.get() + static_cast<size_t>(r) * w, ws.yf.get() + static_cast<size_t>(r) * w,
                  ws.xx.get() + static_cast<size_t>(r) * w, ws.yy.get() + static_cast<size_t>(r) * w,
                  ws.xy.get() + static_cast<size_t>(r) * w);
    }
  });
  auto tc = Clock::now();
  local_prof.convert_ms = std::chrono::duration<double, std::milli>(tc - tb).count();

  double* hbufs[5] = {ws.h1.get(), ws.h2.get(), ws.h3.get(), ws.h4.get(), ws.h5.get()};
  const double* srcs[5] = {ws.xf.get(), ws.yf.get(), ws.xx.get(), ws.yy.get(), ws.xy.get()};
  double* dsts[5] = {ws.mu1.get(), ws.mu2.get(), ws.s1.get(), ws.s2.get(), ws.s12.get()};

  auto t_h0 = Clock::now();
  parallel_for_ranges(threads, 0, h, [&](int y0, int y1) {
    for (int yy = y0; yy < y1; ++yy) {
      for (int c = 0; c < 5; ++c) {
        blur_h_row_avx2(srcs[c] + static_cast<size_t>(yy) * w, hbufs[c] + static_cast<size_t>(yy) * w,
                        w);
      }
    }
  });
  auto t_h1 = Clock::now();
  parallel_for_ranges(threads, kRadius, h - kRadius, [&](int y0, int y1) {
    for (int c = 0; c < 5; ++c) blur_v_rows_avx2(hbufs[c], w, h, y0, y1, dsts[c]);
  });
  auto t_v1 = Clock::now();
  local_prof.blur_h_ms += std::chrono::duration<double, std::milli>(t_h1 - t_h0).count();
  local_prof.blur_v_ms += std::chrono::duration<double, std::milli>(t_v1 - t_h1).count();

  const double c1 = (0.01 * L) * (0.01 * L);
  const double c2 = (0.03 * L) * (0.03 * L);
  auto tm0 = Clock::now();
  double sum = 0.0;
  for (int v = 0; v < oh; ++v) {
    sum += map_row_sum(ws.mu1.get() + static_cast<size_t>(v) * ow,
                       ws.mu2.get() + static_cast<size_t>(v) * ow,
                       ws.s1.get() + static_cast<size_t>(v) * ow,
                       ws.s2.get() + static_cast<size_t>(v) * ow,
                       ws.s12.get() + static_cast<size_t>(v) * ow, ow, c1, c2, full);
  }
  auto tm1 = Clock::now();
  local_prof.map_ms = std::chrono::duration<double, std::milli>(tm1 - tm0).count();
  profile_add(local_prof);
  return sum / static_cast<double>(n);
}

double ssim_mean_avx2_tiled(PlaneView x, PlaneView y, double L, bool full, int threads) {
  if (x.width != y.width || x.height != y.height) throw std::runtime_error("ssim shape mismatch");
  const int w = x.width;
  const int h = x.height;
  if (w < kTaps || h < kTaps) throw std::runtime_error("plane too small for 11x11 SSIM window");
  if (threads < 1) threads = 1;

  const int ow = w - 2 * kRadius;
  const int oh = h - 2 * kRadius;
  const size_t n = static_cast<size_t>(ow) * oh;
  const double c1 = (0.01 * L) * (0.01 * L);
  const double c2 = (0.03 * L) * (0.03 * L);

  int strip = choose_strip(oh, threads);

  ProfileAccum local_prof{};
  local_prof.calls = 1;
  auto t0 = Clock::now();

  const int nstrips = (oh + strip - 1) / strip;
  std::vector<double> partial(static_cast<size_t>(nstrips), 0.0);

  if (slide_enabled()) {
    // One contiguous span per worker. Halo rows are filtered once inside the span.
    // Per-strip partials keep the same reduction order as independent strips.
    if (threads <= 1 || nstrips <= 1) {
      process_span(x, y, 0, nstrips, strip, oh, c1, c2, full, partial.data());
    } else {
      const int nworkers = std::min(threads, nstrips);
      PersistentFor::instance().run_indexed(nworkers, nworkers, [&](int w) {
        const int s0 = static_cast<int>((static_cast<long long>(nstrips) * w) / nworkers);
        const int s1 =
            static_cast<int>((static_cast<long long>(nstrips) * (w + 1)) / nworkers);
        process_span(x, y, s0, s1, strip, oh, c1, c2, full, partial.data());
      });
    }
  } else {
    // Checkpoint A2: one independent task per strip (halo recomputed on every strip).
    PersistentFor::instance().run_indexed(threads, nstrips, [&](int s) {
      const int v0 = s * strip;
      const int v1 = std::min(oh, v0 + strip);
      partial[static_cast<size_t>(s)] = process_strip(x, y, v0, v1, c1, c2, full);
    });
  }

  // Deterministic reduction in strip order (matches top-to-bottom row order).
  double sum = 0.0;
  for (double p : partial) sum += p;

  auto t1 = Clock::now();
  local_prof.blur_h_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  profile_add(local_prof);
  return sum / static_cast<double>(n);
}

double ssim_mean_avx2(PlaneView x, PlaneView y, double L, bool full, int threads) {
  return ssim_mean_avx2_tiled(x, y, L, full, threads);
}

}  // namespace pixvqa
