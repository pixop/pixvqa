// CUDA PU21. The integer PSNR, SSIM, and MS-SSIM kernels stay on sample codes.
// This path encodes display light on the device, then applies the same double
// formulas as ssim_mean_doubles and msssim_plane_doubles.
//
// SSIM uses the clamp-to-edge 11-tap Gaussian. MS-SSIM uses a valid 11-tap
// Gaussian and the odd-size 2x2 pool. The peak is 256. RGB is not downloaded.

#include "pixvqa/cuda_backend.hpp"
#include "pixvqa/msssim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pixvqa {
namespace {

constexpr int kRadius = 5;
constexpr int kTaps = 11;
constexpr int kThreads = 256;
constexpr double kLMin = 0.005;
constexpr double kLMax = 10000.0;
constexpr double kPuPeak = 256.0;

#define PU21_CHECK(expr)                                                                    \
  do {                                                                                      \
    cudaError_t _e = (expr);                                                                \
    if (_e != cudaSuccess) {                                                                \
      throw std::runtime_error(std::string("CUDA: ") + #expr + ": " + cudaGetErrorString(_e)); \
    }                                                                                       \
  } while (0)

__constant__ double pu21_k64[kTaps];

struct Pu21GpuHdr {
  int transfer = 0;  // 1 PQ, 2 HLG
  int full_range = 0;
  int bit_depth = 10;
  int chroma = 0;  // 0 420, 1 422, 2 444
  int loc = 0;     // 0 left, 1 center, 2 topleft
  int nearest = 0;
  int pq_clip = 0;
  double peak = 0;
  double black = 0;
  double gamma = 0;
  double kr = 0;
  double kg = 0;
  double kb = 0;
};

struct Pu21Ctx {
  bool ready = false;
  cudaStream_t stream = nullptr;
  uint16_t* pin = nullptr;
  size_t pin_samples = 0;
  uint16_t* y[2] = {};
  uint16_t* u[2] = {};
  uint16_t* v[2] = {};
  size_t y_cap[2] = {};
  size_t u_cap[2] = {};
  size_t v_cap[2] = {};
  double* luma[2] = {};
  size_t luma_cap[2] = {};
  double* work = nullptr;
  size_t work_cap = 0;
  double* pyr = nullptr;
  size_t pyr_cap = 0;
  double* partials = nullptr;
  int partial_cap = 0;
  int* clamped = nullptr;
  double* acc = nullptr;
};

Pu21Ctx& ctx() {
  static Pu21Ctx c;
  return c;
}

void upload_kernel() {
  double k[kTaps];
  double sum = 0.0;
  for (int i = 0; i < kTaps; ++i) {
    const double x = static_cast<double>(i - kRadius);
    k[i] = std::exp(-(x * x) / (2.0 * 1.5 * 1.5));
    sum += k[i];
  }
  for (int i = 0; i < kTaps; ++i) k[i] /= sum;
  PU21_CHECK(cudaMemcpyToSymbol(pu21_k64, k, sizeof(k)));
}

void init_once() {
  auto& c = ctx();
  if (c.ready) return;
  PU21_CHECK(cudaFree(nullptr));
  PU21_CHECK(cudaStreamCreate(&c.stream));
  PU21_CHECK(cudaMalloc(&c.clamped, sizeof(int)));
  PU21_CHECK(cudaMalloc(&c.acc, 12 * sizeof(double)));
  upload_kernel();
  c.ready = true;
}

void grow_u16(uint16_t*& ptr, size_t& cap, size_t need) {
  if (cap >= need) return;
  if (ptr) PU21_CHECK(cudaFree(ptr));
  ptr = nullptr;
  cap = 0;
  PU21_CHECK(cudaMalloc(&ptr, need * sizeof(uint16_t)));
  cap = need;
}

void grow_f64(double*& ptr, size_t& cap, size_t need) {
  if (cap >= need) return;
  if (ptr) PU21_CHECK(cudaFree(ptr));
  ptr = nullptr;
  cap = 0;
  PU21_CHECK(cudaMalloc(&ptr, need * sizeof(double)));
  cap = need;
}

__device__ double block_sum_256(double v) {
  __shared__ double sm[kThreads];
  const int tx = threadIdx.x;
  sm[tx] = v;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tx < stride) sm[tx] += sm[tx + stride];
    __syncthreads();
  }
  return sm[0];
}

__device__ double dev_pow(double a, double b) { return pow(a, b); }

__device__ double pq_eotf(double e) {
  if (e <= 0.0) return 0.0;
  if (e > 1.0) e = 1.0;
  constexpr double m1 = 2610.0 / 16384.0;
  constexpr double m2 = 2523.0 / 32.0;
  constexpr double c1 = 3424.0 / 4096.0;
  constexpr double c2 = 2413.0 / 128.0;
  constexpr double c3 = 2392.0 / 128.0;
  const double np = dev_pow(e, 1.0 / m2);
  const double num = np > c1 ? np - c1 : 0.0;
  const double den = c2 - c3 * np;
  if (!(den > 0.0)) return 0.0;
  return 10000.0 * dev_pow(num / den, 1.0 / m1);
}

__device__ double hlg_inverse_oetf(double e) {
  if (e <= 0.0) return 0.0;
  if (e > 1.0) e = 1.0;
  constexpr double a = 0.17883277;
  const double b = 1.0 - 4.0 * a;
  const double c = 1.0 - a * log(12.0 - b);
  if (e <= 0.5) return (e * e) / 3.0;
  return (exp((e - c) / a) + b) / 12.0;
}

__device__ double pu21_encode_d(double y, int* clamped) {
  if (y < kLMin || y > kLMax) *clamped = 1;
  if (y < kLMin) y = kLMin;
  if (y > kLMax) y = kLMax;
  constexpr double p1 = 0.353487901;
  constexpr double p2 = 0.3734658629;
  constexpr double p3 = 8.277049286e-05;
  constexpr double p4 = 0.9062562627;
  constexpr double p5 = 0.09150303166;
  constexpr double p6 = 0.9099517204;
  constexpr double p7 = 596.3148142;
  const double yp = dev_pow(y, p4);
  const double inner = (p1 + p2 * yp) / (1.0 + p3 * yp);
  const double value = p7 * (dev_pow(inner, p5) - p6);
  return value > 0.0 ? value : 0.0;
}

__device__ void codes_to_nominal(const Pu21GpuHdr& h, double y, double cb, double cr, double& yn,
                                 double& cbn, double& crn) {
  if (h.full_range) {
    const double maxv = static_cast<double>((1 << h.bit_depth) - 1);
    yn = y / maxv;
    cbn = cb / maxv - 0.5;
    crn = cr / maxv - 0.5;
    return;
  }
  const double scale = static_cast<double>(1 << (h.bit_depth - 8));
  yn = (y - 16.0 * scale) / (219.0 * scale);
  cbn = (cb - 128.0 * scale) / (224.0 * scale);
  crn = (cr - 128.0 * scale) / (224.0 * scale);
}

__device__ double sample_clamp(const uint16_t* data, int stride, int w, int h, int x, int y) {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x >= w) x = w - 1;
  if (y >= h) y = h - 1;
  return static_cast<double>(data[static_cast<size_t>(y) * stride + x]);
}

__device__ double bilinear_code(const uint16_t* data, int stride, int w, int h, double x, double y) {
  if (x < 0.0) x = 0.0;
  if (y < 0.0) y = 0.0;
  const double x_max = static_cast<double>(w - 1);
  const double y_max = static_cast<double>(h - 1);
  if (x > x_max) x = x_max;
  if (y > y_max) y = y_max;
  const int x0 = static_cast<int>(floor(x));
  const int y0 = static_cast<int>(floor(y));
  const int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
  const int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
  const double tx = x - static_cast<double>(x0);
  const double ty = y - static_cast<double>(y0);
  const double a = sample_clamp(data, stride, w, h, x0, y0);
  const double b = sample_clamp(data, stride, w, h, x1, y0);
  const double c = sample_clamp(data, stride, w, h, x0, y1);
  const double d = sample_clamp(data, stride, w, h, x1, y1);
  return (1.0 - tx) * (1.0 - ty) * a + tx * (1.0 - ty) * b + (1.0 - tx) * ty * c + tx * ty * d;
}

__device__ double nearest_code(const uint16_t* data, int stride, int w, int h, double x, double y) {
  const int ix = static_cast<int>(floor(x + 0.5));
  const int iy = static_cast<int>(floor(y + 0.5));
  return sample_clamp(data, stride, w, h, ix, iy);
}

__device__ void chroma_coord(const Pu21GpuHdr& h, int x, int y, double& cx, double& cy) {
  if (h.chroma == 2) {
    cx = static_cast<double>(x);
    cy = static_cast<double>(y);
    return;
  }
  if (h.chroma == 1) {
    cy = static_cast<double>(y);
    cx = h.loc == 1 ? (static_cast<double>(x) - 0.5) / 2.0 : static_cast<double>(x) / 2.0;
    return;
  }
  if (h.loc == 1) {
    cx = (static_cast<double>(x) - 0.5) / 2.0;
    cy = (static_cast<double>(y) - 0.5) / 2.0;
  } else if (h.loc == 2) {
    cx = static_cast<double>(x) / 2.0;
    cy = static_cast<double>(y) / 2.0;
  } else {
    cx = static_cast<double>(x) / 2.0;
    cy = (static_cast<double>(y) - 0.5) / 2.0;
  }
}

__device__ bool display_rgb(const Pu21GpuHdr& h, double y, double cb, double cr, double& r, double& g,
                            double& b) {
  const double rn = y + 2.0 * (1.0 - h.kr) * cr;
  const double bn = y + 2.0 * (1.0 - h.kb) * cb;
  const double gn = (y - h.kr * rn - h.kb * bn) / h.kg;
  if (h.transfer == 1) {
    r = pq_eotf(rn);
    g = pq_eotf(gn);
    b = pq_eotf(bn);
    if (h.pq_clip) {
      if (r > h.peak) r = h.peak;
      if (g > h.peak) g = h.peak;
      if (b > h.peak) b = h.peak;
    }
    return isfinite(r) && isfinite(g) && isfinite(b);
  }
  const double rs = hlg_inverse_oetf(rn);
  const double gs = hlg_inverse_oetf(gn);
  const double bs = hlg_inverse_oetf(bn);
  const double ys = h.kr * rs + h.kg * gs + h.kb * bs;
  if (!(ys > 0.0)) {
    r = g = b = h.black;
    return true;
  }
  // Same BT.2100 split as hdr_display_rgb. Lb = 0 stays on the previous expression.
  if (h.black == 0.0) {
    const double scale = h.peak * dev_pow(ys, h.gamma) / ys;
    r = rs * scale;
    g = gs * scale;
    b = bs * scale;
  } else {
    const double factor = (h.peak - h.black) * dev_pow(ys, h.gamma - 1.0);
    r = rs * factor + h.black;
    g = gs * factor + h.black;
    b = bs * factor + h.black;
  }
  return isfinite(r) && isfinite(g) && isfinite(b);
}

__global__ void encode_kernel(const uint16_t* ry, const uint16_t* ru, const uint16_t* rv, const uint16_t* dy,
                              const uint16_t* du, const uint16_t* dv, int w, int h, int cw, int ch,
                              Pu21GpuHdr href, Pu21GpuHdr hdist, int want_psnr, int want_luma, double* yref,
                              double* ydist, double* partials, int* clamped) {
  const int n = w * h;
  double local = 0.0;
  int local_bad = 0;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
       i += static_cast<int>(blockDim.x * gridDim.x)) {
    const int y = i / w;
    const int x = i - y * w;
    auto encode_side = [&](const Pu21GpuHdr& hdr, const uint16_t* yy, const uint16_t* uu, const uint16_t* vv,
                           double& er, double& eg, double& eb, double& ey) {
      double cx = 0, cy = 0;
      chroma_coord(hdr, x, y, cx, cy);
      const double cb = hdr.nearest ? nearest_code(uu, cw, cw, ch, cx, cy)
                                    : bilinear_code(uu, cw, cw, ch, cx, cy);
      const double cr = hdr.nearest ? nearest_code(vv, cw, cw, ch, cx, cy)
                                    : bilinear_code(vv, cw, cw, ch, cx, cy);
      double yn = 0, cbn = 0, crn = 0;
      codes_to_nominal(hdr, static_cast<double>(yy[i]), cb, cr, yn, cbn, crn);
      double rn = 0, gn = 0, bn = 0;
      display_rgb(hdr, yn, cbn, crn, rn, gn, bn);
      const double lum = hdr.kr * rn + hdr.kg * gn + hdr.kb * bn;
      int flag = 0;
      er = pu21_encode_d(rn, &flag);
      eg = pu21_encode_d(gn, &flag);
      eb = pu21_encode_d(bn, &flag);
      ey = pu21_encode_d(lum, &flag);
      if (flag) local_bad = 1;
    };
    double rr = 0, rg = 0, rb = 0, ryv = 0, dr = 0, dg = 0, db = 0, dyv = 0;
    encode_side(href, ry, ru, rv, rr, rg, rb, ryv);
    encode_side(hdist, dy, du, dv, dr, dg, db, dyv);
    if (want_psnr) {
      const double drr = rr - dr;
      const double dgg = rg - dg;
      const double dbb = rb - db;
      local += drr * drr + dgg * dgg + dbb * dbb;
    }
    if (want_luma) {
      yref[i] = ryv;
      ydist[i] = dyv;
    }
  }
  if (local_bad) atomicExch(clamped, 1);
  const double sum = block_sum_256(local);
  if (threadIdx.x == 0 && partials) partials[blockIdx.x] = sum;
}

__global__ void ssim_h(const double* a, const double* b, int w, int h, double* o0, double* o1, double* o2,
                       double* o3, double* o4) {
  __shared__ double sa[kThreads + 2 * kRadius];
  __shared__ double sb[kThreads + 2 * kRadius];
  const int y = static_cast<int>(blockIdx.y);
  const int x0 = static_cast<int>(blockIdx.x) * kThreads;
  const int tx = threadIdx.x;
  if (y >= h) return;
  for (int i = tx; i < kThreads + 2 * kRadius; i += kThreads) {
    int gx = x0 + i - kRadius;
    if (gx < 0) gx = 0;
    if (gx >= w) gx = w - 1;
    sa[i] = a[static_cast<size_t>(y) * w + gx];
    sb[i] = b[static_cast<size_t>(y) * w + gx];
  }
  __syncthreads();
  const int x = x0 + tx;
  if (x >= w) return;
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
#pragma unroll
  for (int t = 0; t < kTaps; ++t) {
    const double k = pu21_k64[t];
    const double av = sa[tx + t];
    const double bv = sb[tx + t];
    sx += k * av;
    sy += k * bv;
    sxx += k * av * av;
    syy += k * bv * bv;
    sxy += k * av * bv;
  }
  const size_t idx = static_cast<size_t>(y) * w + x;
  o0[idx] = sx;
  o1[idx] = sy;
  o2[idx] = sxx;
  o3[idx] = syy;
  o4[idx] = sxy;
}

__global__ void ssim_v(const double* m1s, const double* m2s, const double* s1s, const double* s2s,
                       const double* s12s, int w, int h, double c1, double c2, double* acc) {
  const int ow = w - 2 * kRadius;
  const int oh = h - 2 * kRadius;
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y);
  double score = 0.0;
  if (x < ow && y < oh) {
    const int cx = x + kRadius;
    const int cy = y + kRadius;
    double m1 = 0, m2 = 0, s1 = 0, s2 = 0, s12 = 0;
#pragma unroll
    for (int t = 0; t < kTaps; ++t) {
      int yy = cy + (t - kRadius);
      if (yy < 0) yy = 0;
      if (yy >= h) yy = h - 1;
      const double k = pu21_k64[t];
      const size_t idx = static_cast<size_t>(yy) * w + cx;
      m1 += k * m1s[idx];
      m2 += k * m2s[idx];
      s1 += k * s1s[idx];
      s2 += k * s2s[idx];
      s12 += k * s12s[idx];
    }
    const double v1 = s1 - m1 * m1;
    const double v2 = s2 - m2 * m2;
    const double v12 = s12 - m1 * m2;
    const double cs = (2.0 * v12 + c2) / (v1 + v2 + c2);
    const double lum = (2.0 * m1 * m2 + c1) / (m1 * m1 + m2 * m2 + c1);
    score = lum * cs;
  }
  const double sum = block_sum_256(score);
  if (threadIdx.x == 0 && y < oh) atomicAdd(acc, sum);
}

__global__ void ms_h(const double* a, const double* b, int w, int h, double* ox, double* oy, double* oxx,
                     double* oyy, double* oxy) {
  const int ow = w - (kTaps - 1);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= ow || y >= h) return;
  const double* ra = a + static_cast<size_t>(y) * w + x;
  const double* rb = b + static_cast<size_t>(y) * w + x;
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
#pragma unroll
  for (int t = 0; t < kTaps; ++t) {
    const double k = pu21_k64[t];
    const double av = ra[t];
    const double bv = rb[t];
    sx += k * av;
    sy += k * bv;
    sxx += k * av * av;
    syy += k * bv * bv;
    sxy += k * av * bv;
  }
  const size_t idx = static_cast<size_t>(y) * ow + x;
  ox[idx] = sx;
  oy[idx] = sy;
  oxx[idx] = sxx;
  oyy[idx] = syy;
  oxy[idx] = sxy;
}

__global__ void ms_v(const double* hx, const double* hy, const double* hxx, const double* hyy, const double* hxy,
                     int stride, int h, double c1, double c2, double* acc_cs, double* acc_ssim) {
  const int ow = stride;
  const int oh = h - (kTaps - 1);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y);
  double cs_v = 0.0;
  double ss_v = 0.0;
  if (x < ow && y < oh) {
    double mx = 0, my = 0, xx = 0, yy = 0, xy = 0;
#pragma unroll
    for (int t = 0; t < kTaps; ++t) {
      const double k = pu21_k64[t];
      const size_t idx = static_cast<size_t>(y + t) * stride + x;
      mx += k * hx[idx];
      my += k * hy[idx];
      xx += k * hxx[idx];
      yy += k * hyy[idx];
      xy += k * hxy[idx];
    }
    const double v1 = xx - mx * mx;
    const double v2 = yy - my * my;
    const double v12 = xy - mx * my;
    const double cs = (2.0 * v12 + c2) / (v1 + v2 + c2);
    const double lum = (2.0 * mx * my + c1) / (mx * mx + my * my + c1);
    cs_v = cs;
    ss_v = lum * cs;
  }
  const double sum_cs = block_sum_256(cs_v);
  const double sum_ss = block_sum_256(ss_v);
  if (threadIdx.x == 0 && y < oh) {
    atomicAdd(acc_cs, sum_cs);
    atomicAdd(acc_ssim, sum_ss);
  }
}

__global__ void ms_pool(const double* src, int w, int h, double* dst, int ow, int oh) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= ow || y >= oh) return;
  const int pad_h = h & 1;
  const int pad_w = w & 1;
  double sum = 0.0;
#pragma unroll
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const int sy = y * 2 + dy - pad_h;
      const int sx = x * 2 + dx - pad_w;
      if (sy >= 0 && sy < h && sx >= 0 && sx < w) sum += src[static_cast<size_t>(sy) * w + sx];
    }
  }
  dst[static_cast<size_t>(y) * ow + x] = sum * 0.25;
}

int pool_dim(int n) {
  const int pad = n & 1;
  return (n + 2 * pad - 2) / 2 + 1;
}

void pack_plane(const uint16_t* src, int stride, int w, int h, uint16_t* dst) {
  for (int y = 0; y < h; ++y) {
    std::memcpy(dst + static_cast<size_t>(y) * w, src + static_cast<size_t>(y) * stride, w * sizeof(uint16_t));
  }
}

Pu21GpuHdr pack_hdr(const HdrEffective& hdr, const VideoFormat& fmt) {
  Pu21GpuHdr h;
  h.transfer = hdr.side.transfer == HdrTransfer::Pq ? 1 : 2;
  h.full_range = hdr.side.range == HdrRange::Full ? 1 : 0;
  h.bit_depth = fmt.bit_depth;
  h.chroma = fmt.chroma == ChromaFormat::YUV444 ? 2 : fmt.chroma == ChromaFormat::YUV422 ? 1 : 0;
  h.loc = hdr.chroma_loc == HdrChromaLoc::Center ? 1 : hdr.chroma_loc == HdrChromaLoc::TopLeft ? 2 : 0;
  h.nearest = hdr.chroma_up == HdrChromaUp::Nearest ? 1 : 0;
  h.pq_clip = hdr.pq_clip ? 1 : 0;
  h.peak = hdr.peak_nits;
  h.black = hdr.black_nits;
  h.gamma = hdr.hlg_gamma;
  if (hdr.side.matrix == HdrMatrix::Bt709) {
    h.kr = 0.2126;
    h.kg = 0.7152;
    h.kb = 0.0722;
  } else {
    h.kr = 0.2627;
    h.kg = 0.6780;
    h.kb = 0.0593;
  }
  return h;
}

void check_launch(const char* what) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel ") + what + ": " + cudaGetErrorString(e));
  }
}

}  // namespace

std::string cuda_score_pu21(const Frame& ref, const Frame& dist, const VideoFormat& fmt,
                            const HdrEffective& ref_hdr, const HdrEffective& dist_hdr, bool want_psnr,
                            bool want_ssim, bool want_msssim, Pu21FrameScore& out) {
  out = Pu21FrameScore{};
  if (!want_psnr && !want_ssim && !want_msssim) return {};
  if (fmt.bit_depth != 8 && fmt.bit_depth != 10) return "GPU PU21 supports 8-bit and 10-bit only";
  if ((want_ssim || want_msssim) && (fmt.width < 11 || fmt.height < 11)) {
    return "PU21-SSIM and PU21-MS-SSIM need at least 11x11 luma";
  }
  if (fmt.chroma != ChromaFormat::YUV444 && (fmt.width % 2 || (fmt.chroma == ChromaFormat::YUV420 && fmt.height % 2))) {
    return "PU21 chroma geometry is not even";
  }
  auto usable = [](const HdrEffective& h) {
    const bool transfer = h.side.transfer == HdrTransfer::Pq || h.side.transfer == HdrTransfer::Hlg;
    const bool matrix = h.side.matrix == HdrMatrix::Bt709 || h.side.matrix == HdrMatrix::Bt2020Ncl;
    const bool range = h.side.range == HdrRange::Limited || h.side.range == HdrRange::Full;
    return transfer && matrix && range;
  };
  if (!usable(ref_hdr) || !usable(dist_hdr)) return "no display transform for this transfer";

  init_once();
  auto& c = ctx();
  const int w = fmt.width;
  const int h = fmt.height;
  const int cw = fmt.chroma_width();
  const int ch = fmt.chroma_height();
  const size_t ny = static_cast<size_t>(w) * h;
  const size_t nc = static_cast<size_t>(cw) * ch;
  const size_t packed = ny * 2 + nc * 4;
  if (c.pin_samples < packed) {
    if (c.pin) PU21_CHECK(cudaFreeHost(c.pin));
    c.pin = nullptr;
    PU21_CHECK(cudaMallocHost(&c.pin, packed * sizeof(uint16_t)));
    c.pin_samples = packed;
  }
  uint16_t* slot = c.pin;
  auto take = [&](size_t n) {
    uint16_t* p = slot;
    slot += n;
    return p;
  };
  uint16_t* pry = take(ny);
  uint16_t* pdy = take(ny);
  uint16_t* pru = take(nc);
  uint16_t* pdu = take(nc);
  uint16_t* prv = take(nc);
  uint16_t* pdv = take(nc);
  pack_plane(ref.y.ptr(), ref.y.stride, w, h, pry);
  pack_plane(dist.y.ptr(), dist.y.stride, w, h, pdy);
  pack_plane(ref.u.ptr(), ref.u.stride, cw, ch, pru);
  pack_plane(dist.u.ptr(), dist.u.stride, cw, ch, pdu);
  pack_plane(ref.v.ptr(), ref.v.stride, cw, ch, prv);
  pack_plane(dist.v.ptr(), dist.v.stride, cw, ch, pdv);

  grow_u16(c.y[0], c.y_cap[0], ny);
  grow_u16(c.y[1], c.y_cap[1], ny);
  grow_u16(c.u[0], c.u_cap[0], nc);
  grow_u16(c.u[1], c.u_cap[1], nc);
  grow_u16(c.v[0], c.v_cap[0], nc);
  grow_u16(c.v[1], c.v_cap[1], nc);
  const bool want_luma = want_ssim || want_msssim;
  if (want_luma) {
    grow_f64(c.luma[0], c.luma_cap[0], ny);
    grow_f64(c.luma[1], c.luma_cap[1], ny);
  }
  const int blocks = 1024;
  if (c.partial_cap < blocks) {
    if (c.partials) PU21_CHECK(cudaFree(c.partials));
    PU21_CHECK(cudaMalloc(&c.partials, blocks * sizeof(double)));
    c.partial_cap = blocks;
  }
  PU21_CHECK(cudaMemsetAsync(c.clamped, 0, sizeof(int), c.stream));
  auto copy = [&](uint16_t* dst, const uint16_t* src, size_t n) {
    PU21_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(uint16_t), cudaMemcpyHostToDevice, c.stream));
  };
  copy(c.y[0], pry, ny);
  copy(c.y[1], pdy, ny);
  copy(c.u[0], pru, nc);
  copy(c.u[1], pdu, nc);
  copy(c.v[0], prv, nc);
  copy(c.v[1], pdv, nc);

  encode_kernel<<<blocks, kThreads, 0, c.stream>>>(
      c.y[0], c.u[0], c.v[0], c.y[1], c.u[1], c.v[1], w, h, cw, ch, pack_hdr(ref_hdr, fmt),
      pack_hdr(dist_hdr, fmt), want_psnr ? 1 : 0, want_luma ? 1 : 0, want_luma ? c.luma[0] : nullptr,
      want_luma ? c.luma[1] : nullptr, want_psnr ? c.partials : nullptr, c.clamped);
  check_launch("pu21_encode");

  std::vector<double> host_partials(want_psnr ? blocks : 0);
  if (want_psnr) {
    PU21_CHECK(cudaMemcpyAsync(host_partials.data(), c.partials, blocks * sizeof(double), cudaMemcpyDeviceToHost,
                               c.stream));
  }
  const double c1 = (0.01 * kPuPeak) * (0.01 * kPuPeak);
  const double c2 = (0.03 * kPuPeak) * (0.03 * kPuPeak);
  if (want_ssim) {
    const size_t moments = 5 * ny;
    grow_f64(c.work, c.work_cap, moments);
    PU21_CHECK(cudaMemsetAsync(c.acc, 0, sizeof(double), c.stream));
    double* o[5];
    for (int i = 0; i < 5; ++i) o[i] = c.work + static_cast<size_t>(i) * ny;
    ssim_h<<<dim3((w + kThreads - 1) / kThreads, h), kThreads, 0, c.stream>>>(c.luma[0], c.luma[1], w, h, o[0],
                                                                              o[1], o[2], o[3], o[4]);
    check_launch("pu21_ssim_h");
    const int ow = w - 2 * kRadius;
    const int oh = h - 2 * kRadius;
    ssim_v<<<dim3((ow + kThreads - 1) / kThreads, oh), kThreads, 0, c.stream>>>(o[0], o[1], o[2], o[3], o[4], w, h,
                                                                               c1, c2, c.acc);
    check_launch("pu21_ssim_v");
  }
  const int scales = want_msssim ? msssim_scales(h, w) : 0;
  if (want_msssim && scales < 1) return "PU21-MS-SSIM plane is too small";
  if (want_msssim) {
    const size_t moments = 5 * ny;
    grow_f64(c.work, c.work_cap, moments);
    grow_f64(c.pyr, c.pyr_cap, ny * 2);
    PU21_CHECK(cudaMemsetAsync(c.acc + 1, 0, 10 * sizeof(double), c.stream));
    const double* src_a = c.luma[0];
    const double* src_b = c.luma[1];
    bool src_is_luma = true;
    int cw_s = w;
    int ch_s = h;
    for (int scale = 0; scale < scales; ++scale) {
      const int ow = cw_s - (kTaps - 1);
      const int oh = ch_s - (kTaps - 1);
      double* maps[5];
      for (int i = 0; i < 5; ++i) maps[i] = c.work + static_cast<size_t>(i) * ny;
      dim3 block(16, 16);
      ms_h<<<dim3((ow + 15) / 16, (ch_s + 15) / 16), block, 0, c.stream>>>(src_a, src_b, cw_s, ch_s, maps[0],
                                                                           maps[1], maps[2], maps[3], maps[4]);
      check_launch("pu21_ms_h");
      double* acc_cs = c.acc + 1 + scale * 2;
      double* acc_ss = acc_cs + 1;
      ms_v<<<dim3((ow + kThreads - 1) / kThreads, oh), kThreads, 0, c.stream>>>(
          maps[0], maps[1], maps[2], maps[3], maps[4], ow, ch_s, c1, c2, acc_cs, acc_ss);
      check_launch("pu21_ms_v");
      if (scale + 1 == scales) break;
      const int nw = pool_dim(cw_s);
      const int nh = pool_dim(ch_s);
      double* dst_a = src_is_luma ? c.pyr : c.luma[0];
      double* dst_b = src_is_luma ? c.pyr + ny : c.luma[1];
      ms_pool<<<dim3((nw + 15) / 16, (nh + 15) / 16), block, 0, c.stream>>>(src_a, cw_s, ch_s, dst_a, nw, nh);
      ms_pool<<<dim3((nw + 15) / 16, (nh + 15) / 16), block, 0, c.stream>>>(src_b, cw_s, ch_s, dst_b, nw, nh);
      check_launch("pu21_ms_pool");
      src_a = dst_a;
      src_b = dst_b;
      src_is_luma = !src_is_luma;
      cw_s = nw;
      ch_s = nh;
    }
  }
  int host_clamped = 0;
  double h_acc[12] = {};
  if (want_ssim || want_msssim) {
    PU21_CHECK(cudaMemcpyAsync(h_acc, c.acc, 12 * sizeof(double), cudaMemcpyDeviceToHost, c.stream));
  }
  PU21_CHECK(cudaMemcpyAsync(&host_clamped, c.clamped, sizeof(int), cudaMemcpyDeviceToHost, c.stream));
  PU21_CHECK(cudaStreamSynchronize(c.stream));

  out.clamped = host_clamped != 0;
  if (want_psnr) {
    long double acc = 0.0;
    for (double part : host_partials) acc += static_cast<long double>(part);
    out.mse = static_cast<double>(acc / static_cast<long double>(ny * 3));
    if (out.mse == 0.0) out.psnr = std::numeric_limits<double>::infinity();
    else out.psnr = 10.0 * std::log10((kPuPeak * kPuPeak) / out.mse);
  }
  if (want_ssim) {
    const double nwin = static_cast<double>(w - 2 * kRadius) * static_cast<double>(h - 2 * kRadius);
    out.ssim = h_acc[0] / nwin;
  }
  if (want_msssim) {
    int cw_s = w;
    int ch_s = h;
    double score = 1.0;
    for (int scale = 0; scale < scales; ++scale) {
      const double nwin =
          static_cast<double>(cw_s - (kTaps - 1)) * static_cast<double>(ch_s - (kTaps - 1));
      const double cs = h_acc[1 + scale * 2] / nwin;
      const double ssim = h_acc[1 + scale * 2 + 1] / nwin;
      const double term = std::max(0.0, (scale + 1 == scales) ? ssim : cs);
      score *= std::pow(term, kMsssimWeight[scale]);
      if (scale + 1 < scales) {
        cw_s = pool_dim(cw_s);
        ch_s = pool_dim(ch_s);
      }
    }
    out.msssim = score;
  }
  return {};
}

}  // namespace pixvqa
